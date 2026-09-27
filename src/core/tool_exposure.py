"""工具暴露策略引擎 — 根据意图置信度分层暴露工具 Schema。

Phase 6 新增模块，核心功能：
1. 渐进式工具暴露：根据意图置信度从推荐集 → 扩展集 → 全量集
2. Schema 动态优先级标注：在 description 前添加 [推荐]/[备选] 引导模型
3. 工具依赖自动解析：确保依赖工具不被意外过滤
4. 自动回退：连续失败时自动升级到更大工具集

设计原则："引导而非限制，渐进而非决断"
"""

from __future__ import annotations

import copy
import json
import logging
from typing import Any

from src.core.prompts import (
    INTENT_PRIORITY_MAP,
    INTENT_TOOL_MAPPING,
    LOCAL_CTX_PROVIDERS,
    IntentResult,
)
from src.core.tool_retrieval import ToolRetrievalIndex

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Schema 确定性排序（前缀缓存 V3 · R1）
# ------------------------------------------------------------------

def sort_schemas_for_local(schemas: list[dict], model_cfg) -> list[dict]:
    """【前缀缓存 V3 · R1】本地 provider 下 tools schema 按函数名确定性排序。

    llama-server 前缀缓存要求连续请求的 tools 块逐字节一致；registry 输出序
    依赖 dict 插入序与懒加载迁移（registry.py L519-568：懒加载工具首次实例化
    时从 _lazy_tools 删除并追加到 _tools 末尾），cron 多步任务中途实例化懒工具
    会改变后续步骤的 tools 块字节序（实测：倒序即全量重算 8004 tokens）。

    仅 model_cfg.provider ∈ LOCAL_CTX_PROVIDERS 时排序；云端原样返回——
    agent._resolve_effective_model_key 按 schema 顺序 first-match-wins 解析
    preferred_model（当前因 split("_")[0] 截断实为死代码，门控为防御），
    云端排序可能改变模型选择结果。
    """
    if getattr(model_cfg, "provider", "") not in LOCAL_CTX_PROVIDERS:
        return schemas
    return sorted(schemas, key=lambda s: s.get("function", {}).get("name", ""))


# ------------------------------------------------------------------
# 会话内 schema 冻结（前缀缓存 V3 · R2）
# ------------------------------------------------------------------

_PRIORITY_PREFIXES = ("[推荐] ", "[备选] ", "[禁用] ")


def _strip_priority_annotation(schema: dict) -> dict:
    """剥离 description 前的优先级标注（仅标注时深拷贝，否则返回原对象）。"""
    desc = schema.get("function", {}).get("description", "")
    for prefix in _PRIORITY_PREFIXES:
        if desc.startswith(prefix):
            schema = copy.deepcopy(schema)
            schema["function"]["description"] = desc[len(prefix):]
            return schema
    return schema


def apply_schema_freeze(
    schemas: list[dict],
    *,
    session,
    model_cfg,
    sp_text: str,
) -> tuple[list[dict], int]:
    """【前缀缓存 V3 · R2】会话内 schema 集合冻结（仅本地 provider）。

    首轮捕获最终发送集（剥离标注后）为会话快照（Session._frozen_schemas，
    随会话对象回收）；后续轮次输出 = 冻结区（快照，字节稳定）+ 增量区
    （当轮新增工具按当前序追加尾部），冻结区永不重排/删除，保证 tools 块
    跨意图前缀逐字节一致（实测意图切换轮全量重算 57.5s → 冻结后仅算增量）。

    标注处理（偏离 V3.2 评审原裁决"标注集合变化整快照重置"）：意图切换
    必然同时改变标注集，按原裁决 R2 对主目标场景失效；本地路径 P1 推荐
    集收敛已承担选择引导，标注边际价值低，故统一剥离换取前缀稳定。代价：
    A3 超预算裁剪时标注分档退化为未标注档（CORE_TOOLS/tool_info 保护集
    不依赖标注，不受影响）；云端不受影响。

    上限口径（与 A3 同公式）：冻结区估算 > 当轮 budget → 快照重置为当轮
    集（接受当轮断裂）；冻结区+增量区超 budget → 增量区尾部丢弃（跳过
    装不下的，保持确定性）。

    Returns:
        (发送集, 发送集 token 估算)；非本地 provider 原样返回 (schemas, 0)
    """
    if getattr(model_cfg, "provider", "") not in LOCAL_CTX_PROVIDERS:
        return schemas, 0
    if session is None or not schemas:
        return schemas, 0

    from src.core import token_utils as _tu
    from src.core.prompts import TOOLS_EST_SAFETY

    def _est(items) -> int:
        return sum(
            int(_tu.estimate_tokens(
                [{"role": "system", "content": json.dumps(t, ensure_ascii=False)}]
            ) * TOOLS_EST_SAFETY)
            for t in items
        )

    # 当轮集剥离标注，与快照同口径比对
    current = [_strip_priority_annotation(s) for s in schemas]

    frozen = getattr(session, "_frozen_schemas", None)
    if not frozen:
        session._frozen_schemas = current
        session._frozen_schemas_tokens = _est(current)
        logger.info(
            "schema_freeze 捕获快照: schemas=%d tokens≈%d session=%s",
            len(current), session._frozen_schemas_tokens,
            getattr(session, "id", "?"),
        )
        return current, session._frozen_schemas_tokens

    frozen_names = {s.get("function", {}).get("name", "") for s in frozen}
    additions = [
        s for s in current
        if s.get("function", {}).get("name", "") not in frozen_names
    ]

    window = model_cfg.context_window
    sp_tokens = _tu.estimate_tokens([{"role": "system", "content": sp_text}])
    headroom = max(2048, int(window * 0.10))
    budget = window - sp_tokens - model_cfg.max_tokens - headroom

    frozen_tokens = session._frozen_schemas_tokens or _est(frozen)
    if frozen_tokens > budget:
        # 冻结区本身超当轮预算（SP 浮动/窗口变小）→ 快照重置为当轮集
        session._frozen_schemas = current
        session._frozen_schemas_tokens = _est(current)
        logger.warning(
            "schema_freeze 冻结区超预算重置: frozen≈%d > budget=%d，快照重建为当轮 %d schemas",
            frozen_tokens, budget, len(current),
        )
        return current, session._frozen_schemas_tokens

    # 增量区尾部追加，跳过装不下预算的新增（冻结区不动）
    result = list(frozen)
    result_tokens = frozen_tokens
    for s in additions:
        sz = _est([s])
        if result_tokens + sz > budget:
            logger.info(
                "schema_freeze 增量超预算跳过: tool=%s tokens≈%d",
                s.get("function", {}).get("name", ""), sz,
            )
            continue
        result.append(s)
        result_tokens += sz

    if additions:
        logger.info(
            "schema_freeze 命中: 冻结区=%d 增量区=%d 发送=%d tokens≈%d",
            len(frozen), len(additions), len(result), result_tokens,
        )
    return result, result_tokens


# ------------------------------------------------------------------
# Schema 动态优先级标注
# ------------------------------------------------------------------

def annotate_schema_priority(
    schemas: list[dict[str, Any]],
    intent_result: Any,
) -> list[dict[str, Any]]:
    """根据意图在 Schema description 前添加优先级标注。

    不删除任何工具，只在 description 前添加 [推荐] / [备选] / [禁用] 前缀，
    引导模型优先选择相关工具。

    Args:
        schemas: 原始 function calling schema 列表
        intent_result: 意图识别结果（支持对象或字典格式）

    Returns:
        标注后的 schema 列表（深拷贝，不修改原始数据）
    """
    # Phase 6+: LLM 模式优先使用 LLM 返回的推荐/禁用信息
    llm_recommended = set()
    llm_forbidden = set()
    if hasattr(intent_result, 'llm_recommended_tools'):
        llm_recommended = set(intent_result.llm_recommended_tools or [])
        llm_forbidden = set(intent_result.llm_forbidden_tools or [])

    if llm_recommended or llm_forbidden:
        # LLM 模式标注
        annotated: list[dict[str, Any]] = []
        for schema in schemas:
            func_info = schema.get("function", {})
            func_name = func_info.get("name", "")
            tool_name = _extract_tool_name(func_name)

            if tool_name in llm_recommended:
                schema = copy.deepcopy(schema)
                desc = schema["function"].get("description", "")
                schema["function"]["description"] = f"[推荐] {desc}"
            elif tool_name in llm_forbidden:
                schema = copy.deepcopy(schema)
                desc = schema["function"].get("description", "")
                schema["function"]["description"] = f"[禁用] {desc}"

            annotated.append(schema)
        return annotated

    # 规则模式标注（原有逻辑）
    # 支持对象和字典两种格式
    if hasattr(intent_result, 'primary_intent'):
        primary_intent = intent_result.primary_intent
    elif isinstance(intent_result, dict):
        primary_intent = intent_result.get("primary_intent", "")
    else:
        primary_intent = ""

    if not primary_intent:
        return schemas

    priority = INTENT_PRIORITY_MAP.get(primary_intent, {})
    recommended = set(priority.get("recommended", []))
    alternative = set(priority.get("alternative", []))

    if not recommended and not alternative:
        return schemas

    annotated: list[dict[str, Any]] = []
    for schema in schemas:
        func_info = schema.get("function", {})
        func_name = func_info.get("name", "")

        # 从 func_name 提取工具名（格式: tool_name_action_name）
        tool_name = _extract_tool_name(func_name)

        if tool_name in recommended:
            schema = copy.deepcopy(schema)
            desc = schema["function"].get("description", "")
            schema["function"]["description"] = f"[推荐] {desc}"
        elif tool_name in alternative:
            schema = copy.deepcopy(schema)
            desc = schema["function"].get("description", "")
            schema["function"]["description"] = f"[备选] {desc}"

        annotated.append(schema)

    return annotated


def _extract_tool_name(func_name: str) -> str:
    """从函数名中提取工具名。

    函数名格式为 tool_name_action_name，但工具名本身可能包含下划线
    （如 browser_use, app_control, voice_input 等）。
    """
    # 已知的多下划线工具名前缀
    known_prefixes = [
        "mcp_browserbase-csdn", "mcp_browserbase",
        "browser_use", "app_control", "voice_input", "voice_output",
        "weclaw_browser",  # WeClaw 内置浏览器（validate check_4 遗漏补齐）
        "datetime_tool", "chat_history", "doc_generator",
        "image_generator", "python_runner", "tool_info",
        "knowledge_rag", "batch_paper_analyzer", "user_profile",
        "course_schedule", "mind_map", "resume_builder", "data_processor",
        "coding_assistant", "data_visualization", "speech_to_text",
        "format_converter", "id_photo", "literature_search", "pdf_tool", "pdf_generator",
        "gif_maker", "ai_writer", "education_tool", "ppt_generator",
        "ui_generator",  # UI 生成
        "financial_report", "contract_generator", "family_member", "meal_menu",
        "document_scanner",  # 高拍仪文档扫描
        "music_player",  # 歌曲库
        "listen_radio",  # 听电台
        "watch_tv",  # 看电视（央视/卫视/B站新闻直播）
        "english_conversation",  # 英语口语练习
        "family_milestone",  # 家庭大事记
        "todo",  # 待办事项
        "remote_file_share",  # 远程文件分享
        "daily_task",  # 每日任务
        "medication",  # 用药管理
        "family_album",  # 家庭相册
        "exercise_plan",  # 运动计划
        "social_circle",  # 社交圈
        "storage_organizer",  # 收纳管理
        "insurance",  # 保险方案
        "emergency_supply",  # 应急物资
        "media_capture",  # 媒体捕获
        "experience_recall",  # 经验回忆
        "poetry",  # 古诗词库
        "stock_query",  # 股票查询
        "quant_trading",  # 量化交易
        "system_monitor",  # 系统监控
        # 学术工具
        "research_landscape",
        "research_lineage",
        "research_lookup",
        "idea_migration",
        "stock_photo",
        "methodology_deconstructor",
        "citation_storyteller",
        "journal_intelligence",
        "contrarian_finder",
        "paper_lifecycle",
        "qualitative_analysis",  # 定性分析（Nvivo风格）
        "local_paper_search",
        "oss_admin",         # OSS 管理（索引重建、扫描、同步、健康检查）
        "oss_pdf_search",      # OSS PDF搜索（分层下载）
        "oss_pdf_download",    # OSS PDF下载（分层下载）
        "crawlee_tool",  # 批量网页爬取
        # 自我反思工具（P0/P1/P2）
        "tool_audit",     # 工具调用审计
        "log_viewer",     # 日志查看器
        "codebase_search",  # 代码库搜索
        "self_control",   # 自我控制
        "desktop_control",  # 桌面控制（v9.4.0：主题/模型/会话/窗口/语音指令化）
        "desktop_data",  # 桌面数据查询（PWA 直通：生成空间/会话/邮件/学术项目）
        # v2 新增工具
        "enterprise_query",  # 工商企业查询
        "baike_lookup",      # 百度百科
        # v3 新增工具
        "medical_lookup",    # 医学知识检索
        "bilibili_search",   # 哔哩哔哩视频检索（否则被默认截断为 "bilibili"）
        # 其他多下划线工具（同步 validate_tool_chain.py）
        "fred_query",        # FRED经济数据查询
        "literature_review", # 文献综述
        "pc_maintenance",    # 系统维护（健康总览/垃圾清理/网络诊断）
        "mcp_registry",      # MCP 注册表检索（否则 mcp_registry_search 被截断为 mcp）
        "mcp_book-writer",   # MCP 写书桥接工具（否则 mcp_book-writer_xxx 被截断为 mcp，_extract_tool_name 归组失效）
        "workflow_queue",    # 工作流队列（v7.0.0：批量/定时执行多条指令）
        # OA 网关工具（融合方案 Phase 1）
        "oa_auth",           # OA 账号绑定/校验
        "oa_query",          # OA 只读查询
        "oa_action",         # OA 写操作
        # 菜谱库（HowToCook 集成 v1.0）
        "recipe_library",    # 菜谱检索（经 meal_menu 代理触达）
        # Phase 2.2② 补齐（validate check_4 修复：schema 标注截断防漂移）
        "attachment_search",  # 历史附件检索
        "study_solver",      # 试卷作业解答
        "audio_processor",   # 音频格式转换/剪辑/混音
        "sound_library",     # 声音素材库
        "music_generator",   # AI 音乐生成（ACE-Step 1.5，需本地 API 服务）
        "nutrition_query",   # USDA 食物营养查询
        "family_site",       # 家庭年度回顾/家庭叙事
        "education_family",  # 家庭子女教育
        # 资源库第一批（有意不加 EXTENDED_TOOLS，保持中置信度曝光集合不变）
        "chinese_dictionary",  # 国学字典（成语/歇后语/汉字/词语）
        "lunar_calendar",      # 农历黄历
        "english_vocab",       # 英语词汇
        "pinyin_dict",         # 拼音字典
        "daily_quote",         # 每日一句
        "dev_progress",        # v11.15.0：GitHub 协作进度治理（多下划线，防被截断为 dev）
        "capsule",             # 时间胶囊（多 action 名以 capsule_ 开头）
    ]
    for prefix in known_prefixes:
        if func_name.startswith(prefix + "_") or func_name == prefix:
            return prefix

    # 默认：取第一个下划线前的部分
    return func_name.split("_")[0] if "_" in func_name else func_name


# ------------------------------------------------------------------
# 渐进式工具暴露引擎
# ------------------------------------------------------------------

class ToolExposureEngine:
    """工具暴露策略引擎 — 根据意图置信度分层暴露工具。

    三层工具集：
    - 推荐工具集（~10个）：核心工具 + 意图相关工具（高置信度时使用）
    - 扩展工具集（~20个）：推荐 + 扩展工具（中置信度时使用）
    - 全量工具集（35+个）：所有已注册工具（低置信度 / 回退时使用）

    风险控制：
    - 核心工具（shell/file/screen/search）始终保留
    - 连续失败 >= 2 次自动升级到更大工具集
    - 支持配置化开关，可随时关闭
    """

    # 核心工具（始终保留在任何工具集中）
    CORE_TOOLS: set[str] = {"shell", "file", "screen", "search"}

    # 扩展工具（中置信度时额外包含）
    EXTENDED_TOOLS: set[str] = {
        "browser", "browser_use", "notify", "clipboard",
        "app_control", "calculator", "datetime_tool",
        "family_album",  # 家庭相册
        "stock_query",  # 股票行情
        "crawlee_tool",  # 批量网页爬取
        "wechat",             # v4.x: 微信消息
        "remote_file_share",  # v4.x: 远程文件分享
        # 自我反思工具（归入 recommended 层级）
        "tool_audit", "log_viewer", "codebase_search", "self_control",
        "desktop_control",  # v9.4.0：桌面控制（中置信度可见）
        "desktop_data",  # 桌面数据查询（中置信度可见，与 desktop_control 同模式）
        "experience_recall",  # 经验回忆
        "capsule",  # 时间胶囊（中置信度可见，EXTENDED 档）
        # v2 新增工具
        "enterprise_query",  # 工商企业查询
        "baike_lookup",      # 百度百科
        "bilibili_search",   # 哔哩哔哩视频检索（中置信度可见）
        "pc_maintenance",    # 系统维护（中置信度可见，非高频核心）
        "dev_progress",      # v11.15.0：GitHub 协作进度治理（中置信度可见）
    }

    # 工具集层级
    TIER_RECOMMENDED = "recommended"  # 推荐集
    TIER_EXTENDED = "extended"        # 扩展集
    TIER_FULL = "full"                # 全量集

    def __init__(
        self,
        tool_registry: Any,
        *,
        enabled: bool = True,
        enable_annotation: bool = True,
        failures_to_upgrade: int = 2,
    ) -> None:
        """初始化。

        Args:
            tool_registry: ToolRegistry 实例
            enabled: 是否启用渐进式暴露（False 则始终全量）
            enable_annotation: 是否启用 Schema 优先级标注
            failures_to_upgrade: 连续失败多少次后自动升级工具集
        """
        self._registry = tool_registry
        self._enabled = enabled
        self._enable_annotation = enable_annotation
        self._failures_to_upgrade = failures_to_upgrade

        # 运行时状态（每次对话重置）
        self._consecutive_failures: int = 0
        self._forced_tier: str | None = None  # 被强制升级的层级
        self._deviation_count: int = 0  # 工具调用偏离次数（被前置验证拒绝）

        # G4 AdaptiveHarness: 基于历史成功模式的自适应 tier 覆盖（intent -> tier）
        # 由 AdaptiveHarness 写入，遵循“只能放宽不能收窄”原则；默认为空（无自适应调整）
        self._adaptive_overrides: dict[str, str] = {}

        # 本轮生效 tier（由 _determine_tier 记录，供 _upgrade_tier 判定是否已在 FULL 档）
        self._effective_tier: str | None = None

        # 【修复2 · RAG-over-tools】FULL 兜底档检索式暴露：配置 + 运行期状态。
        # 经 AppConfig 读取 [agent.tool_optimization.rag_fallback]（复用 domain_guides
        # 读取范式），引擎自读配置、不改 Agent.__init__ 签名；读取失败按保守默认降级。
        _rag = self._load_rag_config()
        self._rag_enabled: bool = _rag["enabled"]
        self._rag_top_k: int = _rag["top_k"]
        self._rag_min_score: float = _rag["min_score"]
        self._rag_include_extended_floor: bool = _rag["include_extended_floor"]
        self._rag_escalate_on_failure: bool = _rag["escalate_on_failure"]
        self._rag_index: ToolRetrievalIndex | None = None  # 懒建（首次 FULL 检索时构建）
        self._rag_escalate_full: bool = False  # 运行期升级：本轮 FULL 无界全量暴露

        # 【完善 LLM 意图模式】把 llm_recommended_tools 并入暴露集：配置 + 上限。
        # 经 AppConfig 读取 [agent.tool_optimization.llm_exposure]（复用 _load_rag_config
        # 范式），修复 round2 揭示的"LLM 点对工具却没暴露"GAP；读取失败按默认降级。
        _llm_exp = self._load_llm_exposure_config()
        self._llm_merge_recommended: bool = _llm_exp["merge_recommended"]
        self._llm_max_merge: int = _llm_exp["max_merge"]

    @staticmethod
    def _load_rag_config() -> dict[str, Any]:
        """读取 [agent.tool_optimization.rag_fallback]（缺省即回退契约默认值）。

        复用 domain_guides_enabled() 的 AppConfig 读取范式；任何读取/解析失败均按
        保守默认降级，保证配置缺失或损坏时引擎仍以确定行为工作。
        """
        defaults = {
            "enabled": True,
            "top_k": 15,
            "min_score": 0.0,
            "include_extended_floor": True,
            "escalate_on_failure": True,
        }
        try:
            from src.core.config import AppConfig

            cfg = AppConfig.load()
            base = "agent.tool_optimization.rag_fallback"
            return {
                "enabled": bool(cfg.get(f"{base}.enabled", defaults["enabled"])),
                "top_k": int(cfg.get(f"{base}.top_k", defaults["top_k"])),
                "min_score": float(cfg.get(f"{base}.min_score", defaults["min_score"])),
                "include_extended_floor": bool(
                    cfg.get(f"{base}.include_extended_floor",
                            defaults["include_extended_floor"])
                ),
                "escalate_on_failure": bool(
                    cfg.get(f"{base}.escalate_on_failure",
                            defaults["escalate_on_failure"])
                ),
            }
        except Exception as e:  # pragma: no cover - 配置读取失败降级路径
            logger.debug(
                "读取 [agent.tool_optimization.rag_fallback] 失败，按保守默认: %s", e,
            )
            return defaults

    @staticmethod
    def _load_llm_exposure_config() -> dict[str, Any]:
        """读取 [agent.tool_optimization.llm_exposure]（缺省即回退默认值）。

        【完善 LLM 意图模式】复用 _load_rag_config 的 AppConfig 读取范式；任何
        读取/解析失败均按默认降级，保证配置缺失或损坏时引擎仍以确定行为工作。
        """
        defaults = {"merge_recommended": True, "max_merge": 8}
        try:
            from src.core.config import AppConfig

            cfg = AppConfig.load()
            base = "agent.tool_optimization.llm_exposure"
            return {
                "merge_recommended": bool(
                    cfg.get(f"{base}.merge_recommended", defaults["merge_recommended"])
                ),
                "max_merge": int(cfg.get(f"{base}.max_merge", defaults["max_merge"])),
            }
        except Exception as e:  # pragma: no cover - 配置读取失败降级路径
            logger.debug(
                "读取 [agent.tool_optimization.llm_exposure] 失败，按默认: %s", e,
            )
            return defaults

    def set_adaptive_override(self, intent: str, tier: str) -> None:
        """G4: 设置自适应 tier 覆盖（仅放宽不收窄）。"""
        tier_order = {self.TIER_RECOMMENDED: 0, self.TIER_EXTENDED: 1, self.TIER_FULL: 2}
        current = self._adaptive_overrides.get(intent)
        if current is None or tier_order.get(tier, 0) >= tier_order.get(current, 0):
            self._adaptive_overrides[intent] = tier

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def get_schemas(self, intent_result: IntentResult) -> list[dict[str, Any]]:
        """根据意图结果返回分层 Schema。

        Args:
            intent_result: detect_intent_with_confidence 的返回值

        Returns:
            function calling schema 列表
        """
        if not self._enabled:
            schemas = self._registry.get_all_schemas()
            if self._enable_annotation:
                schemas = annotate_schema_priority(schemas, intent_result)
            return schemas

        tier = self._determine_tier(intent_result)
        tool_names = self._get_tool_names_for_tier(tier, intent_result)

        # 依赖解析
        tool_names = self._resolve_dependencies(tool_names)

        logger.info(
            "工具暴露策略: tier=%s, confidence=%.2f, intent=%s, tools=%d",
            tier, intent_result.confidence, intent_result.primary_intent,
            len(tool_names),
        )

        schemas = self._registry.get_schemas_by_names(tool_names)

        if self._enable_annotation:
            schemas = annotate_schema_priority(schemas, intent_result)

        return schemas

    def recommended_tool_names(self, intent_result: IntentResult) -> set[str]:
        """推荐层工具名集合（核心工具 + 意图映射 + 依赖解析 + tool_info）。

        【上下文预算治理 V2 · P1】供本地小窗口模型在空意图轮次收敛暴露集
        （默认 FULL 全量 → 推荐集），不改变 _determine_tier 既有分级语义。
        """
        if not self._enabled:
            return set(self._registry.list_all_tool_names())
        names = self._get_tool_names_for_tier(self.TIER_RECOMMENDED, intent_result)
        return self._resolve_dependencies(names)

    def report_failure(self) -> tuple[str, str] | None:
        """报告工具调用失败，可能触发自动升级。

        Returns:
            如果发生层级升级，返回 (from_tier, to_tier)；否则返回 None
        """
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._failures_to_upgrade:
            return self._upgrade_tier()
        return None

    def report_success(self) -> None:
        """报告工具调用成功，重置连续失败计数。"""
        self._consecutive_failures = 0

    def report_deviation(self) -> None:
        """报告一次工具调用偏离（被前置验证拒绝）。

        连续偏离 >= 2 次时，强制收敛到推荐集，防止模型继续发散。
        偏离收敛优先级 > 失败升级优先级。
        """
        self._deviation_count += 1
        if self._deviation_count >= 2:
            # 强制收敛到推荐集
            self._forced_tier = self.TIER_RECOMMENDED
            logger.warning(
                "连续偏离 %d 次，工具集强制收敛到推荐集",
                self._deviation_count,
            )

    def reset(self) -> None:
        """重置状态（新对话开始时调用）。"""
        self._consecutive_failures = 0
        self._forced_tier = None
        self._deviation_count = 0
        # 【修复2】清本轮生效 tier 与运行期升级标志（per-task 语义）
        self._effective_tier = None
        self._rag_escalate_full = False

    @property
    def current_tier(self) -> str:
        """当前生效的工具集层级（调试用）。"""
        return self._forced_tier or "auto"

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _determine_tier(self, intent_result: IntentResult) -> str:
        """根据意图置信度和运行状态确定工具集层级。"""
        # 强制层级优先
        if self._forced_tier:
            self._effective_tier = self._forced_tier
            return self._forced_tier

        confidence = intent_result.confidence

        if confidence >= 0.8:
            tier = self.TIER_RECOMMENDED
        elif confidence >= 0.5:
            tier = self.TIER_EXTENDED
        else:
            tier = self.TIER_FULL

        # G4 AdaptiveHarness: 若存在基于历史成功模式的 adaptive 覆盖，则应用（仅放宽）。
        # 默认 _adaptive_overrides 为空，行为与原静态策略完全一致（向后兼容）。
        override = self._adaptive_overrides.get(intent_result.primary_intent)
        if override:
            _order = {self.TIER_RECOMMENDED: 0, self.TIER_EXTENDED: 1, self.TIER_FULL: 2}
            if _order.get(override, 0) >= _order.get(tier, 0):
                tier = override

        # 【修复2】记录本轮生效 tier，供 _upgrade_tier 判定是否已在 FULL 档
        self._effective_tier = tier
        return tier

    def _get_tool_names_for_tier(
        self,
        tier: str,
        intent_result: IntentResult,
    ) -> set[str]:
        """获取指定层级的工具名称集合。"""
        # 核心工具始终包含
        tools = set(self.CORE_TOOLS)

        if tier == self.TIER_FULL:
            # 【修复2 · RAG-over-tools】检索式兜底：
            # 灰度关闭 或 本轮已运行期升级 → 回退原全量暴露（一键回退契约 / 无界兜底）
            if not self._rag_enabled or self._rag_escalate_full:
                return self._registry.list_all_tool_names()
            # 保守并集：核心底线(已含) + tool_info + 弱意图映射 + 检索 Top-K。
            # tools 已在函数首部初始化为 CORE_TOOLS 副本，此处按配置扩充。
            if self._rag_include_extended_floor:
                tools.update(self.EXTENDED_TOOLS)
            for intent_name in intent_result.intents:
                tools.update(INTENT_TOOL_MAPPING.get(intent_name, []))
            if self._rag_index is None:
                self._rag_index = ToolRetrievalIndex(self._registry)
            # getattr 防御：真实 IntentResult 恒有 user_input，兼容测试 ad-hoc mock
            query = getattr(intent_result, "user_input", "") or ""
            retrieved = self._rag_index.retrieve(
                query, self._rag_top_k, self._rag_min_score,
            )
            tools.update(name for name, _ in retrieved)
            tools.add("tool_info")
            logger.info(
                "RAG 兜底检索: query_len=%d top_k=%d 命中=%d 并集=%d",
                len(query), self._rag_top_k, len(retrieved), len(tools),
            )
            # 【完善 LLM 意图模式】并入 LLM 推荐工具（防御性；LLM 模式 conf=0.9 实际
            # 落 RECOMMENDED 档不达此 FULL-RAG 分支，此处保证 tier 无关的一致语义）
            return self._augment_with_llm_recommended(tools, intent_result)

        # 意图相关工具
        for intent_name in intent_result.intents:
            intent_tools = INTENT_TOOL_MAPPING.get(intent_name, [])
            tools.update(intent_tools)

        if tier == self.TIER_EXTENDED:
            # 扩展集额外包含通用扩展工具
            tools.update(self.EXTENDED_TOOLS)

        # 始终包含工具信息查询
        tools.add("tool_info")

        # 【完善 LLM 意图模式 · 核心生效点】把 llm_recommended_tools 并入暴露集，
        # 修复 round2 GAP：LLM 模式 conf=0.9 落 RECOMMENDED 档，此前推荐工具只标注不
        # 暴露，导致"点对却调不到"。规则模式下 llm_recommended_tools 恒空 → 字节级 no-op。
        # 并入须在 get_schemas 的 _resolve_dependencies 之前，使推荐工具的依赖被自动解析。
        return self._augment_with_llm_recommended(tools, intent_result)

    def _resolve_dependencies(self, tool_names: set[str]) -> set[str]:
        """自动包含依赖工具（避免过滤掉内容源工具）。

        读取 tools.json 中的 dependencies.input_sources 字段。
        """
        all_tools = set(tool_names)
        for name in list(tool_names):
            cfg = self._registry.get_tool_config(name)
            deps = cfg.get("dependencies", {})
            # 兼容两种格式：列表 ["fredapi"] 或字典 {"input_sources": [...]}
            if isinstance(deps, dict):
                input_sources = deps.get("input_sources", [])
                all_tools.update(input_sources)
        return all_tools

    def _augment_with_llm_recommended(
        self, tool_names: set[str], intent_result: IntentResult,
    ) -> set[str]:
        """把 LLM 推荐工具并入暴露集（配置门控 / 已知名过滤 / forbidden 优先 / 上限）。

        【完善 LLM 意图模式】修复 round2 揭示的 GAP：LLM 模式 conf=0.9 落 RECOMMENDED
        档，暴露集仅 CORE ∪ 规则意图映射 ∪ tool_info，llm_recommended_tools 此前只用于
        标注/prompt/预校验，导致"LLM 点对工具却没暴露、模型调不到"（离线实测 GAP 47%，
        并入后覆盖率→100%）。规则模式下 llm_recommended_tools 恒空 → 字节级等价 no-op。

        护栏：① 配置总开关 _llm_merge_recommended；② 仅并入注册表已知名（幻觉名跳过 +
        日志）；③ forbidden 优先（推荐∩禁用不并入）；④ 上限 _llm_max_merge 防撑大。
        禁用工具的二次拦截由 get_schemas_by_names 的 is_tool_enabled 保证。
        """
        if not self._llm_merge_recommended:
            return tool_names
        recommended = getattr(intent_result, "llm_recommended_tools", None) or []
        if not recommended:
            return tool_names
        forbidden = set(getattr(intent_result, "llm_forbidden_tools", None) or [])
        known = set(self._registry.list_all_tool_names())
        merged = set(tool_names)
        added = 0
        skipped: list[str] = []
        for name in recommended:
            if added >= self._llm_max_merge:
                break
            if name in merged or name in forbidden:
                continue
            if name not in known:
                skipped.append(name)
                continue
            merged.add(name)
            added += 1
        if added or skipped:
            logger.info(
                "LLM 推荐并入暴露集: +%d 上限=%d 未知跳过=%s",
                added, self._llm_max_merge, skipped or "无",
            )
        return merged

    def _upgrade_tier(self) -> tuple[str, str] | None:
        """自动升级工具集层级。

        Returns:
            如果升级成功，返回 (from_tier, to_tier)；已经是最大集则返回 None
        """
        current = self._forced_tier or self.TIER_RECOMMENDED

        if current == self.TIER_RECOMMENDED:
            # 【修复2】本轮生效 tier 已是 FULL（低置信度自动 FULL，_forced_tier 仍为
            # None）：走 REC→EXT 阶梯会把自动 FULL 反向收窄到 EXTENDED，故 RAG 启用
            # 时直接在 FULL 档内升级为无界全量。RAG 关闭时严格保持原阶梯行为不变。
            if (
                self._rag_enabled
                and self._rag_escalate_on_failure
                and self._effective_tier == self.TIER_FULL
                and not self._rag_escalate_full
            ):
                self._rag_escalate_full = True
                self._forced_tier = self.TIER_FULL  # 固化 FULL，避免下轮波动回退
                logger.warning(
                    "连续失败 %d 次，FULL 档检索兜底升级为无界全量暴露",
                    self._consecutive_failures,
                )
                return (self.TIER_FULL, "full_unbounded")
            self._forced_tier = self.TIER_EXTENDED
            logger.warning(
                "连续失败 %d 次，工具集自动升级: %s → %s",
                self._consecutive_failures, current, self._forced_tier,
            )
            return (current, self._forced_tier)
        elif current == self.TIER_EXTENDED:
            self._forced_tier = self.TIER_FULL
            logger.warning(
                "连续失败 %d 次，工具集自动升级: %s → %s",
                self._consecutive_failures, current, self._forced_tier,
            )
            return (current, self._forced_tier)
        # current == TIER_FULL（已强制到 FULL）
        # 【修复2】检索式兜底启用时，FULL 再失败（多为检索未命中所需工具）→ 升级为
        # 无界全量暴露（绕过 Top-K）。复用既有 report_failure 生效轨道，无需 agent.py
        # 新增接线。RAG 关闭时保持原“FULL 即最大集”语义（return None）。
        if (
            self._rag_enabled
            and self._rag_escalate_on_failure
            and not self._rag_escalate_full
        ):
            self._rag_escalate_full = True
            logger.warning(
                "连续失败 %d 次，FULL 档检索兜底升级为无界全量暴露",
                self._consecutive_failures,
            )
            return (self.TIER_FULL, "full_unbounded")
        # TIER_FULL 已经是最大集，无需升级
        return None
