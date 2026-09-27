# -*- coding: utf-8 -*-
r"""N1 · P0-1/P0-3 —— 500 条口语 query 扩充集生成器（含客观 ref + 规则命中判定）。

方法学（回应评审 M1 样本量 + M4 循环性）：
  · ref **by construction**：每条 query 由"目标工具"反向生成，ref 即该工具（客观、非事后主观标注）。
  · 生成 LLM（deepseek-v4-flash）为每个目标工具产出 20 条**口语化、简短、不出现工具名/术语**
    的日常请求，且每条应唯一指向该工具。
  · rule_hit：用**当前生产规则引擎**复刻 before 臂 RECOMMENDED 暴露集，判定 ref 是否已被规则
    暴露（rule_hit=False → 该 query 是 gap 候选，即"规则漏、LLM 可能对"的战场）。透明记录，不挑样本。
  · 第二标注者 Kappa 由 n1_annotate_kappa.py 独立跑（盲标 → Cohen's Kappa 验证 ref 无歧义）。

产物：test_scripts/rag_experiments/n1_queries_500.json
运行：.\.venv\Scripts\python.exe test_scripts\rag_experiments\n1_expand_queries.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXP = Path(__file__).resolve().parent
OUT_JSON = EXP / "n1_queries_500.json"


def _load_env() -> None:
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env()
logging.basicConfig(level=logging.ERROR)
for _n in ("src", "LiteLLM", "litellm", "httpx", "asyncio", "openai"):
    logging.getLogger(_n).setLevel(logging.ERROR)

from src.core.intent_engine import IntentResult, detect_intent_with_confidence  # noqa: E402
from src.core.tool_exposure import ToolExposureEngine  # noqa: E402
from src.core.tool_retrieval import ToolRetrievalIndex  # noqa: E402
from src.models.registry import ModelRegistry  # noqa: E402
from src.tools.registry import ToolRegistry  # noqa: E402

GEN_MODEL = "deepseek-v4-flash"
GEN_TIMEOUT = 60.0
N_PER_TOOL = 20
INTENT_LLM_CONFIDENCE = 0.9  # 复刻 LLM 模式 conf → RECOMMENDED 档

# 目标工具（gap 友好：口语场景清晰、单一最适工具）。desc 供生成提示词。
TARGET_TOOLS: dict[str, str] = {
    "datetime_tool": "获取当前日期、时间、星期、几号",
    "calculator": "做数学计算、算账、四则运算",
    "browser": "打开网页、上网浏览、访问网站看内容",
    "clipboard": "复制、粘贴、拷贝一段文字或图片",
    "notify": "弹一个系统通知/提醒",
    "app_control": "打开/启动/关闭某个桌面应用程序",
    "stock_query": "查询股票价格、行情、涨跌",
    "email": "写一封邮件、发邮件、查看收件箱",
    "enterprise_query": "查某家公司的工商注册信息/背景",
    "bilibili_search": "在B站搜索视频",
    "pc_maintenance": "清理电脑垃圾、电脑变慢、系统维护体检",
    "family_album": "查看/管理家庭相册和照片",
    "baike_lookup": "查百度百科词条（人物、概念、事物）",
    "weather": "查天气、气温、要不要带伞",
    "music_player": "播放音乐、放首歌",
    "listen_radio": "听电台/广播节目",
    "watch_tv": "看电视直播/电视频道",
    "recipe_library": "查菜谱、今天做什么菜、怎么做某道菜",
    "todo": "记一条待办/任务、看待办清单",
    "diary": "写一篇日记、记录今天",
    "poetry": "生成/欣赏一首诗、按主题作诗",
    "health": "记录或查询健康数据（血压、体重、步数等）",
    "exercise_plan": "制定锻炼/健身计划",
    "ocr": "识别图片里的文字、把图片文字转出来",
    "medical_lookup": "查药品、疾病、医学常识",
}

GEN_PROMPT = (
    "你在为一个桌面 AI 助手构造**评测用的口语化用户指令**。\n"
    "目标工具功能：{desc}（工具内部名 {tool}，但**指令里绝不能出现工具名或英文/专业术语**）。\n"
    "请给出 {n} 条**普通用户随口会说**的中文指令，要求：\n"
    "1) 口语、简短（多为 4-15 字）、说法多样（疑问/祈使/省略主语都要有）；\n"
    "2) 每条都**唯一明确**地需要上面这个工具来完成（不能更适合别的工具）；\n"
    "3) 不要出现工具名、函数名、英文；不要编号、不要解释。\n"
    "只输出一个 JSON 数组，形如 [\"...\", \"...\"]，含 {n} 个字符串。"
)


def _parse_json_array(raw: str) -> list[str]:
    txt = raw.strip()
    if "```" in txt:
        m = re.search(r"```(?:json)?\s*(.*?)```", txt, re.S)
        if m:
            txt = m.group(1).strip()
    # 抓第一个 [ ... ]
    s, e = txt.find("["), txt.rfind("]")
    if s >= 0 and e > s:
        txt = txt[s:e + 1]
    arr = json.loads(txt)
    return [str(x).strip() for x in arr if str(x).strip()]


def _recommended_exposure(engine: ToolExposureEngine, q: str) -> set[str]:
    """复刻 before 臂（merge off）LLM 模式 conf=0.9 的 RECOMMENDED 暴露集。"""
    rule = detect_intent_with_confidence(q)
    ir = IntentResult(
        intents=rule.intents, confidence=INTENT_LLM_CONFIDENCE,
        primary_intent=rule.primary_intent, matched_keywords=rule.matched_keywords,
        scores=rule.scores, prompt_modules=rule.prompt_modules,
        llm_recommended_tools=[], llm_forbidden_tools=[],
    )
    ir.user_input = q
    tier = engine._determine_tier(ir)
    names = engine._resolve_dependencies(engine._get_tool_names_for_tier(tier, ir))
    return names


def _extract_content(resp) -> str:
    """取回模型文本：优先 message.content，空则回退 reasoning_content（思考模型）。"""
    raw = ""
    if hasattr(resp, "choices") and resp.choices:
        msg = resp.choices[0].message
        raw = (getattr(msg, "content", None) or "").strip()
        if not raw:
            raw = (getattr(msg, "reasoning_content", None) or "").strip()
    return raw


async def _gen_for_tool(registry: ModelRegistry, tool: str, desc: str) -> list[str]:
    prompt = GEN_PROMPT.format(desc=desc, tool=tool, n=N_PER_TOOL)
    last = ""
    for attempt in range(6):  # 空响应/解析失败均重试（deepseek 偶发限流返回空）
        try:
            resp = await asyncio.wait_for(
                registry.chat(
                    model_key=GEN_MODEL,
                    messages=[
                        {"role": "system", "content": "你只输出合法 JSON 数组，无多余文本。"},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=1600, max_retries=1, session_id="n1_expand_queries",
                ),
                timeout=GEN_TIMEOUT,
            )
            raw = _extract_content(resp)
            arr = _parse_json_array(raw) if raw else []
            if arr:
                return arr
            last = f"empty(raw_len={len(raw)})"
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {str(e)[:50]}"
        await asyncio.sleep(1.5 * (attempt + 1))
    print(f"  [gen-fail] {tool}: {last}")
    return []


async def main_async() -> None:
    reg = ToolRegistry()
    reg.load_config()
    reg.auto_discover(lazy=True)
    registered = set(reg.list_all_tool_names())

    engine = ToolExposureEngine(reg, enabled=True, enable_annotation=False)
    engine._rag_enabled = True
    engine._rag_top_k = 15
    engine._rag_min_score = 0.0
    engine._rag_include_extended_floor = True
    engine._llm_merge_recommended = False  # before 臂
    engine._llm_max_merge = 8
    engine._rag_index = ToolRetrievalIndex(reg)

    registry = ModelRegistry(config_path=ROOT / "config" / "models.toml")

    # 断点续跑：加载已有结果，跳过已覆盖工具，仅补缺口（幂等）
    records: list[dict] = []
    seen: set[str] = set()
    covered: set[str] = set()
    if OUT_JSON.exists():
        try:
            prev = json.loads(OUT_JSON.read_text(encoding="utf-8"))["queries"]
            for r in prev:
                records.append(r)
                seen.add(r["q"])
                covered.add(r["ref"])
            print(f"[resume] 已有 {len(records)} 条，覆盖 {len(covered)} 工具，仅补缺口")
        except Exception:  # noqa: BLE001
            records = []
    gid = max((r["id"] for r in records), default=0)

    print("=" * 90)
    print(f"N1 500-query 扩充集生成  目标工具={len(TARGET_TOOLS)}  每工具={N_PER_TOOL}  "
          f"目标总数={len(TARGET_TOOLS) * N_PER_TOOL}")
    print("=" * 90)

    for tool, desc in TARGET_TOOLS.items():
        if tool not in registered:
            print(f"  [skip] {tool} 未注册")
            continue
        if tool in covered:
            continue
        qs = await _gen_for_tool(registry, tool, desc)
        await asyncio.sleep(0.8)  # 工具间节流，降低连续限流概率
        kept = 0
        for q in qs:
            q = re.sub(r"\s+", "", q)  # 去空白，便于去重与规则匹配稳定
            if not q or q in seen:
                continue
            seen.add(q)
            expo = _recommended_exposure(engine, q)
            rule_hit = tool in expo
            gid += 1
            kept += 1
            records.append({
                "id": gid, "q": q, "ref": tool, "desc": desc,
                "rule_hit_before": rule_hit, "rec_expo_size_before": len(expo),
            })
        print(f"  {tool:20} gen={len(qs):2d} kept={kept:2d} "
              f"rule_hit={sum(1 for r in records if r['ref']==tool and r['rule_hit_before'])}")

    n = len(records)
    rule_hits = sum(1 for r in records if r["rule_hit_before"])
    gap_candidates = n - rule_hits
    print("\n" + "-" * 90)
    print(f"总 query={n}  规则已暴露(rule_hit)={rule_hits} ({rule_hits/n*100:.1f}%)  "
          f"gap 候选(规则漏)={gap_candidates} ({gap_candidates/n*100:.1f}%)")
    print(f"目标工具数={len({r['ref'] for r in records})}")

    OUT_JSON.write_text(
        json.dumps({
            "generated_at": "2026-09-27", "gen_model": GEN_MODEL,
            "n": n, "n_per_tool": N_PER_TOOL, "registered_tools": len(registered),
            "rule_hit_before": rule_hits, "gap_candidates": gap_candidates,
            "queries": records,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[written] {OUT_JSON}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
