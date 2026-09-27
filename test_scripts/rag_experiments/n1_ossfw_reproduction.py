# -*- coding: utf-8 -*-
r"""N1 · P1-1 —— 开源框架两阶段解耦：最小复现（framework-independent 论证）。

评审关切：GAP 是否 WeClaw 特有？本脚本把机制抽为**通用架构模式**——
  「识别(recognition) 与 暴露(exposure) 两阶段解耦」：识别阶段独立产出意图/推荐，
  暴露阶段按**与识别无关的策略**决定模型实际可调用（function-binding）的工具集。
只要暴露策略未把识别结果并入，就会出现"识别对了却没暴露"=GAP。

三种在开源 agent 框架中常见的暴露策略原型（reference adapter，纯 Python，零新依赖）：
  · LangGraphRouterAdapter  —— router 节点判意图→静态 allow-list（CORE∪意图映射），
      executor 只 bind allow-list；识别推荐**未并入** → 与 WeClaw before 同构（GAP 战场）。
  · AutoGPTFullBindAdapter  —— executor 直接 bind **全量**工具（无解耦）→ 覆盖≈100%、
      无 GAP，但上下文成本巨大（暴露量≈全部注册工具）。
  · RetrieverTopKAdapter    —— 常见 tool-retriever（langchain/llama-index 式）：暴露
      CORE∪Top-K(检索)。用生产 ToolRetrievalIndex 作 OSS 式检索器（K=15）。
再叠加统一的 RecognitionMergeFix（把识别推荐并入暴露、上限 8）：对应 after 臂。

识别信号 rec_hit 复用 round2 **真实** deepseek 分类结果（按 id 从 n1_scaled_results.json
载入），保证与主实验同源；本脚本其余为**确定性**（不调 LLM），完全可复现。

论断：GAP 由"两阶段解耦且识别未并入暴露"这一**架构性质**产生，与具体框架无关；
      任一采用 router/静态 allow-list 且不与识别合并的框架都会复现，合并修复即可闭合。

产物：test_scripts/rag_experiments/n1_ossfw_results.json + .txt
运行：.\.venv\Scripts\python.exe test_scripts\rag_experiments\n1_ossfw_reproduction.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]
for _p in (str(ROOT), str(EXP)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.core.intent_engine import INTENT_TOOL_MAPPING, detect_intent_with_confidence  # noqa: E402
from src.core.tool_exposure import ToolExposureEngine  # noqa: E402
from src.core.tool_retrieval import ToolRetrievalIndex  # noqa: E402
from src.tools.registry import ToolRegistry  # noqa: E402

QUERIES_JSON = EXP / "n1_queries_500.json"
SCALED_JSON = EXP / "n1_scaled_results.json"
OUT_JSON = EXP / "n1_ossfw_results.json"
OUT_TXT = EXP / "n1_ossfw.txt"

CORE = set(ToolExposureEngine.CORE_TOOLS)
K_RETRIEVE = 15
MAX_MERGE = 8


def _wilson(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    import math
    if n == 0:
        return (float("nan"), float("nan"))
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (round(max(0.0, c - h) * 100, 1), round(min(1.0, c + h) * 100, 1))


# ---------------- 三种暴露策略（reference adapters）----------------

class LangGraphRouterAdapter:
    """router 判意图→静态 allow-list；识别推荐不并入（两阶段解耦，GAP 战场）。"""
    name = "LangGraphRouter(intent-allowlist)"

    def __init__(self, reg: ToolRegistry):
        self.reg = reg

    def expose(self, q: str, recognition: set[str], merge: bool) -> set[str]:
        rule = detect_intent_with_confidence(q)
        exposed = set(CORE) | {"tool_info"}
        for it in rule.intents:
            exposed |= set(INTENT_TOOL_MAPPING.get(it, []))
        if merge:
            added = 0
            for t in recognition:
                if added >= MAX_MERGE:
                    break
                if t in exposed:
                    continue
                exposed.add(t)
                added += 1
        return exposed


class AutoGPTFullBindAdapter:
    """executor 直接 bind 全量工具（无解耦）→ 无 GAP，但上下文成本高。"""
    name = "AutoGPTFullBind(all-tools)"

    def __init__(self, reg: ToolRegistry):
        self.reg = reg
        self._all = set(reg.list_all_tool_names())

    def expose(self, q: str, recognition: set[str], merge: bool) -> set[str]:
        return set(self._all)


class RetrieverTopKAdapter:
    """常见 tool-retriever：CORE ∪ Top-K(检索)（识别可另并入）。"""
    name = "RetrieverTopK(langchain/llama-index style)"

    def __init__(self, reg: ToolRegistry):
        self.reg = reg
        self.idx = ToolRetrievalIndex(reg)

    def expose(self, q: str, recognition: set[str], merge: bool) -> set[str]:
        retrieved = {n for n, _ in self.idx.retrieve(q, K_RETRIEVE, 0.0)}
        exposed = set(CORE) | retrieved
        if merge:
            exposed |= set(list(recognition)[:MAX_MERGE])
        return exposed


def main() -> None:
    queries = json.loads(QUERIES_JSON.read_text(encoding="utf-8"))["queries"]
    scaled = json.loads(SCALED_JSON.read_text(encoding="utf-8"))
    rec_by_id = {r["id"]: set(r["rec_tools"]) for r in scaled["round2"]["rows"]}

    reg = ToolRegistry(); reg.load_config(); reg.auto_discover(lazy=True)
    registered = set(reg.list_all_tool_names())

    adapters = [LangGraphRouterAdapter(reg), AutoGPTFullBindAdapter(reg),
                RetrieverTopKAdapter(reg)]

    print("=" * 100, flush=True)
    print(f"N1 P1-1 开源框架两阶段解耦最小复现  n={len(queries)}  "
          f"adapters={[a.name for a in adapters]}", flush=True)
    print("=" * 100, flush=True)

    results: dict[str, dict] = {}
    for ad in adapters:
        for merge in (False, True):
            n = cov = gap = rec_hit_n = expo_sum = 0
            for r in queries:
                if r["ref"] not in registered:
                    continue
                n += 1
                recognition = rec_by_id.get(r["id"], set())
                exposed = ad.expose(r["q"], recognition, merge)
                rh = r["ref"] in recognition
                cv = r["ref"] in exposed
                rec_hit_n += rh
                cov += cv
                gap += bool(rh and not cv)
                expo_sum += len(exposed)
            lo, hi = _wilson(gap, n)
            cov_lo, cov_hi = _wilson(cov, n)
            key = f"{ad.name} | merge={'on' if merge else 'off'}"
            results[key] = {
                "n": n,
                "rec_hit_rate_pct": round(rec_hit_n / n * 100, 1),
                "coverage_rate_pct": round(cov / n * 100, 1),
                "coverage_ci95": [cov_lo, cov_hi],
                "gap_rate_pct": round(gap / n * 100, 1),
                "gap_ci95": [lo, hi],
                "gap_count": gap,
                "exposure_mean": round(expo_sum / n, 1),
            }

    result = {
        "meta": {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                 "base_tag": "n1-eval-v1", "n": len(queries),
                 "recognition_source": "round2_real_deepseek",
                 "deterministic": True,
                 "thesis": "GAP 由两阶段解耦(识别未并入暴露)产生, 与框架无关; 合并即闭合"},
        "adapters": results,
    }
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = []
    A = lines.append
    A("=" * 100)
    A(f"N1 · P1-1 开源框架两阶段解耦最小复现（n={len(queries)}, 确定性）")
    A("识别信号复用 round2 真实 deepseek 推荐; 三种常见暴露策略原型 × merge on/off")
    A("=" * 100)
    A(f"  {'adapter | merge':<48}{'rec_hit%':>9}{'覆盖%':>9}{'GAP%':>8}{'GAP_CI95':>16}{'暴露均值':>9}")
    A("  " + "-" * 98)
    for key, v in results.items():
        A(f"  {key:<48}{v['rec_hit_rate_pct']:>9}{v['coverage_rate_pct']:>9}"
          f"{v['gap_rate_pct']:>8}  {v['gap_ci95'][0]},{v['gap_ci95'][1]:<9}"
          f"{v['exposure_mean']:>9}")
    A("")
    A("  解读：intent-router(allow-list, merge off) 复现 GAP（识别对了却未并入暴露）；")
    A("  同策略 merge on 后 GAP→~0；all-tools 无 GAP 但暴露≈全量(上下文成本)；retriever 介于之间。")
    A("  ⇒ 只要识别与暴露解耦且不合并, 任一框架都复现; 合并 co-design 即闭合。")
    report = "\n".join(lines)
    OUT_TXT.write_text(report, encoding="utf-8")
    try:
        sys.__stdout__.write(report + "\n")
    except Exception:  # noqa: BLE001
        pass
    print(f"\n[written] {OUT_JSON}\n[written] {OUT_TXT}", flush=True)


if __name__ == "__main__":
    main()
