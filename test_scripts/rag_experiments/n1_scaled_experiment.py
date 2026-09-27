# -*- coding: utf-8 -*-
r"""N1 · P0-1 —— 500-query 扩样本重跑（round2 离线反事实 + round3 确定性重放 + Wilson CI）。

评审 M1 关切：n=17 单快照太小、无置信区间。本脚本在 **n=500**（25 工具×20，ref by
construction）上重算 round2/round3 全部指标，并为每个比例给出 **Wilson 95% CI**。

复现快照：src/core 代码状态固定于 git tag **n1-eval-v1**（见输出 meta.base_tag）。
本脚本在 HEAD 全量重算，产出**内部一致的新数字**（旧 17-query 数字因意图关键词表
自原快照后扩充而漂移，本重算不修补旧值、直接以 500 集为准）。

两段实验：
  · round2（真实 LLM）：deepseek-v4-flash 跑 LLM_TOOL_CLASSIFY_PROMPT，实测
      rec_hit=ref∈推荐集；cov_expo_before=ref∈暴露集(merge off)；
      cov_expo_after=ref∈暴露集(merge on, 生产 _augment_with_llm_recommended, max_merge=8)；
      gap=rec_hit∧¬cov_expo_before。记录真实单次延迟。
  · round3（确定性重放，不调 LLM）：注入 llm_recommended=[ref]，走真实生产链路
      (_determine_tier/_get_tool_names_for_tier/report_failure/_upgrade_tier)，
      before(merge off)/after(merge on) 双臂量化首轮可达率、绕轮代价、额外延迟下界。

产物：test_scripts/rag_experiments/n1_scaled_results.json + .txt 报告
运行：.\.venv\Scripts\python.exe test_scripts\rag_experiments\n1_scaled_experiment.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]
for _p in (str(ROOT), str(EXP)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import n1_expand_queries as gen  # noqa: E402  复用 _extract_content / env / logging

from src.core.intent_engine import IntentResult  # noqa: E402
from src.core.prompts import (  # noqa: E402
    LLM_TOOL_CLASSIFY_PROMPT,
    build_tool_summary,
    detect_intent_with_confidence,
)
from src.core.tool_exposure import ToolExposureEngine  # noqa: E402
from src.core.tool_retrieval import ToolRetrievalIndex  # noqa: E402
from src.models.registry import ModelRegistry  # noqa: E402
from src.tools.registry import ToolRegistry  # noqa: E402

IN_JSON = EXP / "n1_queries_500.json"
OUT_JSON = EXP / "n1_scaled_results.json"
OUT_TXT = EXP / "n1_scaled_results.txt"

# ---- 生产常量（与 round2/round3 探针一致）----
INTENT_LLM_MODEL = "deepseek-v4-flash"
INTENT_LLM_CONFIDENCE = 0.9
FAILURES_TO_UPGRADE = 2
MAX_ROUNDS = 6
LLM_ROUNDTRIP_MS_LB = 1950   # 冷启动保守下界；真实均值由 round2 实测覆盖
CONCURRENCY = 6
GEN_TIMEOUT = 30.0
BASE_TAG = "n1-eval-v1"


def _git_short_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT), text=True,
        ).strip()
    except Exception:  # noqa: BLE001
        return "?"


def _parse_llm_classify_json(raw_text: str) -> dict:
    """复刻 src/core/agent.py::_parse_llm_classify_json。"""
    json_text = raw_text.strip()
    if "```" in json_text:
        m = re.search(r"```(?:json)?\s*(.*?)```", json_text, re.DOTALL)
        if m:
            json_text = m.group(1).strip()
    s, e = json_text.find("{"), json_text.rfind("}")
    if s >= 0 and e > s:
        json_text = json_text[s:e + 1]
    obj = json.loads(json_text)
    return {
        "recommended_tools": obj.get("recommended_tools", []) or [],
        "forbidden_tools": obj.get("forbidden_tools", []) or [],
        "task_type": obj.get("task_type", ""),
    }


def wilson_ci(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    """Wilson score 95% interval for a binomial proportion. Returns percent bounds."""
    if n == 0:
        return (float("nan"), float("nan"))
    ph = k / n
    denom = 1 + z * z / n
    center = (ph + z * z / (2 * n)) / denom
    half = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half) * 100, min(1.0, center + half) * 100)


def _rate_ci(k: int, n: int) -> dict:
    lo, hi = wilson_ci(k, n)
    return {
        "k": k, "n": n,
        "pct": round(k / n * 100, 1) if n else None,
        "ci95_low": round(lo, 1), "ci95_high": round(hi, 1),
    }


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else float("nan")


# ====================================================================
# round2 · 真实 LLM 分类
# ====================================================================

def _expose(engine: ToolExposureEngine, q: str, rec_tools: list[str],
            merge_on: bool) -> tuple[str, set[str]]:
    """走真实生产链路算暴露集（merge_on 控制 _augment_with_llm_recommended 生效）。"""
    rule = detect_intent_with_confidence(q)
    ir = IntentResult(
        intents=rule.intents, confidence=INTENT_LLM_CONFIDENCE,
        primary_intent=rule.primary_intent, matched_keywords=rule.matched_keywords,
        scores=rule.scores, prompt_modules=rule.prompt_modules,
        llm_recommended_tools=rec_tools, llm_forbidden_tools=[],
    )
    ir.user_input = q
    engine._llm_merge_recommended = merge_on
    tier = engine._determine_tier(ir)
    names = engine._resolve_dependencies(engine._get_tool_names_for_tier(tier, ir))
    return tier, names


def _make_engine(reg: ToolRegistry) -> ToolExposureEngine:
    engine = ToolExposureEngine(
        reg, enabled=True, enable_annotation=False,
        failures_to_upgrade=FAILURES_TO_UPGRADE,
    )
    engine._rag_enabled = True
    engine._rag_top_k = 15
    engine._rag_min_score = 0.0
    engine._rag_include_extended_floor = True
    engine._rag_escalate_full = False
    engine._rag_escalate_on_failure = True
    engine._rag_index = ToolRetrievalIndex(reg)
    engine._llm_max_merge = 8
    return engine


async def _classify_one(registry: ModelRegistry, summary: str, q: str) -> tuple[dict, float]:
    """单条 LLM 分类 + 实测延迟；空响应/异常重试（deepseek 偶发限流返回空）。"""
    prompt = LLM_TOOL_CLASSIFY_PROMPT.format(user_input=q, tool_summary=summary)
    last = ""
    for attempt in range(5):
        t0 = time.perf_counter()
        try:
            resp = await asyncio.wait_for(
                registry.chat(
                    model_key=INTENT_LLM_MODEL,
                    messages=[
                        {"role": "system", "content": "你是工具选择专家，只输出JSON。"},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=500, max_retries=1, session_id="n1_scaled_r2",
                ),
                timeout=GEN_TIMEOUT,
            )
            dt = (time.perf_counter() - t0) * 1000
            raw = gen._extract_content(resp)
            if raw:
                return _parse_llm_classify_json(raw), dt
            last = f"empty(raw_len={len(raw)})"
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {str(e)[:50]}"
        await asyncio.sleep(1.2 * (attempt + 1))
    raise RuntimeError(f"classify-failed: {last}")


async def run_round2(queries: list[dict], engine: ToolExposureEngine,
                     registry: ModelRegistry, registered: set) -> list[dict]:
    summary = build_tool_summary(engine._registry)
    sem = asyncio.Semaphore(CONCURRENCY)
    rows: list[dict] = []
    latencies: list[float] = []

    async def _work(rec: dict) -> dict:
        q, ref = rec["q"], rec["ref"]
        async with sem:
            err = ""
            parsed: dict = {"recommended_tools": [], "forbidden_tools": [], "task_type": ""}
            dt = float("nan")
            llm_ok = False
            try:
                parsed, dt = await _classify_one(registry, summary, q)
                llm_ok = True
                latencies.append(dt)
            except Exception as e:  # noqa: BLE001
                err = str(e)[:80]
        rec_tools = [t for t in parsed["recommended_tools"] if t in registered]
        rec_set = set(rec_tools)
        tier_b, names_b = _expose(engine, q, rec_tools, merge_on=False)
        tier_a, names_a = _expose(engine, q, rec_tools, merge_on=True)
        rec_hit = ref in rec_set
        cov_before = ref in names_b
        cov_after = ref in names_a
        gap = bool(rec_hit and not cov_before)
        return {
            "id": rec["id"], "q": q, "ref": ref, "llm_ok": llm_ok, "err": err,
            "lat_ms": round(dt, 1) if dt == dt else None,
            "tier_before": tier_b, "expo_before": len(names_b), "expo_after": len(names_a),
            "rec_tools": rec_tools, "rec_hit": rec_hit,
            "cov_expo_before": cov_before, "cov_expo_after": cov_after, "gap": gap,
            "rule_hit_before": rec.get("rule_hit_before"),
        }

    tasks = [_work(r) for r in queries]
    print(f"[round2] 真实 LLM 分类 n={len(queries)} 并发={CONCURRENCY} ...")
    done = 0
    for fut in asyncio.as_completed(tasks):
        rows.append(await fut)
        done += 1
        if done % 50 == 0:
            print(f"  ...{done}/{len(queries)}")
    rows.sort(key=lambda r: r["id"])
    return rows, latencies


# ====================================================================
# round3 · 确定性重放（before/after 双臂，注入 [ref]）
# ====================================================================

def _build_llm_intent(q: str, ref: str) -> IntentResult:
    rule = detect_intent_with_confidence(q)
    ir = IntentResult(
        intents=rule.intents, confidence=INTENT_LLM_CONFIDENCE,
        primary_intent=rule.primary_intent, matched_keywords=rule.matched_keywords,
        scores=rule.scores, prompt_modules=rule.prompt_modules,
        llm_recommended_tools=[ref] if ref else [], llm_forbidden_tools=[],
    )
    ir.user_input = q
    return ir


def _run_chain(engine: ToolExposureEngine, ir: IntentResult, ref: str) -> dict:
    engine.reset()
    engine._rag_escalate_full = False
    trajectory: list[tuple[int, str, int, bool]] = []
    failed_rt = 0
    escalations = 0
    recovered = False
    first_ok = False
    for rt in range(1, MAX_ROUNDS + 1):
        tier = engine._determine_tier(ir)
        names = engine._resolve_dependencies(engine._get_tool_names_for_tier(tier, ir))
        ref_exposed = ref in names
        trajectory.append((rt, tier, len(names), ref_exposed))
        if rt == 1:
            first_ok = ref_exposed
        if ref_exposed:
            recovered = True
            engine.report_success()
            break
        failed_rt += 1
        if engine.report_failure():
            escalations += 1
    return {
        "first_ok": first_ok, "recovered": recovered, "failed_rt": failed_rt,
        "escalations": escalations, "total_rt": len(trajectory),
        "first_expo": trajectory[0][2] if trajectory else 0,
        "final_expo": trajectory[-1][2] if trajectory else 0,
        "final_tier": trajectory[-1][1] if trajectory else None,
        "trajectory": trajectory,
    }


def _run_arm_r3(reg: ToolRegistry, queries: list[dict], merge_on: bool) -> list[dict]:
    engine = _make_engine(reg)
    engine._llm_merge_recommended = merge_on
    out: list[dict] = []
    for rec in queries:
        ir = _build_llm_intent(rec["q"], rec["ref"])
        r = _run_chain(engine, ir, rec["ref"])
        r["id"] = rec["id"]
        out.append(r)
    return out


def _agg_r3(rows: list[dict]) -> dict:
    n = len(rows)
    first_ok = sum(1 for r in rows if r["first_ok"])
    detour = [r for r in rows if not r["first_ok"]]
    recovered = sum(1 for r in detour if r["recovered"])
    lb = _mean([r["failed_rt"] for r in rows if r["failed_rt"] > 0] or [0])
    return {
        "n": n,
        "first_ok": _rate_ci(first_ok, n),
        "need_detour": _rate_ci(len(detour), n),
        "recovery_of_detour": _rate_ci(recovered, len(detour)) if detour else _rate_ci(0, 0),
        "failed_rt_total": sum(r["failed_rt"] for r in rows),
        "failed_rt_mean": round(_mean([r["failed_rt"] for r in rows]), 3),
        "escalations_total": sum(r["escalations"] for r in rows),
        "first_expo_mean": round(_mean([r["first_expo"] for r in rows]), 1),
        "final_expo_mean": round(_mean([r["final_expo"] for r in rows]), 1),
        "extra_latency_ms_lb_total": sum(r["failed_rt"] for r in rows) * LLM_ROUNDTRIP_MS_LB,
    }


# ====================================================================
# 聚合 + 报告
# ====================================================================

def _agg_r2(rows: list[dict], latencies: list[float]) -> dict:
    refd = [r for r in rows if r["llm_ok"]]
    n = len(refd)
    rec_hit = sum(1 for r in refd if r["rec_hit"])
    cov_b = sum(1 for r in refd if r["cov_expo_before"])
    cov_a = sum(1 for r in refd if r["cov_expo_after"])
    gap = sum(1 for r in refd if r["gap"])
    return {
        "n_llm_ok": n,
        "n_total": len(rows),
        "rec_hit_rate": _rate_ci(rec_hit, n),
        "cov_expo_before_rate": _rate_ci(cov_b, n),
        "cov_expo_after_rate": _rate_ci(cov_a, n),
        "gap_rate": _rate_ci(gap, n),
        "expo_before_mean": round(_mean([r["expo_before"] for r in refd]), 1),
        "expo_after_mean": round(_mean([r["expo_after"] for r in refd]), 1),
        "llm_latency_ms_mean": round(_mean(latencies), 0),
        "llm_latency_ms_min": round(min(latencies), 0) if latencies else None,
        "llm_latency_ms_max": round(max(latencies), 0) if latencies else None,
    }


def main() -> None:
    data = json.loads(IN_JSON.read_text(encoding="utf-8"))
    queries = data["queries"]

    reg = ToolRegistry()
    reg.load_config()
    reg.auto_discover(lazy=True)
    registered = set(reg.list_all_tool_names())
    registry = ModelRegistry(config_path=ROOT / "config" / "models.toml")

    print("=" * 96)
    print(f"N1 扩样本重跑  n={len(queries)}  base_tag={BASE_TAG}  HEAD={_git_short_head()}")
    print("=" * 96)

    engine = _make_engine(reg)
    rows_r2, latencies = asyncio.run(run_round2(queries, engine, registry, registered))
    agg_r2 = _agg_r2(rows_r2, latencies)

    r3_before = _run_arm_r3(reg, queries, merge_on=False)
    r3_after = _run_arm_r3(reg, queries, merge_on=True)
    agg_before = _agg_r3(r3_before)
    agg_after = _agg_r3(r3_after)

    result = {
        "meta": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "base_tag": BASE_TAG, "head": _git_short_head(),
            "gen_model": gen.GEN_MODEL, "round2_model": INTENT_LLM_MODEL,
            "n_queries": len(queries), "n_tools": len({r["ref"] for r in queries}),
            "registered_tools": len(registered),
            "intent_confidence": INTENT_LLM_CONFIDENCE,
            "failures_to_upgrade": FAILURES_TO_UPGRADE, "max_merge": 8,
        },
        "round2": {"aggregate": agg_r2, "rows": rows_r2},
        "round3": {"before": agg_before, "after": agg_after},
    }
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 文本报告 ----
    lines: list[str] = []
    A = lines.append
    A("=" * 96)
    A("N1 · P0-1 扩样本重跑（500 query）· round2 真实LLM + round3 确定性重放 + Wilson 95% CI")
    A(f"base_tag={BASE_TAG}  HEAD={_git_short_head()}  生成时间={result['meta']['generated_at']}")
    A(f"n={len(queries)}  工具数={result['meta']['n_tools']}  注册工具={len(registered)}")
    A("=" * 96)
    A("\n## round2 · 真实 LLM 意图分类（merge off 为 before 臂）")
    A(f"  LLM 有效样本={agg_r2['n_llm_ok']}/{agg_r2['n_total']}")
    for key, lab in [
        ("rec_hit_rate", "LLM 推荐命中率 ref∈推荐集"),
        ("cov_expo_before_rate", "真实暴露覆盖率 ref∈暴露集(before)"),
        ("cov_expo_after_rate", "修复后覆盖率 ref∈暴露集(after merge)"),
        ("gap_rate", "GAP 率 rec_hit∧¬cov_expo(before)"),
    ]:
        c = agg_r2[key]
        A(f"  [{lab}] {c['k']}/{c['n']} = {c['pct']}%  "
          f"Wilson95%CI[{c['ci95_low']}, {c['ci95_high']}]")
    A(f"  暴露量均值 before={agg_r2['expo_before_mean']}  after={agg_r2['expo_after_mean']}")
    A(f"  LLM 单次延迟 均值={agg_r2['llm_latency_ms_mean']}ms "
      f"(min={agg_r2['llm_latency_ms_min']} max={agg_r2['llm_latency_ms_max']})")

    A("\n## round3 · 确定性重放（before/after 双臂, 注入[ref]）")
    A(f"  {'指标':<32}{'before':>26}{'after':>26}")
    A("  " + "-" * 84)

    def _fmt(ci: dict) -> str:
        return f"{ci['k']}/{ci['n']}={ci['pct']}% [{ci['ci95_low']},{ci['ci95_high']}]"
    A(f"  {'首轮可达率':<30}{_fmt(agg_before['first_ok']):>26}{_fmt(agg_after['first_ok']):>26}")
    A(f"  {'需绕轮率':<30}{_fmt(agg_before['need_detour']):>26}{_fmt(agg_after['need_detour']):>26}")
    A(f"  {'绕轮后恢复率':<29}{_fmt(agg_before['recovery_of_detour']):>26}{_fmt(agg_after['recovery_of_detour']):>26}")
    A(f"  {'失败RT总数':<30}{agg_before['failed_rt_total']:>26}{agg_after['failed_rt_total']:>26}")
    A(f"  {'失败RT均值':<30}{agg_before['failed_rt_mean']:>26}{agg_after['failed_rt_mean']:>26}")
    A(f"  {'升级触发总数':<29}{agg_before['escalations_total']:>26}{agg_after['escalations_total']:>26}")
    A(f"  {'首轮暴露均值':<29}{agg_before['first_expo_mean']:>26}{agg_after['first_expo_mean']:>26}")
    A(f"  {'末档暴露均值':<29}{agg_before['final_expo_mean']:>26}{agg_after['final_expo_mean']:>26}")
    A(f"  {'额外延迟下界(ms)':<28}{agg_before['extra_latency_ms_lb_total']:>26}"
      f"{agg_after['extra_latency_ms_lb_total']:>26}")

    report = "\n".join(lines)
    OUT_TXT.write_text(report, encoding="utf-8")
    try:
        sys.__stdout__.write(report + "\n")
    except Exception:  # noqa: BLE001
        pass
    print(f"\n[written] {OUT_JSON}")
    print(f"[written] {OUT_TXT}")


if __name__ == "__main__":
    main()
