# -*- coding: utf-8 -*-
r"""N1 · P0-2 —— 真实端到端 LIVE 跑（每轮真调 LLM function-calling + 实测 wall-clock）。

评审 M2/m 关切：round3 是"确定性重放 + 仿真延迟下界(1950ms)"，非实测、未真调模型。
本脚本把触发链升级为 **LIVE**：每一 round-trip **真实调用** deepseek function-calling
（把当前暴露集 schema 作为 tools 传入，取回模型真实 tool_calls），并**实测**每条样本
在 before/after 两臂下"直到推荐工具可达"所消耗的真实墙钟时间。

机制保真（不夸大）：
  · 识别（recognition）：沿用 round2 **真实** LLM 分类得到的 recommended_tools（本脚本按
    id 从 n1_scaled_results.json 载入，不重复分类，保证与 P0-1 同源一致）。
  · 暴露（exposure）+ 升级（escalation）：走**真实生产** ToolExposureEngine
    (_determine_tier / _get_tool_names_for_tier / _resolve_dependencies / report_failure /
    _upgrade_tier)，merge off=before、merge on=after。
  · 每轮真实模型决策：registry.chat(tools=exposed_schemas) → 解析 message.tool_calls[0].name，
    既产出**真实单轮延迟**，又验证"工具一旦进入暴露集，模型确能 function-call 到它"。
  · 可达判据（硬边界）：ref ∈ 本轮暴露集 ⇒ 该轮 function-calling 可调 ref（成功）；
    ref ∉ 暴露集 ⇒ 模型只能从暴露集里选别的（无法调 ref）→ 记一次执行失败 → report_failure
    → 升级 → 下一轮。**不真实执行被选工具**（避开 25 类工具副作用），聚焦识别-暴露机制本身。

双臂对照（同一批 500 query）：
  · before：merge_recommended=False → gap 样本须绕多轮真实 LLM 才兜回 ref（真实累计延迟）。
  · after ：merge_recommended=True  → ref 首轮即并入暴露集 → 多数一轮即达（零绕轮）。

产物：test_scripts/rag_experiments/n1_live_results.json + .txt
运行：.\.venv\Scripts\python.exe test_scripts\rag_experiments\n1_live_experiment.py
可选：$env:N_LIVE="120"  仅跑分层抽样子集（默认全部 500）
"""
from __future__ import annotations

import asyncio
import json
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]
for _p in (str(ROOT), str(EXP)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import n1_expand_queries as gen  # noqa: E402
from n1_scaled_experiment import (  # noqa: E402  复用生产链路助手
    INTENT_LLM_MODEL, INTENT_LLM_CONFIDENCE, FAILURES_TO_UPGRADE, MAX_ROUNDS,
    _make_engine, _expose, _build_llm_intent, wilson_ci, _mean,
)

from src.core.intent_engine import IntentResult  # noqa: E402
from src.models.registry import ModelRegistry  # noqa: E402
from src.tools.registry import ToolRegistry  # noqa: E402

QUERIES_JSON = EXP / "n1_queries_500.json"
SCALED_JSON = EXP / "n1_scaled_results.json"
OUT_JSON = EXP / "n1_live_results.json"
OUT_TXT = EXP / "n1_live_results.txt"

CONCURRENCY = 8
CHAT_TIMEOUT = 45.0
BASE_TAG = "n1-eval-v1"


def _extract_tool_calls(resp) -> tuple[str, float]:
    """从真实 function-calling 响应取首选工具名（无则空串）。"""
    if hasattr(resp, "choices") and resp.choices:
        msg = resp.choices[0].message
        tcs = getattr(msg, "tool_calls", None) or []
        for tc in tcs:
            fn = getattr(tc, "function", None)
            if fn and getattr(fn, "name", ""):
                return fn.name, float(getattr(tc, "index", 0) or 0)
    return "", 0.0


async def _live_roundtrip(registry: ModelRegistry, sem, q: str,
                          schemas: list[dict]) -> tuple[str, float]:
    """真实调用模型一次（带 tools），返回(选中工具名, 单轮实测延迟ms)；空响应重试。"""
    messages = [
        {"role": "system", "content": "你是桌面助手，依据用户请求从可用工具中选择最合适的一个调用。"},
        {"role": "user", "content": q},
    ]
    last_err = ""
    for attempt in range(3):
        async with sem:
            t0 = time.perf_counter()
            try:
                resp = await asyncio.wait_for(
                    registry.chat(
                        model_key=INTENT_LLM_MODEL, messages=messages,
                        tools=schemas or None, max_retries=1, session_id="n1_live",
                    ),
                    timeout=CHAT_TIMEOUT,
                )
                dt = (time.perf_counter() - t0) * 1000
                chosen, _ = _extract_tool_calls(resp)
                if chosen or schemas:
                    return chosen, dt
                last_err = "empty"
            except Exception as e:  # noqa: BLE001
                last_err = f"{type(e).__name__}:{str(e)[:40]}"
            await asyncio.sleep(1.0 * (attempt + 1))
    return "", float("nan")


async def _live_arm(registry: ModelRegistry, reg: ToolRegistry, sem,
                    q: str, ref: str, rec_tools: list[str], merge_on: bool) -> dict:
    """真实端到端单臂：逐轮真调 LLM 直到 ref 可达或到顶，累计实测延迟。"""
    engine = _make_engine(reg)
    engine._llm_merge_recommended = merge_on
    ir = _build_llm_intent(q, ref)
    # 用 round2 真实推荐（非 [ref] 注入）驱动 after 臂 merge，语义与 P0-1 一致
    ir = IntentResult(
        intents=ir.intents, confidence=INTENT_LLM_CONFIDENCE,
        primary_intent=ir.primary_intent, matched_keywords=ir.matched_keywords,
        scores=ir.scores, prompt_modules=ir.prompt_modules,
        llm_recommended_tools=rec_tools, llm_forbidden_tools=[],
    )
    ir.user_input = q

    engine.reset()
    engine._rag_escalate_full = False
    wall0 = time.perf_counter()
    roundtrips = 0
    reached = False
    model_chose_ref = False
    per_rt_ms: list[float] = []
    for _rt_i in range(MAX_ROUNDS):
        tier = engine._determine_tier(ir)
        names = engine._resolve_dependencies(engine._get_tool_names_for_tier(tier, ir))
        schemas = engine._registry.get_schemas_by_names(names) or []
        roundtrips += 1
        chosen, dt = await _live_roundtrip(registry, sem, q, schemas)
        if dt == dt:
            per_rt_ms.append(dt)
        if ref in names:                      # 硬边界：ref 已进暴露集 → 本轮可达
            reached = True
            resolved = engine._registry.resolve_function_name(chosen) if chosen else None
            model_chose_ref = bool(resolved and resolved[0] == ref)
            engine.report_success()
            break
        if engine.report_failure():           # ref 不可达 → 生产升级阶梯
            pass
    total_ms = (time.perf_counter() - wall0) * 1000
    return {
        "arm": "after" if merge_on else "before", "roundtrips": roundtrips,
        "reached": reached, "model_chose_ref": model_chose_ref,
        "rt_ms_mean": round(_mean(per_rt_ms), 0) if per_rt_ms else None,
        # 并发无关的干净度量：该 query 各轮**实测模型延迟**之和（不含排队等待）
        "sum_rt_ms": round(sum(per_rt_ms)), "n_timed_rt": len(per_rt_ms),
        "wall_ms": round(total_ms, 0), "llm_calls": roundtrips,
    }


async def _live_one(registry, reg, sem, rec, ref: str, rec_tools: list[str]) -> dict:
    before = await _live_arm(registry, reg, sem, rec["q"], ref, rec_tools, merge_on=False)
    after = await _live_arm(registry, reg, sem, rec["q"], ref, rec_tools, merge_on=True)
    return {"id": rec["id"], "q": rec["q"], "ref": ref,
            "rec_hit": ref in set(rec_tools), "before": before, "after": after}


def _wilson(k: int, n: int) -> dict:
    lo, hi = wilson_ci(k, n)
    return {"k": k, "n": n, "pct": round(k / n * 100, 1) if n else None,
            "ci95_low": round(lo, 1), "ci95_high": round(hi, 1)}


def _agg(rows: list[dict], arm: str) -> dict:
    n = len(rows)
    a = [r[arm] for r in rows]
    reached = sum(1 for x in a if x["reached"])
    chose = sum(1 for x in a if x["model_chose_ref"])
    rt1 = sum(1 for x in a if x["roundtrips"] == 1)
    rts = [x["roundtrips"] for x in a]
    sums = [x["sum_rt_ms"] for x in a if x.get("sum_rt_ms") is not None]
    llm_calls = sum(x["llm_calls"] for x in a)
    return {
        "n": n,
        "reached_rate": _wilson(reached, n),
        "first_rt_rate": _wilson(rt1, n),
        "model_chose_ref_rate": _wilson(chose, n),
        "roundtrips_mean": round(_mean(rts), 2),
        "accum_ms_mean": round(_mean(sums), 0),
        "accum_ms_median": round(statistics.median(sums), 0) if sums else None,
        "accum_ms_p90": round(sorted(sums)[int(len(sums) * 0.9) - 1], 0) if sums else None,
        "per_rt_ms_mean": round(_mean([x["rt_ms_mean"] for x in a if x.get("rt_ms_mean")]), 0),
        "llm_calls_total": llm_calls,
    }


def _stratified_sample(queries: list[dict], k: int) -> list[dict]:
    if k >= len(queries):
        return queries
    by_tool: dict[str, list[dict]] = {}
    for r in queries:
        by_tool.setdefault(r["ref"], []).append(r)
    out: list[dict] = []
    per = max(1, k // len(by_tool))
    for tname, rs in by_tool.items():
        out.extend(rs[:per])
    return out[:k]


async def main_async() -> None:
    data = json.loads(QUERIES_JSON.read_text(encoding="utf-8"))
    queries = data["queries"]
    scaled = json.loads(SCALED_JSON.read_text(encoding="utf-8"))
    rec_by_id = {r["id"]: r["rec_tools"] for r in scaled["round2"]["rows"]}

    n_live = int(os.environ.get("N_LIVE", "0") or 0)
    if n_live > 0:
        queries = _stratified_sample(queries, n_live)
    else:
        n_live = len(queries)

    reg = ToolRegistry()
    reg.load_config()
    reg.auto_discover(lazy=True)
    registry = ModelRegistry(config_path=ROOT / "config" / "models.toml")
    sem = asyncio.Semaphore(CONCURRENCY)

    print("=" * 96)
    print(f"N1 P0-2 LIVE 端到端  n={len(queries)}  model={INTENT_LLM_MODEL}  "
          f"并发={CONCURRENCY}  base_tag={BASE_TAG}")
    print("每轮真调 function-calling + 实测 wall-clock；before/after 双臂")
    print("=" * 96)

    tasks = [_live_one(registry, reg, sem, r, r["ref"], rec_by_id.get(r["id"], []))
             for r in queries]
    rows: list[dict] = []
    done = 0
    for fut in asyncio.as_completed(tasks):
        rows.append(await fut)
        done += 1
        if done % 25 == 0:
            print(f"  ...{done}/{len(queries)}")
    rows.sort(key=lambda r: r["id"])

    agg_before = _agg(rows, "before")
    agg_after = _agg(rows, "after")
    result = {
        "meta": {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                 "base_tag": BASE_TAG, "model": INTENT_LLM_MODEL, "n": len(queries),
                 "concurrency": CONCURRENCY, "recognitions_from": "round2_real"},
        "before": agg_before, "after": agg_after, "rows": rows,
    }
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = []
    A = lines.append
    A("=" * 96)
    A(f"N1 · P0-2 真实端到端 LIVE（n={len(queries)}, model={INTENT_LLM_MODEL}）· 实测 wall-clock")
    A(f"生成 {result['meta']['generated_at']}  base_tag={BASE_TAG}")
    A("=" * 96)
    A(f"  {'指标':<30}{'before(merge off)':>32}{'after(merge on)':>32}")
    A("  " + "-" * 92)

    def _fmt(ci: dict) -> str:
        return f"{ci['k']}/{ci['n']}={ci['pct']}% [{ci['ci95_low']},{ci['ci95_high']}]"
    A(f"  {'首轮即达率':<28}{_fmt(agg_before['first_rt_rate']):>32}{_fmt(agg_after['first_rt_rate']):>32}")
    A(f"  {'最终可达率':<28}{_fmt(agg_before['reached_rate']):>32}{_fmt(agg_after['reached_rate']):>32}")
    A(f"  {'模型选中ref率':<27}{_fmt(agg_before['model_chose_ref_rate']):>32}{_fmt(agg_after['model_chose_ref_rate']):>32}")
    A(f"  {'平均round-trip数':<27}{agg_before['roundtrips_mean']:>32}{agg_after['roundtrips_mean']:>32}")
    A(f"  {'平均单轮实测延迟(ms)':<25}{agg_before['per_rt_ms_mean']:>32}{agg_after['per_rt_ms_mean']:>32}")
    A(f"  {'累计模型延迟均值(ms)':<25}{agg_before['accum_ms_mean']:>32}{agg_after['accum_ms_mean']:>32}")
    A(f"  {'累计模型延迟中位(ms)':<25}{agg_before['accum_ms_median']:>32}{agg_after['accum_ms_median']:>32}")
    A(f"  {'累计模型延迟P90(ms)':<26}{agg_before['accum_ms_p90']:>32}{agg_after['accum_ms_p90']:>32}")
    A(f"  {'真实LLM调用总数':<27}{agg_before['llm_calls_total']:>32}{agg_after['llm_calls_total']:>32}")
    report = "\n".join(lines)
    OUT_TXT.write_text(report, encoding="utf-8")
    try:
        sys.__stdout__.write(report + "\n")
    except Exception:  # noqa: BLE001
        pass
    print(f"\n[written] {OUT_JSON}\n[written] {OUT_TXT}")


if __name__ == "__main__":
    asyncio.run(main_async())
