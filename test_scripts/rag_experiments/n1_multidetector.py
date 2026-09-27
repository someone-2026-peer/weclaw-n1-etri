# -*- coding: utf-8 -*-
r"""N1 · E1 —— 多检测器复制（≥2 额外异提供商 LLM）验证 gap 跨模型稳健。

评审关切：round2 的"识别命中"只用了单一检测器（deepseek-v4-flash），GAP 可能是
该模型的伪影。本脚本用**异提供商**的额外检测器（默认 qwen-max[alibaba] +
glm-5[zhipu]）在同一批 query 上重做"识别"，度量每个检测器的：
  · rec_hit 率   = ref ∈ 该检测器推荐集（识别准确度，跨模型）
  · gap 率       = rec_hit ∧ ¬exposed（识别对了却没暴露）
关键：`exposed`(merge-off RECOMMENDED 暴露) 由**规则意图**决定、与检测器无关
（= 500 集里记录的 rule_hit_before），因此跨检测器的差异**只来自识别阶段**。
若多个异提供商模型都呈现"高 rec_hit + 同 ~42% 暴露 → gap 持续" ⇒ gap 是架构性质，
非某一检测器伪影。

为控制 API 负载，默认在**分层子样本**（env N_DET，默认 150）上跑额外检测器；
主检测器(deepseek)的 per-id 结果直接从 n1_scaled_results.json 复用，保证同集可比。

产物：test_scripts/rag_experiments/n1_multidetector_results.json + .txt
运行：.\.venv\Scripts\python.exe test_scripts\rag_experiments\n1_multidetector.py
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]
for _p in (str(ROOT), str(EXP)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import n1_expand_queries as gen  # noqa: E402
from n1_scaled_experiment import (  # noqa: E402
    _parse_llm_classify_json, wilson_ci, _mean, INTENT_LLM_MODEL,
)

from src.core.prompts import LLM_TOOL_CLASSIFY_PROMPT, build_tool_summary  # noqa: E402
from src.models.registry import ModelRegistry  # noqa: E402
from src.tools.registry import ToolRegistry  # noqa: E402

QUERIES_JSON = EXP / "n1_queries_500.json"
SCALED_JSON = EXP / "n1_scaled_results.json"
OUT_JSON = EXP / "n1_multidetector_results.json"
OUT_TXT = EXP / "n1_multidetector.txt"
CKPT_JSON = EXP / "n1_multidet_ckpt.json"  # 增量断点：{model: {id: [tools]}}

EXTRA_MODELS = [m.strip() for m in os.environ.get(
    "DET_MODELS", "qwen-max,glm-5").split(",") if m.strip()]
N_DET = int(os.environ.get("N_DET", "150") or 0)
CONCURRENCY = int(os.environ.get("DET_CONC", "6") or 6)
TIMEOUT = 40.0


def _wilson(k: int, n: int) -> dict:
    lo, hi = wilson_ci(k, n)
    return {"k": k, "n": n, "pct": round(k / n * 100, 1) if n else None,
            "ci95_low": round(lo, 1), "ci95_high": round(hi, 1)}


def _stratified(queries: list[dict], k: int) -> list[dict]:
    if k <= 0 or k >= len(queries):
        return queries
    by_tool: dict[str, list[dict]] = {}
    for r in queries:
        by_tool.setdefault(r["ref"], []).append(r)
    out: list[dict] = []
    per = max(1, k // len(by_tool))
    for _t, rs in by_tool.items():
        out.extend(rs[:per])
    return out[:k]


async def _classify(registry: ModelRegistry, summary: str, model: str,
                   sem, q: str) -> set[str]:
    prompt = LLM_TOOL_CLASSIFY_PROMPT.format(user_input=q, tool_summary=summary)
    for attempt in range(5):
        async with sem:
            try:
                resp = await asyncio.wait_for(registry.chat(
                    model_key=model,
                    messages=[{"role": "system", "content": "你是工具选择专家，只输出JSON。"},
                              {"role": "user", "content": prompt}],
                    max_tokens=500, max_retries=1, session_id="n1_multidet"), timeout=TIMEOUT)
                raw = gen._extract_content(resp)
                if raw:
                    return set(_parse_llm_classify_json(raw)["recommended_tools"])
            except Exception:  # noqa: BLE001
                pass
            await asyncio.sleep(1.2 * (attempt + 1))
    return set()


async def main_async() -> None:
    queries = json.loads(QUERIES_JSON.read_text(encoding="utf-8"))["queries"]
    sample = _stratified(queries, N_DET)
    scaled = json.loads(SCALED_JSON.read_text(encoding="utf-8"))
    primary_rec = {r["id"]: set(r["rec_tools"]) for r in scaled["round2"]["rows"]}

    reg = ToolRegistry(); reg.load_config(); reg.auto_discover(lazy=True)
    registered = set(reg.list_all_tool_names())
    registry = ModelRegistry(config_path=ROOT / "config" / "models.toml")
    summary = build_tool_summary(reg)
    sem = asyncio.Semaphore(CONCURRENCY)

    detectors = [("primary:" + INTENT_LLM_MODEL, None)] + \
                [(m, m) for m in EXTRA_MODELS]

    print("=" * 96, flush=True)
    print(f"N1 E1 多检测器复制  样本={len(sample)}  检测器={[d[0] for d in detectors]}", flush=True)
    print("=" * 96, flush=True)

    # 预取额外检测器结果（并发 + 增量断点 + 续跑）
    ckpt: dict[str, dict[str, list]] = {}
    if CKPT_JSON.exists():
        try:
            ckpt = json.loads(CKPT_JSON.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            ckpt = {}

    def _save_ckpt() -> None:
        CKPT_JSON.write_text(json.dumps(ckpt, ensure_ascii=False), encoding="utf-8")

    async def _one(rid: int, q: str, mdl: str) -> tuple[int, set]:
        return rid, await _classify(registry, summary, mdl, sem, q)

    per_model_rec: dict[str, dict[int, set]] = {}
    for label, model in detectors:
        if model is None:
            per_model_rec[label] = {r["id"]: primary_rec.get(r["id"], set()) for r in sample}
            continue
        done = ckpt.setdefault(model, {})
        recs = {r["id"]: set(done.get(str(r["id"]), [])) for r in sample}
        todo = [r for r in sample if str(r["id"]) not in done]
        print(f"  [{label}] 待跑 {len(todo)} / {len(sample)}（已续 {len(sample) - len(todo)}）", flush=True)
        cnt = 0
        futs = [asyncio.ensure_future(_one(r["id"], r["q"], model)) for r in todo]
        for fut in asyncio.as_completed(futs):
            rid, res = await fut
            recs[rid] = res
            done[str(rid)] = sorted(res)
            cnt += 1
            if cnt % 25 == 0:
                _save_ckpt()
                print(f"  [{label}] checkpoint {cnt}/{len(todo)}", flush=True)
        _save_ckpt()
        per_model_rec[label] = recs
        print(f"  [{label}] 完成 {len(recs)} 条识别", flush=True)

    # 聚合
    agg: dict[str, dict] = {}
    rows_by_model: dict[str, list[dict]] = {}
    for label, _model in detectors:
        rec_map = per_model_rec[label]
        rec_hit = exposed = gap = valid = 0
        rows = []
        for r in sample:
            if r["ref"] not in registered:
                continue
            valid += 1
            rh = r["ref"] in rec_map.get(r["id"], set())
            ex = bool(r["rule_hit_before"])          # 检测器无关暴露
            gp = bool(rh and not ex)
            rec_hit += rh; exposed += ex; gap += gp
            rows.append({"id": r["id"], "q": r["q"], "ref": r["ref"],
                         "rec_hit": rh, "exposed": ex, "gap": gp,
                         "rec_n": len(rec_map.get(r["id"], set()))})
        agg[label] = {
            "n_valid": valid,
            "rec_hit_rate": _wilson(rec_hit, valid),
            "exposed_rate(detector-independent)": _wilson(exposed, valid),
            "gap_rate": _wilson(gap, valid),
            "rec_tools_mean": round(_mean([x["rec_n"] for x in rows]), 1),
        }
        rows_by_model[label] = rows

    result = {
        "meta": {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                 "base_tag": "n1-eval-v1", "n_sample": len(sample),
                 "detectors": [d[0] for d in detectors],
                 "note": "exposed(rule RECOMMENDED) 与检测器无关；跨检测器差异仅来自识别"},
        "aggregate": agg, "rows": rows_by_model,
    }
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = []
    A = lines.append
    A("=" * 96)
    A(f"N1 · E1 多检测器复制（样本 n={len(sample)}, 分层）· gap 跨模型稳健性")
    A(f"检测器: {result['meta']['detectors']}")
    A("=" * 96)
    for label, a in agg.items():
        A(f"\n[{label}]")
        for key in ("rec_hit_rate", "exposed_rate(detector-independent)", "gap_rate"):
            c = a[key]
            A(f"  {key:<40} {c['k']}/{c['n']} = {c['pct']}%  "
              f"Wilson95%CI[{c['ci95_low']}, {c['ci95_high']}]")
        A(f"  推荐集均值大小={a['rec_tools_mean']}")
    report = "\n".join(lines)
    OUT_TXT.write_text(report, encoding="utf-8")
    try:
        sys.__stdout__.write(report + "\n")
    except Exception:  # noqa: BLE001
        pass
    print(f"\n[written] {OUT_JSON}\n[written] {OUT_TXT}", flush=True)


if __name__ == "__main__":
    asyncio.run(main_async())
