# -*- coding: utf-8 -*-
r"""N1 · P0-3 —— 第二标注者盲标 + Cohen's Kappa（验证 ref 无歧义 / 破循环性）。

评审 M4 关切：ref 由作者自定 → 循环性风险。
本脚本用**异提供商的第二 LLM（glm-4.7-flash）**在**不知道生成种子**的前提下，
从 25 个候选工具中为每条 query 选"最合适的一个"，与 by-construction 的 ref 比对：
  · raw agreement = 完全一致比例；
  · Cohen's Kappa = 机会校正后一致性（多类）。
高 Kappa ⇒ "该 query 唯一指向 ref"这一标注客观可信，非作者主观。

产物：test_scripts/rag_experiments/n1_kappa.json（逐条 annot2 + 汇总）
运行：.\.venv\Scripts\python.exe test_scripts\rag_experiments\n1_annotate_kappa.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(EXP.parents[1]))  # repo root

import n1_expand_queries as gen  # noqa: E402  复用 TARGET_TOOLS / _extract_content / _parse / env

from src.models.registry import ModelRegistry  # noqa: E402

ANNOT_MODEL = os.environ.get("ANNOT_MODEL", "qwen-max")  # 异提供商第二标注者（默认阿里巴巴 qwen）
CONCURRENCY = int(os.environ.get("KAPPA_CONC", "8"))
CKPT_EVERY = 25
OUT_JSON = EXP / "n1_kappa.json"

CANDS = gen.TARGET_TOOLS  # 25 候选工具：name -> desc


def _cand_block() -> str:
    return "\n".join(f"- {name}：{desc}" for name, desc in CANDS.items())


PROMPT = (
    "下面是一个桌面 AI 助手可用的工具清单（工具名：功能）：\n{cands}\n\n"
    "用户请求：{q}\n\n"
    "请从上述工具中选出**最合适完成该请求的唯一一个**工具。"
    '只输出 JSON：{{"tool": "工具名"}}，工具名必须严格来自清单。'
)


def _parse_tool(raw: str) -> str:
    txt = raw.strip()
    if "```" in txt:
        import re
        m = re.search(r"```(?:json)?\s*(.*?)```", txt, __import__("re").S)
        if m:
            txt = m.group(1).strip()
    s, e = txt.find("{"), txt.rfind("}")
    if s >= 0 and e > s:
        try:
            return str(json.loads(txt[s:e + 1]).get("tool", "")).strip()
        except Exception:  # noqa: BLE001
            pass
    # 兜底：直接匹配清单里出现的工具名
    for name in CANDS:
        if name in txt:
            return name
    return ""


def _cohen_kappa(a: list[str], b: list[str]) -> tuple[float, float]:
    labels = sorted(set(a) | set(b))
    n = len(a)
    if n == 0:
        return float("nan"), float("nan")
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    ca = {l: a.count(l) for l in labels}
    cb = {l: b.count(l) for l in labels}
    pe = sum((ca[l] / n) * (cb[l] / n) for l in labels)
    k = (po - pe) / (1 - pe) if pe < 1 else 1.0
    return po, k


async def _annotate_one(registry: ModelRegistry, sem, q: str) -> str:
    prompt = PROMPT.format(cands=_cand_block(), q=q)
    async with sem:
        last = ""
        for attempt in range(5):
            try:
                resp = await asyncio.wait_for(
                    registry.chat(
                        model_key=ANNOT_MODEL,
                        messages=[
                            {"role": "system", "content": "你只输出合法 JSON。"},
                            {"role": "user", "content": prompt},
                        ],
                        max_tokens=64, max_retries=1, session_id="n1_kappa",
                    ),
                    timeout=40.0,
                )
                raw = gen._extract_content(resp)
                t = _parse_tool(raw)
                if t in CANDS:
                    return t
                last = f"invalid(raw={raw[:30]!r})"
            except Exception as e:  # noqa: BLE001
                last = f"{type(e).__name__}:{str(e)[:40]}"
            await asyncio.sleep(1.2 * (attempt + 1))
        print(f"  [annot-fail] {q[:16]} : {last}")
        return ""


def _write(queries: list[dict], done_ids: set) -> None:
    a, b = [], []
    for r in queries:
        t = r.get("annot2_ref", "")
        if t:
            a.append(r["ref"]); b.append(t)
    po, k = _cohen_kappa(a, b)
    agree = sum(1 for x, y in zip(a, b) if x == y)
    n_ok = len(a)
    summary = {
        "annot_model": ANNOT_MODEL, "n_total": len(queries), "n_annotated": n_ok,
        "raw_agreement": round(agree / n_ok * 100, 1) if n_ok else None,
        "cohen_kappa": round(k, 3) if k == k else None,
        "n_disagree": n_ok - agree,
    }
    OUT_JSON.write_text(
        json.dumps({"summary": summary, "queries": queries}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


async def main_async() -> None:
    data = json.loads((EXP / "n1_queries_500.json").read_text(encoding="utf-8"))
    queries = data["queries"]

    # 断点续跑：加载已有 n1_kappa.json，保留已标(非空)结果，仅补缺口
    if OUT_JSON.exists():
        try:
            prev = {r["id"]: r.get("annot2_ref", "")
                    for r in json.loads(OUT_JSON.read_text(encoding="utf-8"))["queries"]}
            for r in queries:
                if prev.get(r["id"]):
                    r["annot2_ref"] = prev[r["id"]]
        except Exception:  # noqa: BLE001
            pass
    todo = [r for r in queries if not r.get("annot2_ref")]

    registry = ModelRegistry(config_path=gen.ROOT / "config" / "models.toml")
    sem = asyncio.Semaphore(CONCURRENCY)

    print("=" * 90, flush=True)
    print(f"N1 第二标注者盲标  model={ANNOT_MODEL}  总数={len(queries)}  "
          f"待标={len(todo)}  已标={len(queries) - len(todo)}  并发={CONCURRENCY}", flush=True)
    print("=" * 90, flush=True)

    done = 0

    async def _wrap(rec: dict) -> tuple[dict, str]:
        t = await _annotate_one(registry, sem, rec["q"])
        return rec, t

    futs = [asyncio.ensure_future(_wrap(r)) for r in todo]
    for fut in asyncio.as_completed(futs):
        r, t = await fut
        r["annot2_ref"] = t
        done += 1
        if done % CKPT_EVERY == 0:
            _write(queries, None)
            print(f"  ...{done}/{len(todo)} (ckpt)", flush=True)

    _write(queries, None)
    s = json.loads(OUT_JSON.read_text(encoding="utf-8"))["summary"]
    print("\n" + "-" * 90, flush=True)
    print(f"成功标注={s['n_annotated']}/{s['n_total']}  原始一致={s['raw_agreement']}%  "
          f"Cohen's Kappa={s['cohen_kappa']}  分歧={s['n_disagree']}", flush=True)
    print("[written] " + str(OUT_JSON), flush=True)


if __name__ == "__main__":
    asyncio.run(main_async())
