# WeClaw · Recognition-Exposure Gap (N1) — Reproduction Artifact

Anonymized repository released for the **double-blind review** of the
manuscript *"The Recognition-Exposure Gap: Why Upgrading Intent Detection
Can Reduce Tool Availability in LLM Agents"* (ETRI Journal submission).

All identity strings (author, affiliation, real email, real repo URL) are
excluded from this mirror per the journal's double-blind policy; the
reviewer should treat this repository as the fixed source snapshot
tagged **`n1-eval-v1`** referenced in the manuscript's Reproducibility
Statement.

## Layout

```
src/core/tool_exposure.py             # the studied production code paths
test_scripts/rag_experiments/
  n1_expand_queries.py                # 500-query set builder (round1)
  n1_scaled_experiment.py             # round2 real LLM + round3 deterministic replay
  n1_live_experiment.py               # real end-to-end (every-round LLM call, measured wall-clock)
  n1_annotate_kappa.py                # blind second-annotator (qwen-max) + Cohen's kappa
  n1_multidetector.py                 # three-detector replication on 150-query stratified subsample
  n1_ossfw_reproduction.py            # reference reproduction inside three open-source exposure strategies
  gen_n1_paper_figures.py             # regenerates fig1..fig5 PDFs used in the manuscript
  n1_queries_500.json                 # seed queries + ground-truth ref (by construction)
  n1_scaled_results.json              # round2 + round3 (before/after arms) with Wilson CIs
  n1_live_results.json                # live end-to-end per-query records
  n1_kappa.json                       # raw second-annotator labels + agreement
  n1_multidetector_results.json       # per-detector rec_hit / exposed / gap
  n1_ossfw_results.json               # per-strategy coverage / gap / exposure
  n1_*.txt                            # concise human-readable summaries
paper/
  paper_etri.tex                      # main manuscript entry (double-blind)
  paper_body_etri.tex                 # \input body
  references_list.tex                 # numbered reference list (appearance order)
  references_anon.bib                 # underlying bib sources for reference generation
  paper_etri.pdf                      # compiled manuscript (10 pages)
  figures/                            # PDF + PNG vector exports of fig1..fig5
```

## How the reported numbers map to files

| Manuscript claim | Source of truth |
|---|---|
| 98.2% rec_hit (491/500) [96.6, 99.1] | `n1_scaled_results.json` → `round2.rec_hit` |
| 42.2% exposed (211/500) [37.9, 46.6] | `n1_scaled_results.json` → `round2.exposed_before` |
| **56.2% gap** (281/500) [51.8, 60.5] | `n1_scaled_results.json` → `round2.gap` |
| 98.4% coverage under merge fix | `n1_scaled_results.json` → `round2.exposed_after` |
| 788 failed round-trips / 499 escalations | `n1_scaled_results.json` → `round3.before` |
| Live latency 11,213 ms → 4,251 ms (2.64×) | `n1_live_results.json` → aggregates |
| Cohen's kappa = 0.987 (raw 98.8%) | `n1_kappa.json` → `metrics` |
| Three-detector gap 52.0 / 51.3 / 42.0% | `n1_multidetector_results.json` → aggregate |
| OSS reproduction (router / all-tools / top-K) | `n1_ossfw_results.json` → adapters |

## Reproduction procedure

1. **Environment**: Python 3.12 with `openai`, `tiktoken`, `requests`, `matplotlib`
   (see main project `pyproject.toml` for pinned versions).
2. **Fixed source snapshot**: the studied production functions
   (`_determine_tier`, `_get_tool_names_for_tier`, `report_failure`,
   `_upgrade_tier`, `_augment_with_llm_recommended`) live in
   `src/core/tool_exposure.py`; the harness scripts assume this path.
3. **Round 1 (query expansion)**: `python n1_expand_queries.py` →
   regenerates `n1_queries_500.json` (uses the DeepSeek generator; requires
   a valid API key via `DEEPSEEK_API_KEY`).
4. **Round 2 + Round 3 (offline counterfactual + deterministic replay)**:
   `python n1_scaled_experiment.py` → regenerates `n1_scaled_results.json`.
5. **Live end-to-end**: `python n1_live_experiment.py` → regenerates
   `n1_live_results.json` (real per-round LLM calls, measured wall-clock).
6. **Kappa (blind second annotator)**: `python n1_annotate_kappa.py` →
   regenerates `n1_kappa.json` (uses `qwen-max`; requires `QWEN_API_KEY`).
7. **Multi-detector replication**: `python n1_multidetector.py` →
   regenerates `n1_multidetector_results.json` (stratified 150-query
   subsample, three different-provider detectors).
8. **Open-source reference reproduction**: `python n1_ossfw_reproduction.py`
   → regenerates `n1_ossfw_results.json` (zero external framework deps;
   adapters implemented from scratch against the same round2 recognition
   signal).
9. **Figures**: `python gen_n1_paper_figures.py` → regenerates fig1..fig5
   PDFs from the JSON artifacts.

## API key handling

No API key or credential file is checked in. Each script reads keys from
standard environment variables at run time (`DEEPSEEK_API_KEY`,
`QWEN_API_KEY`, `GLM_API_KEY`); scripts are non-runnable without user-
provided keys.

## Anonymity statement

The mirror has been scrubbed of:

- author names, emails, affiliations;
- the parent project's public source URL (available in the separate
  title page upload; withheld here per double-blind policy);
- absolute filesystem paths from the development environment;
- runtime `.env`, database, log, and cache artifacts;
- the personal developer identity of any commit author or tagger (this
  repo has a fresh, single-commit history by an anonymous persona).

## License

The studied agent (WeClaw) itself is distributed under its project
license (see parent repo). This artifact subset — the paper's evaluation
harness, result JSONs, and manuscript — is released under MIT for the
purpose of peer review; re-use outside peer review requires contacting
the (currently anonymous) authors via the handling editor.
