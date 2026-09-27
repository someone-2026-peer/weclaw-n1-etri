#!/usr/bin/env python
"""Generate all figures for the N1 workshop short paper
"The Recognition-Exposure Gap: Why Upgrading Intent Detection Can Reduce
Tool Availability in LLM Agents".

Channel routing per skills/research/figure-recommender.md:
  - Fig2 / Fig3 / Fig5 (data charts, channel B): matplotlib + SciencePlots(ieee,no-latex)
  - Fig1 / Fig4 (framework / timeline, channel C): matplotlib free-draw
Output: figures/figN_*.{pdf,png}  (vector PDF + 300 dpi raster).

Deterministic, no network, no LLM. Numbers come from rag_experiments
round2 (offline counterfactual) and round3 (execution-level trigger chain),
i.e. the same data as paper_draft.md Table 1 / Table 2 / Table 3.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless

from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

# ---- SciencePlots (optional; graceful fallback to default) ----
try:
    import scienceplots  # noqa: F401

    _SCI = ["science", "no-latex"]
    _SCII = ["science", "ieee", "no-latex"]
except Exception:  # pragma: no cover
    _SCI = []
    _SCII = []

REPO = Path(__file__).resolve().parents[2]
OUT = (
    REPO
    / "docs"
    / "8 计划发布的论文papers"
    / "weclaw_n1_recog_expo_gap_paper_20260911"
    / "etri_submission"
    / "figures"
)
OUT.mkdir(parents=True, exist_ok=True)

# ---- palette (colorblind-safe, grayscale-readable) ----
C_REC = "#3b6ea5"  # recognition / normal stage (blue)
C_GAP = "#c0504d"  # gap / before / cost (red)
C_AFTER = "#4f9a51"  # after / fix / recovered (green)
C_NEUT = "#5a5a5a"  # neutral
FILL_STAGE = "#eaf0f7"
FILL_GAP = "#f6e3e2"
FILL_FIX = "#e6f2e6"
FILL_NEUT = "#f2f2f2"
DPI = 300


def _force_font() -> None:
    """Guarantee DejaVu Sans so ✗ ✓ ∅ ∉ ∧ ¬ ≥ → · — all render (no tofu),
    and keep usetex off so underscores in rec_hit / _augment_... are literal."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "text.usetex": False,
        }
    )


def _save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] figures/{name}.pdf  +  .png")


def _box(ax, xy, text, fc, ec, fs=6.4, weight="normal"):
    x, y = xy
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        fontweight=weight,
        color="#1a1a1a",
        bbox=dict(boxstyle="round,pad=0.45", fc=fc, ec=ec, lw=0.9),
        zorder=4,
    )


def _arrow(ax, p0, p1, color="#1a1a1a", ls="-", lw=1.0, rad=0.0, ms=11, z=3):
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle="-|>",
            mutation_scale=ms,
            color=color,
            lw=lw,
            linestyle=ls,
            zorder=z,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=1.0,
            shrinkB=1.0,
        )
    )


# =====================================================================
# Fig 2 — Recognition vs Exposure divergence (headline result, Table 1)
# =====================================================================
def fig2_recognition_exposure() -> None:
    with plt.style.context(_SCII if _SCII else "default"):
        _force_font()
        fig, ax = plt.subplots(figsize=(3.6, 2.9))
        labels = ["Recognition\n(rec_hit)", "Exposure\n(before fix)", "Exposure\n(after fix)"]
        vals = [98.2, 42.2, 98.4]
        counts = ["491/500", "211/500", "492/500"]
        # Wilson 95% CI half-widths (asymmetric)
        ci = [(96.6, 99.1), (37.9, 46.6), (96.9, 99.2)]
        yerr = [
            [v - lo for v, (lo, _hi) in zip(vals, ci)],
            [hi - v for v, (_lo, hi) in zip(vals, ci)],
        ]
        cols = [C_REC, C_GAP, C_AFTER]
        x = np.arange(3)
        ax.bar(
            x, vals, color=cols, width=0.6, zorder=3, edgecolor="black", linewidth=0.5,
            yerr=yerr, ecolor=C_NEUT, error_kw=dict(lw=0.8, capsize=3, capthick=0.8),
        )
        for i, (v, c) in enumerate(zip(vals, counts)):
            ax.text(i, v + 3.0, f"{v:.1f}%\n({c})", ha="center", va="bottom", fontsize=6.4)
        # GAP bracket between recognition (98.2%) and exposure-before (42.2%)
        ax.annotate(
            "",
            xy=(1, 42.2),
            xytext=(0, 98.2),
            arrowprops=dict(arrowstyle="<->", color=C_GAP, lw=1.0, linestyle="--"),
        )
        ax.text(
            0.5,
            76,
            "GAP\n56.2%\n(281/500)",
            ha="center",
            va="center",
            fontsize=6.6,
            color=C_GAP,
            fontweight="bold",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=6.4)
        ax.set_ylabel("Coverage (%, Wilson 95% CI)", fontsize=7)
        ax.set_ylim(0, 122)
        ax.grid(axis="y", ls=":", lw=0.4, zorder=0)
        _save(fig, "fig2_recognition_exposure_gap")


# =====================================================================
# Fig 3 — Execution-level runtime cost, before/after (Table 2)
# =====================================================================
def fig3_execution_cost() -> None:
    with plt.style.context(_SCII if _SCII else "default"):
        _force_font()
        fig, (axa, axb) = plt.subplots(1, 2, figsize=(6.9, 2.9))
        w = 0.36

        def _pair(ax, labels_b, before, after, ylab, ylim, lbl_fmt, show_legend):
            x = np.arange(len(labels_b))
            b1 = ax.bar(
                x - w / 2, before, w, label="before (merge off)",
                color=C_GAP, edgecolor="black", linewidth=0.5, zorder=3,
            )
            b2 = ax.bar(
                x + w / 2, after, w, label="after (merge on)",
                color=C_AFTER, edgecolor="black", linewidth=0.5, zorder=3,
            )
            for bars in (b1, b2):
                for r in bars:
                    h = r.get_height()
                    ax.text(
                        r.get_x() + r.get_width() / 2, h + ylim * 0.02, lbl_fmt(h),
                        ha="center", va="bottom", fontsize=6.0,
                    )
            ax.set_xticks(x)
            ax.set_xticklabels(labels_b, fontsize=6.2)
            ax.set_ylabel(ylab, fontsize=7)
            ax.set_ylim(0, ylim)
            ax.grid(axis="y", ls=":", lw=0.4, zorder=0)
            if show_legend:
                ax.legend(fontsize=6, frameon=True, loc="upper right")

        # (a) escalation events, totals over n=500 (deterministic replay)
        _pair(
            axa, ["Failed\nround-trips", "Tier\nescalations"],
            [788.0, 499.0], [0.0, 0.0],
            "Count (total, n=500)", 900, lambda h: f"{int(h)}", True,
        )
        axa.set_title("(a) escalation events (deterministic replay)", fontsize=6.8)
        # (b) per-query cost: exposure inflation + measured live wall-clock
        _pair(
            axb, ["Mean final-tier\nexposure (tools)", "Measured\nwall-clock (s)"],
            [44.7, 11.2], [12.7, 4.3],
            "Per-query value", 52, lambda h: f"{h:g}", False,
        )
        axb.set_title("(b) per-query cost (live, deepseek-v4-flash)", fontsize=6.8)
        fig.tight_layout()
        _save(fig, "fig3_execution_cost")


# =====================================================================
# Fig 1 — Mechanism: two-stage decoupling (channel C, framework diagram)
# =====================================================================
def fig1_mechanism() -> None:
    with plt.style.context(_SCI if _SCI else "default"):
        _force_font()
        fig, ax = plt.subplots(figsize=(7.6, 3.4))
        ax.set_xlim(0, 120)
        ax.set_ylim(0, 40)
        ax.axis("off")

        q = (7, 32)
        b1 = (27, 32)
        b2 = (52, 32)
        b3 = (79, 32)
        b4 = (106, 32)
        _box(ax, q, "colloquial\nquery", FILL_NEUT, C_NEUT, fs=6.0)
        _box(
            ax, b1, "Intent detector\nrule → LLM upgrade\nrecommends ref",
            FILL_STAGE, C_REC, fs=6.0,
        )
        _box(ax, b2, "Stage A\nconfidence → tier\n= RECOMMENDED", FILL_STAGE, C_REC, fs=6.2)
        _box(ax, b3, "Stage B\ntier → tool set\n(rule map = ∅)", FILL_GAP, C_GAP, fs=6.2)
        _box(ax, b4, "Exposure set\nref ∉ set →\nagent 1st call ✗", FILL_GAP, C_GAP, fs=6.0)

        _arrow(ax, (12.5, 32), (18.5, 32))
        _arrow(ax, (35.5, 32), (44.0, 32))
        ax.text(39.5, 34.4, "conf 0.9", fontsize=5.8, ha="center", color=C_NEUT)
        _arrow(ax, (60.0, 32), (71.5, 32))
        ax.text(65.5, 34.4, "tier", fontsize=5.8, ha="center", color=C_NEUT)
        _arrow(ax, (86.5, 32), (98.0, 32))

        # rec-tools payload node below Stage B (provenance stated in label,
        # no horizontal line to detector -> cannot be misread as a bypass of Stage A)
        _box(
            ax, (79, 19), "llm_recommended_tools\n(from Intent detector)",
            FILL_STAGE, C_REC, fs=5.8,
        )
        # merge OFF (as-shipped): red dashed side input stops short of Stage B
        _arrow(ax, (75.5, 20.8), (75.5, 26.3), color=C_GAP, ls=(0, (4, 2)), lw=1.3, ms=12)
        ax.text(75.5, 27.6, "✗", fontsize=11, ha="center", va="center",
                color=C_GAP, fontweight="bold")
        ax.text(65.5, 24.0, "OFF (as-shipped)", fontsize=5.6, ha="center",
                va="center", color=C_GAP, fontweight="bold")
        # merge ON (fix): green solid side input reaches Stage B
        _arrow(ax, (82.5, 20.8), (82.5, 29.2), color=C_AFTER, ls="-", lw=1.4, ms=12)
        ax.text(85.3, 25.0, "✓", fontsize=9, ha="center", va="center",
                color=C_AFTER, fontweight="bold")
        ax.text(90.0, 24.0, "ON (fix)", fontsize=5.6, ha="center",
                va="center", color=C_AFTER, fontweight="bold")
        # captions + legend
        ax.text(
            35, 14,
            "Recognition–Exposure Gap\nrec tools NOT merged into Stage B set\n(rec_hit ∧ ¬exposed)",
            ha="center", va="center", fontsize=6.2, color=C_GAP, fontweight="bold",
        )
        ax.text(
            35, 6.5,
            "✓ fix: _augment_with_llm_recommended\nmerges rec tools into Stage B set",
            ha="center", va="center", fontsize=6.2, color=C_AFTER, fontweight="bold",
        )
        ax.text(
            60, 1.5,
            "black = pipeline (runs in both states; Stage A never skipped) · "
            "red/green = rec-tools side input into Stage B set",
            ha="center", va="center", fontsize=5.4, color=C_NEUT,
        )
        _save(fig, "fig1_mechanism")


# =====================================================================
# Fig 4 — Representative trigger chain / escalation ladder (channel C)
# =====================================================================
def fig4_trigger_chain() -> None:
    with plt.style.context(_SCI if _SCI else "default"):
        _force_font()
        fig, ax = plt.subplots(figsize=(7.4, 3.1))
        ax.set_xlim(0, 104)
        ax.set_ylim(0, 48)
        ax.axis("off")

        ax.text(
            2, 45, 'Query: "what time is it now"   (ref = datetime_tool)',
            fontsize=7.2, ha="left", va="center", fontweight="bold",
        )

        # ---- BEFORE row ----
        ax.text(
            2, 38, "BEFORE (merge off):", fontsize=6.6, ha="left", va="center",
            color=C_GAP, fontweight="bold",
        )
        yb = 30
        rt1, rt2, rt3 = (16, yb), (48, yb), (82, yb)
        _box(ax, rt1, "rt1 · RECOMMENDED\n|exposed| = 5\nref reachable: ✗", FILL_GAP, C_GAP, fs=6.0)
        _box(
            ax, rt2, "rt2 · RECOMMENDED\n|exposed| = 5\nref ✗ → report_failure (2/2)",
            FILL_GAP, C_GAP, fs=6.0,
        )
        _box(ax, rt3, "rt3 · EXTENDED\n|exposed| = 31\nref ✓ recovered", FILL_FIX, C_AFTER, fs=6.0)
        _arrow(ax, (26, yb), (37, yb), color=C_GAP)
        ax.text(31.5, yb + 2.6, "fail (1)", fontsize=5.8, ha="center", color=C_GAP)
        _arrow(ax, (59, yb), (71, yb), color=C_GAP)
        ax.text(
            65, yb + 3.0, "_upgrade_tier\n(threshold 2)",
            fontsize=5.8, ha="center", color=C_GAP,
        )
        ax.text(
            50, 21.5,
            "2 failed round-trips + 1 escalation  →  $\\approx$8.3 s added latency (measured)",
            fontsize=6.2, ha="center", va="center", color=C_GAP, style="italic",
        )

        # ---- AFTER row ----
        ax.text(
            2, 14, "AFTER (merge on):", fontsize=6.6, ha="left", va="center",
            color=C_AFTER, fontweight="bold",
        )
        ya = 7
        _box(
            ax, (16, ya), "rt1 · RECOMMENDED\nref MERGED into set\nref ✓ first attempt",
            FILL_FIX, C_AFTER, fs=6.0,
        )
        ax.text(
            30, ya,
            "0 failed round-trips · 0 escalations · 0 added latency · ref merged in place (|set| 5→6 vs detour to 31)",
            fontsize=6.2, ha="left", va="center", color=C_AFTER,
        )
        _save(fig, "fig4_trigger_chain")


# =====================================================================
# Fig 5 (supplementary) — FULL-tier floor ablation, round0 vs round1
# =====================================================================
def fig5_floor_ablation() -> None:
    with plt.style.context(_SCII if _SCII else "default"):
        _force_font()
        fig, ax = plt.subplots(figsize=(3.9, 2.9))
        metrics = [
            "Escalation trigger\nrate (FULL tier)", "Retrieval\nzero-hit rate",
        ]
        base = [50.0, 20.0]  # round0: rule intent, floor removed
        kw = [0.0, 5.0]  # round1: + colloquial-keyword back-fill
        base_lbl = ["50.0%\n(4/8)", "20.0%\n(4/20)"]
        kw_lbl = ["0.0%\n(0/8)", "5.0%\n(1/20)"]
        x = np.arange(len(metrics))
        w = 0.36
        b1 = ax.bar(
            x - w / 2, base, w, label="round0 baseline (rule)",
            color=C_GAP, edgecolor="black", linewidth=0.5, zorder=3,
        )
        b2 = ax.bar(
            x + w / 2, kw, w, label="round1 +colloquial keywords",
            color=C_AFTER, edgecolor="black", linewidth=0.5, zorder=3,
        )
        for bars, lbls in ((b1, base_lbl), (b2, kw_lbl)):
            for r, lb in zip(bars, lbls):
                h = r.get_height()
                ax.text(
                    r.get_x() + r.get_width() / 2, h + 1.2, lb,
                    ha="center", va="bottom", fontsize=6.0,
                )
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=6.2)
        ax.set_ylabel("Rate (%)", fontsize=7)
        ax.set_ylim(0, 62)
        ax.legend(fontsize=5.8, frameon=True, loc="upper right")
        ax.grid(axis="y", ls=":", lw=0.4, zorder=0)
        ax.text(
            0.5, -0.30,
            "† round1 keywords mirror eval query wording → 0% may\n"
            "overfit; hold-out validation required before dropping floor.",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=5.4, color=C_NEUT, style="italic",
        )
        _save(fig, "fig5_floor_ablation")


def main() -> None:
    print(f"[gen] N1 paper figures → {OUT}")
    fig1_mechanism()
    fig2_recognition_exposure()
    fig3_execution_cost()
    fig4_trigger_chain()
    fig5_floor_ablation()
    print(f"[done] 5 figures generated (PDF + PNG@{DPI}dpi)")


if __name__ == "__main__":
    main()
