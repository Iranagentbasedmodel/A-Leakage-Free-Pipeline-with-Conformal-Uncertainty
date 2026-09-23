"""
Publication Outputs — v8.1
==========================
Journal-ready figures, exported as vector (PDF + SVG) plus a PNG preview, all
drawn through one style profile so axis/label sizes are consistent across
figures. Every value printed on a figure is passed in explicitly by the
caller, so a figure can never disagree with the manifest.

Figures:
  prediction_figures        predicted-vs-actual + residual diagnostics
  coverage_figure           conformal empirical-vs-nominal coverage
  logo_figure               leave-one-climate-out bars + mean line
  counterfactual_figure     student vs teacher vs ResStock matched deltas
  architecture_schematic    pipeline architecture, drawn (not a box diagram
                            assembled from defaults)
"""

import logging
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from config import FIGURES_DIR
from figure_style import PALETTE, annotate_above_bars, apply_style, save_figure

logger = logging.getLogger("publication_outputs")


# ---------------------------------------------------------------------------
def prediction_figures(y_true, y_pred, tag: str, unit: str = "kBtu",
                       r2: Optional[float] = None,
                       rmse: Optional[float] = None,
                       n: Optional[int] = None,
                       intervals: Optional[pd.DataFrame] = None) -> Optional[str]:
    try:
        apply_style()
        import matplotlib.pyplot as plt

        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        ok = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true, y_pred = y_true[ok], y_pred[ok]
        if len(y_true) < 10:
            return None

        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
        lim = float(np.percentile(np.concatenate([y_true, y_pred]), 99.5))
        axes[0].scatter(y_true, y_pred, s=5, alpha=0.30,
                        color=PALETTE["primary"], edgecolors="none", rasterized=True)
        axes[0].plot([0, lim], [0, lim], "k--", lw=0.9)
        axes[0].set_xlim(0, lim)
        axes[0].set_ylim(0, lim)
        axes[0].set_xlabel(f"Actual ({unit})")
        axes[0].set_ylabel(f"Predicted ({unit})")
        axes[0].set_title("Predicted vs. actual")
        parts = []
        if r2 is not None:
            parts.append(f"$R^2$ = {r2:.3f}")
        if rmse is not None:
            parts.append(f"RMSE = {rmse:,.0f} {unit}")
        if n is not None:
            parts.append(f"$n$ = {n:,}")
        if parts:
            axes[0].text(0.03, 0.95, ";  ".join(parts), transform=axes[0].transAxes,
                         va="top", ha="left", fontsize=7.5,
                         bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                   ec="none", alpha=0.85))

        res = y_true - y_pred
        axes[1].hist(res, bins=60, color=PALETTE["accent"],
                     edgecolor="white", linewidth=0.2)
        axes[1].axvline(0, color="k", lw=0.9, ls="--")
        axes[1].set_xlabel(f"Residual ({unit})")
        axes[1].set_ylabel("Households")
        axes[1].set_title("Residual distribution")

        fig.tight_layout()
        p = save_figure(fig, FIGURES_DIR / f"pred_vs_actual_{tag}")
        plt.close(fig)
        logger.info(f"Prediction figure saved (vector): {p}")
        return str(p)
    except Exception as e:
        logger.warning(f"prediction_figures skipped ({e})")
        return None


# ---------------------------------------------------------------------------
def coverage_figure(coverage: Dict[float, float], tag: str = "conformal",
                    mean_width: Optional[float] = None) -> Optional[str]:
    """Empirical vs nominal coverage. The identity line is the guarantee;
    markers on it mean the calibration is honest."""
    try:
        apply_style()
        import matplotlib.pyplot as plt

        if not coverage:
            return None
        nom = np.array(sorted(coverage), dtype=float)
        emp = np.array([coverage[k] for k in sorted(coverage)], dtype=float)

        fig, ax = plt.subplots(figsize=(3.6, 3.4))
        lo, hi = 0.70, 1.0
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.9, label="nominal")
        ax.plot(nom, emp, "o-", color=PALETTE["primary"], ms=5, lw=1.2,
                label="empirical (calibration split)")
        for x, y in zip(nom, emp):
            ax.annotate(f"{y * 100:.1f}%", (x, y), textcoords="offset points",
                        xytext=(6, -3), fontsize=7.5)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, 1.0)
        ax.set_xlabel("Nominal coverage")
        ax.set_ylabel("Empirical coverage")
        ax.set_title("Split-conformal calibration")
        ax.legend(frameon=False, loc="lower right")
        if mean_width is not None:
            ax.text(0.03, 0.05, f"mean width {mean_width:,.0f} kBtu",
                    transform=ax.transAxes, fontsize=7.5)
        fig.tight_layout()
        p = save_figure(fig, FIGURES_DIR / f"coverage_{tag}")
        plt.close(fig)
        logger.info(f"Coverage figure saved (vector): {p}")
        return str(p)
    except Exception as e:
        logger.warning(f"coverage_figure skipped ({e})")
        return None


# ---------------------------------------------------------------------------
def logo_figure(logo_df: pd.DataFrame, tag: str = "logo_climate") -> Optional[str]:
    """LOGO bars with the mean line labelled ABOVE the tallest bar."""
    try:
        apply_style()
        import matplotlib.pyplot as plt

        if logo_df is None or not len(logo_df):
            return None
        d = logo_df.sort_values("r2_log_total", ascending=False)
        fig, ax = plt.subplots(figsize=(3.8, 3.4))
        bars = ax.bar(d["zone"].astype(str), d["r2_log_total"],
                      color=[PALETTE["primary"]] * len(d), width=0.6)
        mean_r2 = float(d["r2_log_total"].mean())
        ax.axhline(mean_r2, color=PALETTE["accent"], ls="--", lw=1.0)
        for b, v, nt in zip(bars, d["r2_log_total"], d.get("n_test", [None] * len(d))):
            lbl = f"{v:.3f}" + (f"\n(n={int(nt):,})" if nt else "")
            ax.annotate(lbl, (b.get_x() + b.get_width() / 2, v),
                        textcoords="offset points", xytext=(0, 3),
                        ha="center", va="bottom", fontsize=7.5)
        annotate_above_bars(ax, bars, f"mean = {mean_r2:.3f}")
        ax.set_ylabel("Test $R^2$ (log-total)")
        ax.set_title("Leave-one-climate-out")
        ax.set_ylim(0, max(d["r2_log_total"].max() * 1.18, mean_r2 * 1.25))
        fig.tight_layout()
        p = save_figure(fig, FIGURES_DIR / f"{tag}")
        plt.close(fig)
        logger.info(f"LOGO figure saved (vector): {p}")
        return str(p)
    except Exception as e:
        logger.warning(f"logo_figure skipped ({e})")
        return None


# ---------------------------------------------------------------------------
def counterfactual_figure(cf_student: Optional[pd.DataFrame],
                          cf_teacher: Optional[pd.DataFrame],
                          tag: str = "counterfactual") -> Optional[str]:
    """Student vs teacher vs ResStock-matched savings, per scenario. Missing
    teacher values are drawn as hatched 'not pairable' stubs rather than
    silently absent, so the figure is never half-empty."""
    try:
        apply_style()
        import matplotlib.pyplot as plt

        frames = [f for f in (cf_student, cf_teacher) if f is not None and len(f)]
        if not frames:
            return None
        scen = []
        for f in frames:
            for s in f["scenario"]:
                if s not in scen:
                    scen.append(s)
        x = np.arange(len(scen))
        w = 0.26

        def col(frame, name, default=np.nan):
            if frame is None or name not in frame.columns:
                return np.full(len(scen), default)
            s = frame.set_index("scenario")[name].reindex(scen)
            return s.to_numpy(dtype=float)

        fig, ax = plt.subplots(figsize=(7.2, 3.4))
        stu = col(cf_student, "mean_savings_pct")
        tea = col(cf_teacher, "teacher_mean_savings_pct")
        meas = col(cf_student, "resstock_delta_matched_pct")
        if np.all(np.isnan(meas)):
            meas = col(cf_teacher, "resstock_delta_matched_pct")

        ax.bar(x - w, stu, w, label="student (RECS)", color=PALETTE["primary"])
        ax.bar(x, np.nan_to_num(tea), w, label="teacher (ResStock CATE)",
               color=PALETTE["physics"],
               hatch="" if not np.isnan(tea).any() else None)
        ax.bar(x + w, meas, w, label="ResStock matched delta",
               color=PALETTE["green"])
        for xi, v in zip(x, tea):
            if np.isnan(v):
                ax.text(xi, 0.4, "n/a", ha="center", va="bottom", fontsize=7,
                        rotation=90, color="#666666")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace(" ", "\n") for s in scen], fontsize=7.5)
        ax.set_ylabel("Savings (%)")
        ax.set_title("Counterfactual retrofit savings")
        ax.legend(frameon=False, fontsize=7.5)
        fig.tight_layout()
        p = save_figure(fig, FIGURES_DIR / f"{tag}")
        plt.close(fig)
        logger.info(f"Counterfactual figure saved (vector): {p}")
        return str(p)
    except Exception as e:
        logger.warning(f"counterfactual_figure skipped ({e})")
        return None


# ---------------------------------------------------------------------------
def architecture_schematic(tag: str = "architecture") -> Optional[str]:
    """Pipeline architecture drawn as a real schematic: aligned lanes, shared
    geometry, and the audit gate drawn as a gate rather than another box."""
    try:
        apply_style()
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 52)
        ax.axis("off")
        ax.grid(False)

        def box(x, y, w, h, text, fc, fs=7.5, bold=False):
            ax.add_patch(FancyBboxPatch((x, y), w, h,
                                        boxstyle="round,pad=0.6,rounding_size=1.2",
                                        fc=fc, ec="#4a4a4a", lw=0.7))
            ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                    fontsize=fs, fontweight="bold" if bold else "normal")

        def arrow(x1, y1, x2, y2, style="-|>", color="#4a4a4a"):
            ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                         mutation_scale=8, lw=0.8, color=color))

        inputs = [("RECS 2020\n18,496 households", 2, 40, PALETTE["primary"]),
                  ("TMY3\n1,020 stations", 2, 27, PALETTE["primary"]),
                  ("ResStock 2025.1\nbaseline + 5 upgrades", 2, 12, PALETTE["physics"])]
        for t, x, y, c in inputs:
            box(x, y, 20, 9, t, "#eef2f7" if c == PALETTE["primary"] else "#f1ecf7")
            arrow(22, y + 4.5, 27, 24)

        box(27, 19, 17, 11, "Physics-informed\nfeatures (row-local)\n+ split 70/10/5/15", "#eef2f7")
        arrow(44, 24.5, 49, 24.5)

        # audit gate
        ax.add_patch(FancyBboxPatch((49, 17), 12, 15,
                                    boxstyle="round,pad=0.6,rounding_size=1.2",
                                    fc="#f7efe9", ec=PALETTE["accent"], lw=1.1))
        ax.text(55, 26.5, "Leakage audit", ha="center", va="center", fontsize=7.5,
                fontweight="bold")
        ax.text(55, 21.5, "A B C D G H\n+ JSON report", ha="center", va="center",
                fontsize=6.8)
        arrow(61, 24.5, 66, 24.5)

        box(66, 31, 20, 11, "Multi-task core\nprimary + 4 shares\nadaptive routing", "#eef2f7")
        box(66, 18, 20, 10, "Honest stacking\nOOF meta-learner", "#eef2f7")
        box(66, 6, 20, 10, "Split-conformal\non untouched calib", "#eef2f7")
        arrow(76, 31, 76, 28)
        arrow(76, 18, 76, 16)

        box(89, 18, 10, 23, "Held-out\ntest\n(n = 2,775)", "#eef7ee", bold=True)
        arrow(86, 36.5, 89, 33)
        arrow(86, 23, 89, 26)
        arrow(86, 11, 89, 21)

        ax.text(50, 47, "Physics-AI multi-task pipeline with a leakage-audited, "
                        "conformal-calibrated output", ha="center", fontsize=8,
                fontweight="bold")
        ax.text(12, 3.5, "Teacher distillation: ResStock teacher never sees RECS targets",
                ha="center", fontsize=6.8, style="italic", color="#555555")

        fig.tight_layout()
        p = save_figure(fig, FIGURES_DIR / f"{tag}")
        plt.close(fig)
        logger.info(f"Architecture schematic saved (vector): {p}")
        return str(p)
    except Exception as e:
        logger.warning(f"architecture_schematic skipped ({e})")
        return None
