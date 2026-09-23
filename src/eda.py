"""
Exploratory Data Analysis — v8.1
================================
Target-distribution summaries; vector figure output via figure_style.
Failures here never block the pipeline.
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd

from config import FIGURES_DIR, TABLES_DIR, TARGET_EUI_COL, TARGET_TOTAL_COL
from figure_style import PALETTE, apply_style, save_figure

logger = logging.getLogger("eda")


def run_eda(df: pd.DataFrame, tag: str = "recs") -> Dict:
    out: Dict = {}
    try:
        apply_style()
        import matplotlib.pyplot as plt

        eui = pd.to_numeric(df.get(TARGET_EUI_COL), errors="coerce").dropna()
        total = pd.to_numeric(df.get(TARGET_TOTAL_COL), errors="coerce").dropna()
        summary = {
            "eui_mean": float(eui.mean()), "eui_median": float(eui.median()),
            "total_mean_kbtu": float(total.mean()), "n": int(len(df)),
        }
        out["summary"] = summary

        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
        axes[0].hist(eui.clip(0, eui.quantile(0.99)), bins=60,
                     color=PALETTE["green"], edgecolor="white", linewidth=0.2)
        axes[0].set_title("Site EUI distribution")
        axes[0].set_xlabel("EUI (kBtu/ft$^2$)")
        axes[0].set_ylabel("Households")
        axes[1].hist(np.log1p(total), bins=60, color="#8c6d5b",
                     edgecolor="white", linewidth=0.2)
        axes[1].set_title("log1p(total site energy)")
        axes[1].set_xlabel("log1p(kBtu)")
        fig.tight_layout()
        p = save_figure(fig, FIGURES_DIR / f"eda_targets_{tag}")
        plt.close(fig)
        out["figure"] = str(p)
        pd.DataFrame([summary]).to_csv(TABLES_DIR / f"eda_summary_{tag}.csv", index=False)
        logger.info(f"EDA done: EUI mean={summary['eui_mean']:.1f}, n={summary['n']:,}")
    except Exception as e:
        logger.warning(f"EDA skipped ({e})")
    return out
