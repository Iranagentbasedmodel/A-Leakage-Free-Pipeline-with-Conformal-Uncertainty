"""
SHAP Analysis — v8.1
====================
Reviewer finding #4: Fig. 3 and the top-feature list quoted in Sec. 4.5 came
from different runs and disagreed (0.160 vs 0.161; different top-ten
memberships). Root cause: importance values were only ever written into a
figure and a loose CSV, then re-typed into the manuscript from memory of a
previous run.

v8.1: one function computes the importances ONCE from the SAME fitted model
and the SAME feature matrix used for the run's metrics, and writes
  * ``feature_importance_<tag>.csv``         full ranked table
  * ``feature_importance_<tag>.json``        top-K values, machine-readable
                                             (the manuscript quotes THIS file)
  * ``feature_importance_<tag>.{pdf,svg,png}`` the figure, from the same values
The JSON is also registered in the run manifest by main.py, so figure, table
and text cannot drift apart again.
"""

import json
import logging
from typing import Optional

import numpy as np
import pandas as pd

from config import FIGURES_DIR, TABLES_DIR
from figure_style import PALETTE, apply_style, save_figure

logger = logging.getLogger("shap_analysis")


def run_shap(model, X: pd.DataFrame, tag: str = "total", max_rows: int = 3000,
             top_k: int = 10) -> Optional[dict]:
    """Compute mean |SHAP| (or gain fallback) once; return the top-K dict."""
    booster = getattr(model, "total_model", model)
    try:
        Xs = X.sample(min(max_rows, len(X)), random_state=42)
        try:
            import shap
            explainer = shap.TreeExplainer(booster)
            sv = explainer.shap_values(Xs)
            imp = pd.Series(np.abs(sv).mean(axis=0),
                            index=Xs.columns).sort_values(ascending=False)
            method = "shap"
        except Exception:
            imp = pd.Series(booster.feature_importances_,
                            index=Xs.columns).sort_values(ascending=False)
            method = "gain"
        top = imp.head(25)
        top.to_csv(TABLES_DIR / f"feature_importance_{tag}.csv")

        topk = {str(k): float(v) for k, v in imp.head(top_k).items()}
        payload = {"method": method, "n_rows_shap": int(len(Xs)),
                   "top_k": top_k, "values": topk,
                   "top_features": list(topk.keys())}
        with open(TABLES_DIR / f"feature_importance_{tag}.json", "w",
                  encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"Top-{top_k} features ({method}, single run): {list(topk.keys())}")

        apply_style()
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4.6, 4.2))
        show = imp.head(12).sort_values()
        colors = [PALETTE["physics"] if (c.startswith(("PH_", "TL_")))
                  else PALETTE["green"] for c in show.index]
        ax.barh(show.index.astype(str), show.values, color=colors, height=0.68)
        for i, v in enumerate(show.values):
            ax.text(v, i, f" {v:.3f}", va="center", fontsize=6.8)
        ax.set_xlabel("mean |SHAP| (log-total)")
        ax.set_title(f"Feature importance ({method})")
        ax.margins(x=0.14)
        fig.tight_layout()
        p = save_figure(fig, FIGURES_DIR / f"feature_importance_{tag}")
        plt.close(fig)
        logger.info(f"SHAP figure saved (vector): {p}")
        return payload
    except Exception as e:
        logger.warning(f"run_shap skipped ({e})")
        return None
