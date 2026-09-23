"""
Figure Style — v8.1
===================
One place for publication-figure policy (reviewer finding: inconsistent font
sizes, raster-only output, overlapping annotations).

  * a single rcParams profile applied to every figure;
  * every figure saved as PDF + SVG (vector, journal-ready) and PNG
    (preview), via ``save_figure``;
  * annotation helpers that place labels so they cannot collide with bars
    (the Fig.-5 mean-label overlap is fixed at the helper level).
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from config import FIG_DPI, FIG_FONT_SIZE, FIG_FORMATS

logger = logging.getLogger("figure_style")

PALETTE = {
    "primary": "#3b5b7a",
    "accent": "#a55f4a",
    "physics": "#6a4c93",
    "green": "#5b8c5a",
    "grid": "#d9d9d9",
}


def apply_style() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": FIG_FONT_SIZE,
        "axes.titlesize": FIG_FONT_SIZE + 1,
        "axes.labelsize": FIG_FONT_SIZE,
        "xtick.labelsize": FIG_FONT_SIZE - 1,
        "ytick.labelsize": FIG_FONT_SIZE - 1,
        "legend.fontsize": FIG_FONT_SIZE - 1,
        "axes.grid": True,
        "grid.color": PALETTE["grid"],
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.dpi": FIG_DPI,
        "savefig.bbox": "tight",
    })


def save_figure(fig, stem) -> Optional[Path]:
    """Save one figure to every configured format (vector first)."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    first = None
    for fmt in FIG_FORMATS:
        p = stem.with_suffix(f".{fmt}")
        try:
            fig.savefig(p, dpi=FIG_DPI if fmt == "png" else None)
            first = first or p
        except Exception as e:  # pragma: no cover
            logger.warning(f"save_figure {p.name} failed ({e})")
    return first


def annotate_above_bars(ax, bars, text: str, pad_frac: float = 0.03):
    """Place a text annotation safely above the tallest bar (fixes the
    mean-label/bars overlap flagged in Fig. 5)."""
    tops = [b.get_height() for b in bars if np.isfinite(b.get_height())]
    if not tops:
        return
    top = max(tops)
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0], max(ylim[1], top * (1 + 4 * pad_frac)))
    ax.annotate(text, xy=(0.985, top * (1 + pad_frac)), xycoords=("axes fraction", "data"),
                ha="right", va="bottom", fontsize=FIG_FONT_SIZE - 1,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
