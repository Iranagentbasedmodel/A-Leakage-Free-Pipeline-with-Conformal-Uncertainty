"""
Evaluation — v8.0
=================
Comprehensive metrics including survey-weighted variants (NWEIGHT),
prediction-interval coverage, and a population-level calibration check
(weighted sum of predictions vs weighted sum of actuals — the external
validation against published EIA aggregates recommended in the RECS
microdata guide).
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger("evaluation")


def comprehensive_metrics(y_true, y_pred, weights: Optional[pd.Series] = None,
                          name: str = "") -> Dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[ok], y_pred[ok]
    if len(y_true) < 10:
        return {"name": name, "n": int(len(y_true))}
    m: Dict = {
        "name": name,
        "n": int(len(y_true)),
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "nmbiased_pct": float(100 * (y_pred.mean() - y_true.mean()) /
                              max(abs(y_true.mean()), 1e-9)),
    }
    m["cvrmse_pct"] = float(100 * m["rmse"] / max(abs(y_true.mean()), 1e-9))
    if weights is not None:
        w = np.asarray(weights, dtype=float)[ok]
        w = w / w.mean()
        var_w = np.average((y_true - np.average(y_true, weights=w)) ** 2, weights=w)
        res_w = np.average((y_true - y_pred) ** 2, weights=w)
        m["r2_weighted"] = float(1 - res_w / var_w) if var_w > 0 else np.nan
    return m


def interval_coverage(y_true, lo, hi) -> Dict:
    y_true = np.asarray(y_true, dtype=float)
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    ok = np.isfinite(y_true) & np.isfinite(lo) & np.isfinite(hi)
    if ok.sum() < 10:
        return {}
    cov = float(np.mean((y_true[ok] >= lo[ok]) & (y_true[ok] <= hi[ok])))
    width = float(np.mean(hi[ok] - lo[ok]))
    return {"coverage": cov, "mean_width": width, "n": int(ok.sum())}


def population_calibration(y_true, y_pred, weights) -> Dict:
    """Weighted national-total comparison (population calibration error)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    w = np.asarray(weights, dtype=float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(w)
    if ok.sum() < 10:
        return {}
    tot_true = float((y_true[ok] * w[ok]).sum())
    tot_pred = float((y_pred[ok] * w[ok]).sum())
    return {"weighted_total_true": tot_true, "weighted_total_pred": tot_pred,
            "calibration_error_pct": 100 * (tot_pred - tot_true) / max(tot_true, 1e-9)}


def log_metrics_block(logger_, title: str, metrics: Dict):
    logger_.info("-" * 56)
    logger_.info(title)
    for k, v in metrics.items():
        if isinstance(v, float):
            logger_.info(f"  {k:<22}: {v:,.4f}")
        else:
            logger_.info(f"  {k:<22}: {v}")


# ---------------------------------------------------------------------------
# v8.1 — multi-seed uncertainty helpers (reviewer finding #3)
# ---------------------------------------------------------------------------
def summarise(values, name: str = "") -> Dict:
    """Mean / sd / 95 % CI (t) of a vector of per-seed scores."""
    from scipy import stats
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    n = len(v)
    if n == 0:
        return {"name": name, "n": 0}
    mean = float(v.mean())
    sd = float(v.std(ddof=1)) if n > 1 else 0.0
    half = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)) if n > 1 else 0.0
    return {"name": name, "n": n, "mean": mean, "sd": sd,
            "ci95_lo": mean - half, "ci95_hi": mean + half,
            "min": float(v.min()), "max": float(v.max())}


def paired_difference(a, b, name: str = "", threshold_pp: float = 0.10) -> Dict:
    """Paired per-seed difference (a - b) in percentage points with a 95 % CI
    and a consistency verdict.

    Verdict logic (stated up-front so the manuscript wording cannot drift):
      * ``consistent_gain``     CI lower bound > +threshold
      * ``no_measurable_gain``  CI upper bound < +threshold
      * ``within_noise``        otherwise
    """
    from scipy import stats
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = int(min(len(a), len(b)))
    d = (a[:n] - b[:n]) * 100.0                    # R2 -> percentage points
    mean = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    half = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)) if n > 1 else 0.0
    lo, hi = mean - half, mean + half
    if lo > threshold_pp:
        verdict = "consistent_gain"
    elif hi < threshold_pp:
        verdict = "no_measurable_gain"
    else:
        verdict = "within_noise"
    return {"name": name, "n": n, "mean_pp": mean, "sd_pp": sd,
            "ci95_lo_pp": lo, "ci95_hi_pp": hi,
            "sign_consistency": f"{int((d > 0).sum())}/{n}",
            "verdict": verdict, "threshold_pp": threshold_pp}


def conformal_coverage_curve(y_true, pred_log, residuals_calib,
                             alphas=(0.20, 0.10, 0.05)) -> Dict[float, float]:
    """Empirical coverage of split-conformal bands at several nominal levels.

    The band half-width for level alpha is the empirical
    ceil((n+1)(1-alpha))/n quantile of the CALIBRATION residuals (same
    construction as the 90 % band, recomputed per level). Coverage is then
    measured on an evaluation sample. Used for the reliability figure.
    """
    res = np.sort(np.abs(np.asarray(residuals_calib, dtype=float)))
    n = len(res)
    y = np.asarray(y_true, dtype=float)
    mu = np.asarray(pred_log, dtype=float)
    ok = np.isfinite(y) & np.isfinite(mu)
    y, mu = y[ok], mu[ok]
    out = {}
    for a in alphas:
        k = int(np.ceil((n + 1) * (1 - a)))
        q = res[min(k, n) - 1]
        out[round(1 - a, 3)] = float(np.mean((y >= mu - q) & (y <= mu + q)))
    return out
