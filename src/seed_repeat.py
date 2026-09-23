"""
Multi-Seed Repetition — v8.1 (reviewer finding #3)
==================================================
Every headline comparison in the paper is a difference of two numbers fitted
once. With n(test) = 2,775 the seed-to-seed fluctuation of a single XGBoost
fit is of the same order as the claimed effects, so single-run differences
are not interpretable. This module re-runs the compared models under an
IDENTICAL fixed-parameter protocol for a list of seeds and reports:

  * per-model mean / sd / 95 % CI over seeds;
  * PAIRED per-seed differences with 95 % CIs and sign consistency for
      - the transfer-learning effect (with TL vs without TL, per base learner)
      - the architecture effect (honest stacking vs the best single baseline)
  * an explicit verdict per comparison (``consistent_gain`` /
    ``no_measurable_gain`` / ``within_noise``) computed from the CI against a
    pre-registered threshold, so the manuscript can only claim what the
    numbers support.

The protocol is deliberately plain (no Optuna, no early-stopping differences
between arms): both arms of every comparison see the same features minus/plus
the TL_* meta-columns, the same hyper-parameters and the same val split for
early stopping.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from config import (EFFECT_THRESHOLD_PP, ENSEMBLE_SEEDS_DEFAULT,
                    SEED_LIST, N_JOBS)
from evaluation import summarise, paired_difference

logger = logging.getLogger("seed_repeat")

try:
    from xgboost import XGBRegressor
    _XGB_OK = True
except ImportError:  # pragma: no cover
    _XGB_OK = False

try:
    from lightgbm import LGBMRegressor
    import lightgbm as lgb
    _LGB_OK = True
except ImportError:  # pragma: no cover
    _LGB_OK = False


def _make_xgb(seed: int, n_jobs: int):
    return XGBRegressor(n_estimators=2000, learning_rate=0.05, max_depth=6,
                        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                        random_state=seed, n_jobs=n_jobs, tree_method="hist")


def _make_lgb(seed: int, n_jobs: int):
    return LGBMRegressor(n_estimators=2000, learning_rate=0.05, max_depth=6,
                         subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                         random_state=seed, n_jobs=n_jobs, verbose=-1)


def _fit_score(make_fn, seed, X_tr, y_log_tr, X_va, y_log_va, X_te, y_total_te,
               w_tr) -> float:
    """One arm of one seed: fit with val early stopping, score test R2 (kBtu)."""
    m = make_fn(seed, N_JOBS)
    fit_kw = {"eval_set": [(X_va, y_log_va)]}
    if w_tr is not None:
        fit_kw["sample_weight"] = w_tr.loc[X_tr.index]
    if m.__class__.__name__.startswith("XGB"):
        m.set_params(early_stopping_rounds=80)
        fit_kw["verbose"] = False
    else:
        fit_kw["callbacks"] = [lgb.early_stopping(80, verbose=False)]
    m.fit(X_tr, y_log_tr, **fit_kw)
    return float(r2_score(y_total_te, np.expm1(m.predict(X_te))))


def run_seed_repetition(X_train: pd.DataFrame, y_log_train: pd.Series,
                        X_val: pd.DataFrame, y_log_val: pd.Series,
                        X_test: pd.DataFrame, y_test_total: pd.Series,
                        w_train: Optional[pd.Series] = None,
                        seeds: Optional[List[int]] = None,
                        ensemble_seeds: int = ENSEMBLE_SEEDS_DEFAULT,
                        run_ensemble: bool = True,
                        threshold_pp: float = EFFECT_THRESHOLD_PP) -> Dict:
    """Returns {"raw": DataFrame, "summary": {...}, "comparisons": {...}}."""
    seeds = list(seeds or SEED_LIST)
    tl_cols = [c for c in X_train.columns if c.startswith("TL_")]
    cols_no_tl = [c for c in X_train.columns if c not in tl_cols]
    has_tl = len(tl_cols) > 0
    logger.info("=" * 60)
    logger.info(f"MULTI-SEED REPETITION v8.1 — {len(seeds)} seeds "
                f"{list(seeds)}{' (TL arms included)' if has_tl else ''}")
    logger.info("=" * 60)

    rows = []
    for s in seeds:
        row = {"seed": s}
        if _XGB_OK:
            row["xgb_noTL"] = _fit_score(_make_xgb, s, X_train[cols_no_tl], y_log_train,
                                         X_val[cols_no_tl], y_log_val,
                                         X_test[cols_no_tl], y_test_total, w_train)
            if has_tl:
                row["xgb_TL"] = _fit_score(_make_xgb, s, X_train, y_log_train,
                                           X_val, y_log_val, X_test, y_test_total, w_train)
        if _LGB_OK:
            row["lgb_noTL"] = _fit_score(_make_lgb, s, X_train[cols_no_tl], y_log_train,
                                         X_val[cols_no_tl], y_log_val,
                                         X_test[cols_no_tl], y_test_total, w_train)
            if has_tl:
                row["lgb_TL"] = _fit_score(_make_lgb, s, X_train, y_log_train,
                                           X_val, y_log_val, X_test, y_test_total, w_train)
        rows.append(row)
        logger.info("  seed %s: " % s + "  ".join(
            f"{k}={v:.4f}" for k, v in row.items() if k != "seed"))
    raw = pd.DataFrame(rows).set_index("seed")

    # ---- optional: honest stacking on a few seeds --------------------------
    if run_ensemble:
        try:
            from ensemble_models import HonestStacking
            ens_rows = []
            for s in seeds[:max(1, min(ensemble_seeds, len(seeds)))]:
                ens = HonestStacking(seed=s, n_jobs=N_JOBS, n_folds=5)
                ens.fit(X_train, y_log_train, X_val, y_log_val, w_train=w_train)
                ens_rows.append({"seed": s,
                                 "ensemble": float(r2_score(
                                     y_test_total, np.expm1(ens.predict(X_test))))})
                logger.info(f"  seed {s}: ensemble={ens_rows[-1]['ensemble']:.4f}")
            raw = raw.join(pd.DataFrame(ens_rows).set_index("seed"), how="left")
        except Exception as e:  # pragma: no cover
            logger.warning(f"  ensemble seed repetition skipped ({e})")

    # ---- summaries ----------------------------------------------------------
    summary = {c: summarise(raw[c].dropna().values, name=c) for c in raw.columns}

    comparisons: Dict[str, Dict] = {}
    if has_tl and _XGB_OK:
        comparisons["tl_effect_xgb"] = paired_difference(
            raw["xgb_TL"], raw["xgb_noTL"], "XGBoost with TL - without TL", threshold_pp)
    if has_tl and _LGB_OK:
        comparisons["tl_effect_lgb"] = paired_difference(
            raw["lgb_TL"], raw["lgb_noTL"], "LightGBM with TL - without TL", threshold_pp)
    if "ensemble" in raw.columns:
        base_cols = [c for c in ("xgb_TL", "xgb_noTL", "lgb_TL", "lgb_noTL")
                     if c in raw.columns]
        best_base = raw[base_cols].max(axis=1)
        both = raw["ensemble"].notna()
        comparisons["stacking_effect"] = paired_difference(
            raw.loc[both, "ensemble"], best_base.loc[both],
            "Honest stacking - best single baseline", threshold_pp)

    logger.info("-" * 60)
    for k, v in comparisons.items():
        logger.info(f"  {k:<18}: {v['mean_pp']:+.2f} pp "
                    f"[{v['ci95_lo_pp']:+.2f}, {v['ci95_hi_pp']:+.2f}] "
                    f"sign {v['sign_consistency']} -> {v['verdict']}")
    return {"raw": raw, "summary": summary, "comparisons": comparisons}
