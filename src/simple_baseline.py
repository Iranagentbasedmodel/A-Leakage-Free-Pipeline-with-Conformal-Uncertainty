"""
Simple single-model baselines (v8.0)
====================================
Fair comparison baselines for the paper (Table 3).
Trains plain XGBoost / LightGBM on the exact same feature matrix
and split used by the main energy-conserving pipeline.
Does NOT modify any state of EnergyConservingModel or HonestStacking.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

logger = logging.getLogger("simple_baseline")

try:
    from xgboost import XGBRegressor
    _XGB_OK = True
except ImportError:
    _XGB_OK = False

try:
    from lightgbm import LGBMRegressor
    import lightgbm as lgb
    _LGB_OK = True
except ImportError:
    _LGB_OK = False


def _train_one(
    name: str,
    model,
    X_tr: pd.DataFrame,
    y_log_tr: pd.Series,
    X_va: pd.DataFrame,
    y_log_va: pd.Series,
    X_te: pd.DataFrame,
    y_total_te: pd.Series,
    w_tr: Optional[pd.Series] = None,
) -> Dict:
    """Fit one model and return test metrics on original (kBtu) scale."""
    fit_kwargs = {"eval_set": [(X_va, y_log_va)]}
    if w_tr is not None:
        fit_kwargs["sample_weight"] = w_tr.loc[X_tr.index]

    is_xgb = model.__class__.__name__.startswith("XGB")
    if is_xgb:
        model.set_params(early_stopping_rounds=80)
        fit_kwargs["verbose"] = False
    else:
        fit_kwargs["callbacks"] = [lgb.early_stopping(80, verbose=False)]

    model.fit(X_tr, y_log_tr, **fit_kwargs)

    pred_log = model.predict(X_te)
    pred_total = np.expm1(pred_log)
    r2 = float(r2_score(y_total_te, pred_total))
    rmse = float(np.sqrt(mean_squared_error(y_total_te, pred_total)))

    best_iter = getattr(model, "best_iteration", None)
    if best_iter is None:
        best_iter = getattr(model, "best_iteration_", None)

    logger.info(f"  {name:<22} → R²={r2:.4f} | RMSE={rmse:,.0f} kBtu")
    return {
        "R2_test": r2,
        "RMSE_test": rmse,
        "best_iteration": int(best_iter) if best_iter is not None else None,
    }


def run_simple_baselines(
    X_train: pd.DataFrame,
    y_log_train: pd.Series,
    X_val: pd.DataFrame,
    y_log_val: pd.Series,
    X_test: pd.DataFrame,
    y_test_total: pd.Series,
    w_train: Optional[pd.Series] = None,
    output_dir: str | Path = "outputs/baselines",
    random_seed: int = 42,
    n_jobs: int = -1,
) -> Dict:
    """
    Run plain single-target baselines on the provided matrices.
    Also reports a version that drops any TL_* meta-features.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: Dict = {}

    # ---------- helpers to build models ----------
    def make_xgb():
        return XGBRegressor(
            n_estimators=2000,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=random_seed,
            n_jobs=n_jobs,
            tree_method="hist",
        )

    def make_lgb():
        return LGBMRegressor(
            n_estimators=2000,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=random_seed,
            n_jobs=n_jobs,
            verbose=-1,
        )

    # ---------- 1) Full feature set (may contain TL_*) ----------
    logger.info("  Simple baselines on FULL feature set (incl. TL if present)...")
    if _XGB_OK:
        try:
            results["XGBoost_simple"] = _train_one(
                "XGBoost_simple",
                make_xgb(),
                X_train, y_log_train, X_val, y_log_val, X_test, y_test_total, w_train,
            )
        except Exception as e:
            logger.warning(f"  XGBoost_simple failed: {e}")

    if _LGB_OK:
        try:
            results["LightGBM_simple"] = _train_one(
                "LightGBM_simple",
                make_lgb(),
                X_train, y_log_train, X_val, y_log_val, X_test, y_test_total, w_train,
            )
        except Exception as e:
            logger.warning(f"  LightGBM_simple failed: {e}")

    # ---------- 2) Without any TL_* meta-features ----------
    tl_cols = [c for c in X_train.columns if c.startswith("TL_")]
    if tl_cols:
        logger.info(f"  Simple baselines WITHOUT TL meta-features (dropped {len(tl_cols)} cols)...")
        cols_no_tl = [c for c in X_train.columns if not c.startswith("TL_")]
        Xtr = X_train[cols_no_tl]
        Xva = X_val[cols_no_tl]
        Xte = X_test[cols_no_tl]

        if _XGB_OK:
            try:
                results["XGBoost_simple_noTL"] = _train_one(
                    "XGBoost_simple_noTL",
                    make_xgb(),
                    Xtr, y_log_train, Xva, y_log_val, Xte, y_test_total, w_train,
                )
            except Exception as e:
                logger.warning(f"  XGBoost_simple_noTL failed: {e}")

        if _LGB_OK:
            try:
                results["LightGBM_simple_noTL"] = _train_one(
                    "LightGBM_simple_noTL",
                    make_lgb(),
                    Xtr, y_log_train, Xva, y_log_val, Xte, y_test_total, w_train,
                )
            except Exception as e:
                logger.warning(f"  LightGBM_simple_noTL failed: {e}")

    # save
    out_path = output_dir / "simple_baselines.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"  Simple baselines saved → {out_path}")
    return results