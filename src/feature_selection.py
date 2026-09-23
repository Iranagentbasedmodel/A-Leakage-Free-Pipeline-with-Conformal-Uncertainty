"""
Feature Selection — v8.0
========================
Train-only, NaN-safe hybrid selection (mutual information + model importance).

v7.2 bug: all-NaN columns (from the broken TMY3 merge) survived
``fillna(median)`` so ``mutual_info_regression`` crashed with
"Input X contains NaN" and selection silently degraded to 87 features.
v8.0: all-NaN/constant columns are dropped first; remaining NaN are imputed
(median, then 0); selection runs on the TRAIN split only.
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

from config import MAX_FEATURES, RANDOM_SEED, N_JOBS

logger = logging.getLogger("feature_selection")

PH_PREFIX = "PH_"


def select_features(X_train: pd.DataFrame,
                    y_train: pd.Series,
                    max_features: int = MAX_FEATURES,
                    keep_physics: bool = True) -> List[str]:
    """Hybrid MI + XGB-importance selection, fitted on train only."""
    X = X_train.copy()
    y = pd.to_numeric(y_train, errors="coerce")

    # ---- 1) drop all-NaN and constant columns first: mutual information is
    # undefined on zero-variance inputs and would abort the whole selection step
    not_all_nan = X.columns[X.notna().any()]
    nunique = X[not_all_nan].nunique(dropna=True)
    usable = list(nunique[nunique > 1].index)
    dropped = X.shape[1] - len(usable)
    if dropped:
        logger.info(f"Dropped {dropped} all-NaN/constant columns before selection")
    X = X[usable]

    # ---- 2) NaN-safe imputation
    X = X.fillna(X.median()).fillna(0.0)
    ok = y.notna() & np.isfinite(y)
    X, y = X.loc[ok], y.loc[ok]
    if len(X) < 100:
        logger.warning("Too few valid rows for selection; keeping all usable columns.")
        return usable[:max_features]

    # ---- 3) mutual information (subsample for speed on CPU)
    n_mi = min(len(X), 10_000)
    idx = X.sample(n_mi, random_state=RANDOM_SEED).index if len(X) > n_mi else X.index
    mi = mutual_info_regression(X.loc[idx], y.loc[idx], random_state=RANDOM_SEED,
                                n_neighbors=3)
    mi_s = pd.Series(mi, index=X.columns)

    # ---- 4) quick model importance (XGBoost or fallback to correlation)
    try:
        from xgboost import XGBRegressor
        m = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.06,
                         subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                         n_jobs=N_JOBS, random_state=RANDOM_SEED)
        m.fit(X, y)
        imp_s = pd.Series(m.feature_importances_, index=X.columns)
    except Exception as e:  # pragma: no cover
        logger.warning(f"XGB importance failed ({e}); falling back to |correlation|")
        imp_s = X.apply(lambda c: abs(np.corrcoef(c, y)[0, 1]) if c.std() > 0 else 0)

    # ---- 5) hybrid score: average of normalized ranks
    score = (mi_s.rank(pct=True) + imp_s.rank(pct=True)) / 2.0
    score = score.sort_values(ascending=False)

    physics = [c for c in score.index if c.startswith(PH_PREFIX)]
    rest = [c for c in score.index if not c.startswith(PH_PREFIX)]
    if keep_physics:
        n_ph = min(len(physics), max_features)
        selected = physics[:n_ph]
        remaining = max_features - len(selected)
        selected += rest[:remaining]
    else:
        selected = list(score.index[:max_features])

    logger.info(f"Selected {len(selected)} features (physics: "
                f"{sum(c.startswith(PH_PREFIX) for c in selected)}, "
                f"from {X.shape[1]} available, train-only)")
    return selected
