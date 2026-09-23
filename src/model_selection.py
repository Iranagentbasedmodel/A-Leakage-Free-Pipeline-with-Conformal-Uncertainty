"""
Model Selection — v8.0 (compact)
================================
Lightweight comparison of candidate regressors on a subsample, kept for
project compatibility. The production architecture (EnergyConservingModel +
HonestStacking) is trained in model_training.py / ensemble_models.py.
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from config import LGB_BASE_PARAMS, XGB_BASE_PARAMS, RANDOM_SEED, N_JOBS

logger = logging.getLogger("model_selection")


def quick_benchmark(X: pd.DataFrame, y: pd.Series, max_rows: int = 5000) -> pd.DataFrame:
    rows = []
    try:
        Xs = X.sample(min(max_rows, len(X)), random_state=RANDOM_SEED)
        ys = y.loc[Xs.index]
        X_tr, X_te, y_tr, y_te = train_test_split(Xs, ys, test_size=0.25,
                                                  random_state=RANDOM_SEED)
        candidates = {}
        try:
            from xgboost import XGBRegressor
            p = {k: v for k, v in XGB_BASE_PARAMS.items() if k != "early_stopping_rounds"}
            p["n_estimators"] = 300
            candidates["xgb"] = XGBRegressor(**p, n_jobs=N_JOBS)
        except ImportError:
            pass
        try:
            from lightgbm import LGBMRegressor
            p = dict(LGB_BASE_PARAMS)
            p["n_estimators"] = 300
            candidates["lgb"] = LGBMRegressor(**p, n_jobs=N_JOBS)
        except ImportError:
            pass
        from sklearn.ensemble import RandomForestRegressor
        candidates["rf"] = RandomForestRegressor(n_estimators=120, n_jobs=N_JOBS,
                                                 random_state=RANDOM_SEED)
        for name, m in candidates.items():
            m.fit(X_tr, y_tr)
            rows.append({"model": name,
                         "r2": float(r2_score(y_te, m.predict(X_te)))})
        df = pd.DataFrame(rows).sort_values("r2", ascending=False)
        logger.info(f"Quick benchmark:\n{df.to_string(index=False)}")
        return df
    except Exception as e:
        logger.warning(f"quick_benchmark skipped ({e})")
        return pd.DataFrame(rows)
