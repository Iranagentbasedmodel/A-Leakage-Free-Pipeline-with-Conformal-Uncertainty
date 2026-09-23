"""
Honest Stacking Ensemble — v8.0
===============================
v7.2 had THREE critical stacking bugs (val R2=0.79 vs test R2=0.58):
  a) at predict time only the LAST CV fold's base models were used, while the
     meta-learner was trained on OOF predictions of ALL folds -> distribution
     mismatch;
  b) meta-feature column order differed between train (sorted names) and
     predict (insertion order) -> coefficients applied to the wrong models;
  c) a residual model was fitted on the VALIDATION residuals and validation
     metrics were then computed on that same split -> inflated val score and
     degraded test performance.

v8.0 protocol (textbook stacking):
  1) KFold out-of-fold predictions train a RidgeCV meta-learner
     (coefficients clipped at 0 afterwards for a convex-ish combination;
     ``positive=True`` is not supported by RidgeCV in sklearn >= 1.7).
  2) Base learners are RETRAINED on 100% of train.
  3) Column order is fixed once (``self.learners``) and reused everywhere.
  4) No residual-on-validation modeling. Validation stays a true holdout.
"""

import logging
import pickle
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from config import (XGB_BASE_PARAMS, LGB_BASE_PARAMS, RANDOM_SEED, N_JOBS,
                    PHYSICS_MONOTONE_SIGNS)

logger = logging.getLogger("ensemble_models")


class HonestStacking:
    def __init__(self, seed: int = RANDOM_SEED, n_jobs: int = N_JOBS,
                 n_folds: int = 5, learners: Optional[List[str]] = None,
                 monotone: bool = False):
        self.seed = seed
        self.n_jobs = n_jobs
        self.n_folds = n_folds
        self.meta_model: Optional[RidgeCV] = None
        self.base_models: Dict[str, object] = {}
        self.learners: List[str] = []
        self._requested = learners
        self.oof_scores: Dict[str, float] = {}
        self.is_trained = False
        # same physics-prior constraints as the primary model; the stacking
        # target is log1p(TOTALBTU), so the aggregate sign map is valid here
        self.monotone = monotone

    # ---------------------------------------------------------------- learners
    def _available_learners(self) -> List[str]:
        avail = []
        if self._requested:
            req = self._requested
        else:
            req = ["xgb", "lgb", "catboost"]
        for name in req:
            if name == "xgb":
                try:
                    import xgboost  # noqa
                    avail.append(name)
                except ImportError:
                    pass
            elif name == "lgb":
                try:
                    import lightgbm  # noqa
                    avail.append(name)
                except ImportError:
                    pass
            elif name == "catboost":
                try:
                    import catboost  # noqa
                    avail.append(name)
                except ImportError:
                    pass
        return avail

    def _make(self, name: str, monotone_cols=None):
        mono = ([int(PHYSICS_MONOTONE_SIGNS.get(c, 0)) for c in monotone_cols]
                if (self.monotone and monotone_cols is not None) else None)
        if name == "xgb":
            from xgboost import XGBRegressor
            p = {k: v for k, v in XGB_BASE_PARAMS.items() if k != "early_stopping_rounds"}
            p.update({"n_estimators": 700, "n_jobs": self.n_jobs,
                      "random_state": self.seed})
            if mono is not None:
                # XGBoost 3.x needs the paren-string form when frames carry
                # feature names (a raw list fails inside Booster)
                p["monotone_constraints"] = "(" + ",".join(map(str, mono)) + ")"
            return XGBRegressor(**p)
        if name == "lgb":
            from lightgbm import LGBMRegressor
            p = dict(LGB_BASE_PARAMS)
            p.update({"n_estimators": 700, "n_jobs": self.n_jobs,
                      "random_state": self.seed})
            if mono is not None:
                p["monotone_constraints"] = mono
            return LGBMRegressor(**p)
        if name == "catboost":
            from catboost import CatBoostRegressor
            kw = {}
            if mono is not None:
                # CatBoost rejects tuples here: "must be str or a list"
                kw["monotone_constraints"] = list(mono)
            return CatBoostRegressor(iterations=700, learning_rate=0.05, depth=7,
                                     l2_leaf_reg=3.0, loss_function="RMSE",
                                     random_seed=self.seed, thread_count=self.n_jobs,
                                     verbose=False, allow_writing_files=False, **kw)
        raise ValueError(name)

    # -------------------------------------------------------------------- fit
    def fit(self, X_train: pd.DataFrame, y_log_total_train: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_log_total_val: Optional[pd.Series] = None,
            w_train: Optional[pd.Series] = None) -> Dict:
        t0 = time.time()
        self.learners = self._available_learners()
        if not self.learners:
            logger.warning("No base learners available; stacking skipped.")
            return {}
        logger.info("=" * 60)
        logger.info("HONEST STACKING ENSEMBLE v8.0 — target: log1p(TOTALBTU)")
        logger.info("=" * 60)
        logger.info(f"Base learners (fixed order): {self.learners}")

        y = pd.to_numeric(y_log_total_train, errors="coerce")
        ok = y.notna() & np.isfinite(y)
        X, y = X_train.loc[ok], y.loc[ok]
        w = w_train.loc[X.index] if w_train is not None else None

        # ---- Stage 1: KFold OOF
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        oof = pd.DataFrame(index=X.index, columns=self.learners, dtype=float)
        for name in self.learners:
            for tr_idx, te_idx in kf.split(X):
                m = self._make(name, monotone_cols=list(X.columns))
                kw = {"sample_weight": w.iloc[tr_idx]} if w is not None and name != "catboost" else {}
                m.fit(X.iloc[tr_idx], y.iloc[tr_idx], **kw)
                oof.iloc[te_idx, oof.columns.get_loc(name)] = m.predict(X.iloc[te_idx])
            self.oof_scores[name] = float(r2_score(y, oof[name]))
            logger.info(f"  OOF R2 {name:<10}: {self.oof_scores[name]:.4f}")
        meta_X = oof[self.learners].astype(float)      # fixed column order
        logger.info(f"OOF meta-features shape: {meta_X.shape}")

        # ---- Stage 2: meta-learner on OOF
        # NOTE (v8.1): the coefficients are solved under an explicit
        # non-negativity constraint. Clipping the unconstrained coefficients
        # after fitting - as v8.0 did - leaves the intercept solved for the
        # *unclipped* coefficients, so predict() applies a biased intercept
        # and the stacked score collapses below every base learner
        # (observed: base OOF 0.734/0.729/0.740, stacked 0.582).
        sw = w.values if w is not None else None
        best = None
        try:
            from sklearn.linear_model import Ridge
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            for a in (0.01, 0.1, 1.0, 10.0):
                cand = make_pipeline(StandardScaler(),
                                     Ridge(alpha=a, positive=True))
                cand.fit(meta_X, y.values, ridge__sample_weight=sw)
                r2 = r2_score(y, cand.predict(meta_X))
                if best is None or r2 > best[1]:
                    best = (cand, r2)
        except (TypeError, ImportError) as exc:      # older sklearn
            logger.warning(f"  Non-negative ridge unavailable ({exc}); "
                           "falling back to unconstrained RidgeCV")
            best = None
        if best is None:
            from sklearn.linear_model import RidgeCV
            cand = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0])
            cand.fit(meta_X, y.values, sample_weight=sw)
            best = (cand, r2_score(y, cand.predict(meta_X)))
        self.meta_model, oof_r2 = best
        coefs = self._meta_coefficients()
        coef = {n: float(c) for n, c in zip(self.learners, coefs)}
        logger.info(f"  Meta-learner coefficients (non-negative): {coef}")
        logger.info(f"  Sum of coefficients: {float(np.sum(coefs)):.4f} "
                    f"(intercept carries the remainder; weights are "
                    f"non-negative by construction)")
        logger.info(f"  OOF stacked R2: {oof_r2:.4f} "
                    f"(best base OOF: {max(self.oof_scores.values()):.4f})")

        # ---- Stage 3: retrain base learners on 100% of train
        logger.info("Retraining base models on 100% of train...")
        for name in self.learners:
            m = self._make(name, monotone_cols=list(X.columns))
            kw = {"sample_weight": w.values} if w is not None and name != "catboost" else {}
            m.fit(X, y, **kw)
            self.base_models[name] = m

        self.is_trained = True
        metrics = {"oof_scores": dict(self.oof_scores),
                   "oof_stacked_r2": float(oof_r2),
                   "meta_coef": {k: float(v) for k, v in coef.items()},
                   "fit_seconds": time.time() - t0}
        if X_val is not None and y_log_total_val is not None:
            pv = self.predict(X_val)
            metrics["val_r2_log"] = float(r2_score(y_log_total_val, pv))
            logger.info(f"  Validation (true holdout): R2(log)={metrics['val_r2_log']:.4f}")
        return metrics

    # ------------------------------------------------------------- meta utils
    def _meta_coefficients(self) -> np.ndarray:
        """Coefficients of the final meta-learner, whether it is a bare
        estimator or a StandardScaler->Ridge pipeline."""
        m = self.meta_model
        if hasattr(m, "steps"):                       # sklearn Pipeline
            return m.steps[-1][1].coef_
        return m.coef_

    # ---------------------------------------------------------------- predict
    def predict(self, X: pd.DataFrame) -> pd.Series:
        assert self.is_trained, "Ensemble not trained."
        base = pd.DataFrame(index=X.index, columns=self.learners, dtype=float)  # fixed order
        for name in self.learners:
            base[name] = self.base_models[name].predict(X)
        # predict() works for both the bare estimator and the pipeline,
        # which carries its own scaler into deployment
        return pd.Series(np.asarray(self.meta_model.predict(
                         base[self.learners].astype(float))),
                         index=X.index, name="stacked_log_total")

    # ------------------------------------------------------------ persistence
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"meta_model": self.meta_model, "base_models": self.base_models,
                         "learners": self.learners, "oof_scores": self.oof_scores,
                         "seed": self.seed, "n_jobs": self.n_jobs,
                         "n_folds": self.n_folds}, f)
        logger.info(f"Ensemble saved: {path}")

    @classmethod
    def load(cls, path) -> "HonestStacking":
        with open(path, "rb") as f:
            s = pickle.load(f)
        e = cls(seed=s["seed"], n_jobs=s["n_jobs"], n_folds=s["n_folds"])
        e.meta_model = s["meta_model"]
        e.base_models = s["base_models"]
        e.learners = s["learners"]
        e.oof_scores = s.get("oof_scores", {})
        e.is_trained = True
        return e
