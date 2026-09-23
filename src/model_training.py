"""
Model Training — v8.1  (Energy-Conserving Multi-Task Architecture)
==================================================================
Architecture (novelty: physics-consistent dual-target formulation):
  * Primary model predicts log1p(TOTALBTU)  (total site energy, kBtu).
  * Four share models predict end-use fractions of the total
    (heating / cooling / dhw / lighting). Predictions are clipped to
    [0,1] and renormalized so that sum(shares) <= 1 — the remainder is
    the implicit "other/appliances" use (energy conservation by construction).
  * Adaptive per-task routing: for every end use, a DIRECT EUI model is
    trained as well; on the VALIDATION split the better route is selected.
  * EUI is derived as total_pred / sqft; a v7-style direct-EUI model is
    trained too, purely as an ablation baseline.
  * Uncertainty: quantile regression (q10/q50/q90, log space) +
    SPLIT-CONFORMAL calibration on a dedicated, untouched calibration split.

v8.1 changes (reviewer finding #2 — conformal validity):
  * ``fit`` accepts ``X_calib``/``y_total_calib``. The conformal residual
    quantile is computed THERE and nowhere else. The validation split — which
    drives early stopping, routing and Optuna — is no longer used for
    calibration, restoring the exchangeability assumption behind the
    finite-sample coverage guarantee.
  * If no calibration split is supplied (e.g. secondary experiments), the
    fallback is CROSS-CONFORMAL: residuals from K-fold out-of-fold predictions
    inside the training split. Both modes are recorded in the report so the
    manuscript can state exactly which guarantee applies.
  * ``fixed_params`` lets an ablation arm reuse the EXACT hyper-parameters of
    the arm it is compared against (reviewer finding #1: the v8.0 TL ablation
    was trained without Optuna while the primary could be tuned — an unfair
    comparison that manufactured the "+0.8 pp" TL gain).

Order inside fit(): primary -> shares -> routing -> ablation -> quantiles
-> conformal -> report.
"""

import logging
import pickle
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from config import (
    PHYSICS_MONOTONE_SIGNS,
    ENDUSE_TASKS, SQFT_COL, XGB_BASE_PARAMS, LGB_BASE_PARAMS,
    QUANTILES, CONFORMAL_ALPHA, RANDOM_SEED, N_JOBS,
)

logger = logging.getLogger("model_training")

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:  # pragma: no cover
    XGB_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:  # pragma: no cover
    LIGHTGBM_AVAILABLE = False


@dataclass
class TrainReport:
    val_r2_total: float = np.nan
    conformal_q: float = np.nan
    conformal_mode: str = "none"       # "calib-split" | "cross-conformal" | "none"
    n_calib: int = 0
    n_features: int = 0
    routing: Dict = field(default_factory=dict)
    best_params: Dict = field(default_factory=dict)
    fit_seconds: float = 0.0


class EnergyConservingModel:
    def __init__(self, random_seed: int = RANDOM_SEED, n_jobs: int = N_JOBS,
                 use_xgb: bool = True, conserve_energy: bool = True,
                 monotone: bool = False):
        self.random_seed = random_seed
        self.n_jobs = n_jobs
        self.use_xgb = use_xgb and XGB_AVAILABLE
        # fail fast, not with a NameError three layers into fit(): the model
        # needs exactly one GBM backend and says so at construction time
        if not self.use_xgb and not LIGHTGBM_AVAILABLE:
            raise ImportError(
                "EnergyConservingModel requires xgboost or lightgbm and "
                "neither could be imported. Install the project "
                "requirements: pip install -r requirements.txt")
        self.conserve_energy = conserve_energy
        # Physics-derived monotonicity constraints, applied ONLY to the
        # aggregate total-energy model (see PHYSICS_MONOTONE_SIGNS scope
        # note). Domain-knowledge priors: no target information is used to
        # build them, so this cannot leak.
        self.monotone = monotone

        self.total_model = None
        self.share_models: Dict[str, object] = {}
        self.direct_task_models: Dict[str, object] = {}
        self.task_routing: Dict[str, str] = {}        # "share" | "direct" (chosen on val)
        self.task_val_scores: Dict[str, Dict] = {}
        self.quantile_models: Dict[float, object] = {}
        self.direct_eui_model = None                  # ablation: single direct-EUI model
        self.conformal_q: Optional[float] = None
        self.conformal_mode: str = "none"
        self.best_params: Dict = {}
        self.metrics: Dict = {}
        self.feature_names: List[str] = []
        self.is_trained = False

    # ---------------------------------------------------------------- helpers
    def _base_params(self) -> Dict:
        p = dict(XGB_BASE_PARAMS) if self.use_xgb else dict(LGB_BASE_PARAMS)
        p["n_jobs"] = self.n_jobs
        p["random_state"] = self.random_seed
        return p

    def _monotone_for(self, columns) -> List[int]:
        """Per-column constraint vector aligned to `columns` order."""
        return [int(PHYSICS_MONOTONE_SIGNS.get(c, 0)) for c in columns]

    def _make_regressor(self, params: Dict, objective: Optional[str] = None,
                        monotone_cols=None):
        if self.use_xgb:
            p = dict(params)
            if objective:
                p["objective"] = objective
            if self.monotone and monotone_cols is not None:
                # XGBoost 3.x wants the paren-string form (or a name->sign
                # dict) when the frame carries feature names; a raw list
                # raises AttributeError inside Booster._configure_constraints
                p["monotone_constraints"] = "(" + ",".join(
                    map(str, self._monotone_for(monotone_cols))) + ")"
            return XGBRegressor(**p)
        p = dict(params)
        p.pop("early_stopping_rounds", None)
        if objective == "reg:quantileerror":
            p["objective"] = "quantile"
        if self.monotone and monotone_cols is not None:
            p["monotone_constraints"] = self._monotone_for(monotone_cols)
        return LGBMRegressor(**p)

    def _fit_with_es(self, model, X_tr, y_tr, X_v, y_v, w_tr, w_v):
        """Fit with early stopping on VALIDATION when possible. The validation
        split is a selection device; it is deliberately NOT the conformal
        calibration sample (see fit step 6)."""
        is_xgb = model.__class__.__name__.startswith("XGB")
        fit_kw = {}
        if w_tr is not None:
            fit_kw["sample_weight"] = w_tr
        if X_v is not None and y_v is not None:
            fit_kw["eval_set"] = [(X_v, y_v)]
            if is_xgb:
                if w_v is not None:
                    fit_kw["sample_weight_eval_set"] = [w_v]
                model.set_params(early_stopping_rounds=100)
                fit_kw["verbose"] = False
            else:
                try:
                    import lightgbm as lgb
                    fit_kw["callbacks"] = [lgb.early_stopping(100, verbose=False)]
                except Exception:
                    pass
        model.fit(X_tr, y_tr, **fit_kw)
        return model

    def _optimize(self, X_tr, y_tr, X_v, y_v, w_tr, n_trials: int,
                  timeout_min: int) -> Dict:
        """Optuna search on the primary log-total target (optional). Runs on
        train+val only; the calib and test splits are never touched."""
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            logger.warning("Optuna not installed; using default parameters.")
            return {}

        def objective(trial):
            p = {
                "n_estimators": 2000,
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.10, log=True),
                "max_depth": trial.suggest_int("max_depth", 4, 9),
                "min_child_weight": trial.suggest_int("min_child_weight", 2, 20),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 3.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 5.0, log=True),
                "tree_method": "hist", "n_jobs": self.n_jobs,
                "random_state": self.random_seed,
            }
            if self.monotone:
                mono = self._monotone_for(X_tr.columns)
                p["monotone_constraints"] = (
                    "(" + ",".join(map(str, mono)) + ")" if self.use_xgb
                    else mono)
            m = XGBRegressor(**p, early_stopping_rounds=80) if self.use_xgb else \
                self._make_regressor({**LGB_BASE_PARAMS, **p})
            if X_v is not None:
                m.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_v, y_v)])
                pred = m.predict(X_v)
                return 1 - r2_score(y_v, pred)
            m.fit(X_tr, y_tr, sample_weight=w_tr)
            pred = m.predict(X_tr)
            return 1 - r2_score(y_tr, pred)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, timeout=timeout_min * 60,
                       show_progress_bar=False)
        logger.info(f"  Optuna: {len(study.trials)} trials, best loss={study.best_value:.5f}")
        # dozens of trial fits leave the allocator high-water elevated;
        # return it to the OS before the rest of the pipeline continues
        import gc
        gc.collect()
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:      # pragma: no cover - non-glibc platforms
            pass
        return dict(study.best_params)

    # -------------------------------------------------------------------- fit
    def fit(self, X_train: pd.DataFrame, y_total_train: pd.Series,
            shares_train: pd.DataFrame,
            X_val: Optional[pd.DataFrame] = None,
            y_total_val: Optional[pd.Series] = None,
            shares_val: Optional[pd.DataFrame] = None,
            eui_train: Optional[Dict[str, pd.Series]] = None,
            eui_val: Optional[Dict[str, pd.Series]] = None,
            w_train: Optional[pd.Series] = None,
            w_val: Optional[pd.Series] = None,
            X_calib: Optional[pd.DataFrame] = None,
            y_total_calib: Optional[pd.Series] = None,
            fixed_params: Optional[Dict] = None,
            optimize: bool = False, n_trials: int = 60,
            timeout_min: int = 30) -> TrainReport:
        """Fit the full multi-task architecture.

        Parameters
        ----------
        X_calib, y_total_calib : dedicated conformal calibration split. Must
            NOT have been used for early stopping, routing, tuning or feature
            selection. If omitted, calibration falls back to cross-conformal
            OOF residuals inside the training split (weaker, but still valid).
        fixed_params : if given, the primary model uses exactly these
            hyper-parameters and skips Optuna. Ablation arms pass the primary
            arm's fitted parameters here so both arms are trained under an
            identical protocol (fair paired comparison).
        """
        t0 = time.time()
        self.feature_names = list(X_train.columns)
        logger.info("=" * 60)
        logger.info("ENERGY-CONSERVING MULTI-TASK TRAINING v8.1")
        logger.info(f"  Primary: log1p(TOTALBTU) | Shares: {ENDUSE_TASKS} | "
                    f"Features: {X_train.shape[1]} | Samples: {len(X_train):,}")
        if fixed_params:
            logger.info(f"  Fixed hyper-parameters supplied ({len(fixed_params)} keys) "
                        "- Optuna skipped (paired-ablation protocol).")
        logger.info("=" * 60)

        y_tr = np.log1p(pd.to_numeric(y_total_train, errors="coerce").clip(lower=0))
        y_v = np.log1p(pd.to_numeric(y_total_val, errors="coerce").clip(lower=0)) \
            if y_total_val is not None else None
        valid_t = y_tr.notna() & np.isfinite(y_tr)
        X_tr, y_tr = X_train.loc[valid_t], y_tr.loc[valid_t]
        w_tr = w_train.loc[X_tr.index] if w_train is not None else None

        # ---- 1) primary total-energy model
        logger.info("  Training primary total-energy model...")
        params = self._base_params()
        if fixed_params:
            params.update({k: v for k, v in fixed_params.items()
                           if k not in ("n_jobs", "random_state")})
        elif optimize:
            best = self._optimize(X_tr, y_tr, X_val, y_v, w_tr, n_trials, timeout_min)
            params.update(best)
            self.best_params = best
        self.total_model = self._make_regressor(
            params, monotone_cols=list(X_tr.columns))
        if self.monotone:
            n_c = sum(1 for v in self._monotone_for(X_tr.columns) if v)
            logger.info(f"    Monotonicity constraints active on {n_c} "
                        f"physics-signed features (aggregate model only).")
        self.total_model = self._fit_with_es(self.total_model, X_tr, y_tr,
                                             X_val, y_v, w_tr, w_val)
        n_iter = getattr(self.total_model, "best_iteration", None)
        if n_iter:
            logger.info(f"    Early stopping at iteration {n_iter}")

        report = TrainReport(n_features=X_train.shape[1])
        if X_val is not None and y_v is not None:
            report.val_r2_total = float(r2_score(y_v, self.total_model.predict(X_val)))

        # ---- 2) end-use share models
        for task in ENDUSE_TASKS:
            s_col = f"SHARE_{task.upper()}"
            # shares are optional: a caller that only wants the total-energy
            # model passes None, and every end-use task is then skipped
            if shares_train is None or s_col not in shares_train.columns:
                continue
            s_tr = pd.to_numeric(shares_train[s_col], errors="coerce").clip(0, 1)
            valid_s = s_tr.notna() & np.isfinite(s_tr) & s_tr.index.isin(X_tr.index)
            s_tr = s_tr.loc[valid_s]
            if len(s_tr) < 500:
                logger.warning(f"    Share model '{task}': only {len(s_tr)} rows; skipped.")
                continue
            sp = dict(LGB_BASE_PARAMS) if LIGHTGBM_AVAILABLE else self._base_params()
            sp["n_estimators"] = 1500
            sp["learning_rate"] = 0.03
            sm = LGBMRegressor(**sp) if LIGHTGBM_AVAILABLE else self._make_regressor(sp)
            s_v = pd.to_numeric(shares_val[s_col], errors="coerce").clip(0, 1) \
                if (shares_val is not None and s_col in shares_val.columns) else None
            sm = self._fit_with_es(sm, X_tr.loc[s_tr.index], s_tr, X_val, s_v,
                                   w_tr.loc[s_tr.index] if w_tr is not None else None, None)
            self.share_models[task] = sm
            logger.info(f"    Share model '{task}': n={len(s_tr):,}, "
                        f"mean share={s_tr.mean():.3f}")

        # ---- 3b) direct end-use EUI models + adaptive per-task routing
        # Routing is a model-SELECTION step, so it uses the validation split
        # (never the conformal calibration split, never the test split).
        if eui_train and X_val is not None and eui_val is not None:
            X_v = X_val
            sqft_v = pd.to_numeric(X_v[SQFT_COL], errors="coerce").clip(lower=1) \
                if SQFT_COL in X_v.columns else pd.Series(2000.0, index=X_v.index)
            total_v = pd.Series(np.expm1(self.total_model.predict(X_v)), index=X_v.index)
            for task in ENDUSE_TASKS:
                key = f"EUI_{task.upper()}"
                if key not in eui_train or key not in eui_val:
                    continue
                y_dtr = pd.to_numeric(eui_train[key], errors="coerce").loc[
                    eui_train[key].index.intersection(X_tr.index)]
                valid_d = y_dtr.notna() & np.isfinite(y_dtr)
                if valid_d.sum() < 500:
                    continue
                params_t = dict(LGB_BASE_PARAMS) if LIGHTGBM_AVAILABLE else self._base_params()
                params_t["n_estimators"] = 1500
                params_t["learning_rate"] = 0.03
                dm = LGBMRegressor(**params_t) if LIGHTGBM_AVAILABLE else self._make_regressor(params_t)
                y_dv = pd.to_numeric(eui_val[key], errors="coerce").loc[
                    eui_val[key].index.intersection(X_v.index)]
                dm = self._fit_with_es(dm, X_tr.loc[y_dtr.index[valid_d]],
                                       y_dtr.loc[valid_d], X_v, y_dv, None, None)
                self.direct_task_models[task] = dm

                r2_share, r2_direct = -np.inf, -np.inf
                vv = y_dv.notna() & np.isfinite(y_dv)
                if vv.sum() > 100:
                    if task in self.share_models:
                        pred_share = pd.Series(
                            np.clip(self.share_models[task].predict(X_v), 0, 1)
                            * total_v / sqft_v, index=X_v.index)
                        r2_share = r2_score(y_dv.loc[vv], pred_share.loc[vv])
                    pred_direct = pd.Series(np.maximum(dm.predict(X_v), 0), index=X_v.index)
                    r2_direct = r2_score(y_dv.loc[vv], pred_direct.loc[vv])
                self.task_routing[task] = "direct" if r2_direct > r2_share else "share"
                self.task_val_scores[task] = {"share_r2": float(r2_share),
                                              "direct_r2": float(r2_direct)}
                logger.info(f"    Routing '{task}': share R2={r2_share:.4f} vs "
                            f"direct R2={r2_direct:.4f} -> {self.task_routing[task].upper()}")
        else:
            for task in self.share_models:
                self.task_routing[task] = "share"
        report.routing = dict(self.task_routing)

        # ---- 4) ablation baseline: direct EUI model
        if eui_train and "AGGREGATE" in eui_train:
            y_a = pd.to_numeric(eui_train["AGGREGATE"], errors="coerce")
            va = y_a.notna() & np.isfinite(y_a) & y_a.index.isin(X_tr.index)
            if va.sum() > 500:
                params_a = self._base_params()
                params_a["n_estimators"] = 1200
                self.direct_eui_model = self._make_regressor(params_a)
                y_av = pd.to_numeric(eui_val["AGGREGATE"], errors="coerce") \
                    if (eui_val and "AGGREGATE" in eui_val) else None
                self.direct_eui_model = self._fit_with_es(
                    self.direct_eui_model, X_tr.loc[y_a.index[va]], y_a.loc[va],
                    X_val, y_av, None, None)
                logger.info("    Ablation baseline (direct EUI) trained.")

        # ---- 5) quantile models (log space, train only)
        for q in QUANTILES:
            try:
                if self.use_xgb:
                    qm = XGBRegressor(**{k: v for k, v in self._base_params().items()
                                         if k != "early_stopping_rounds"},
                                      objective="reg:quantileerror", quantile_alpha=q)
                else:
                    qm = LGBMRegressor(**{**LGB_BASE_PARAMS, "objective": "quantile",
                                          "alpha": q, "n_estimators": 800})
                qm.fit(X_tr, y_tr, sample_weight=w_tr)
                self.quantile_models[q] = qm
            except Exception as e:
                logger.warning(f"    Quantile model q={q} failed: {e}")

        # ---- 6) conformal calibration — UNTOUCHED calib split (preferred)
        if X_calib is not None and y_total_calib is not None:
            y_c = np.log1p(pd.to_numeric(y_total_calib, errors="coerce").clip(lower=0))
            ok_c = y_c.notna() & np.isfinite(y_c)
            X_c = X_calib.loc[y_c.index[ok_c]]
            res = np.abs(y_c.loc[ok_c].values - self.total_model.predict(X_c))
            n = len(res)
            if n >= 50:
                k = int(np.ceil((n + 1) * (1 - CONFORMAL_ALPHA)))
                self.conformal_q = float(np.sort(res)[min(k, n) - 1])
                self.conformal_mode = "calib-split"
                report.n_calib = n
                logger.info(f"    Conformal [calib-split, n={n:,}, untouched]: "
                            f"q_hat={self.conformal_q:.4f} (90% interval, log-space)")
            else:
                logger.warning(f"    Calibration split too small (n={n}); "
                               "falling back to cross-conformal.")
        if self.conformal_q is None:
            # Fallback: cross-conformal on out-of-fold residuals within train.
            # Valid under exchangeability; slightly wider intervals in theory.
            try:
                kf = KFold(n_splits=5, shuffle=True, random_state=self.random_seed)
                oof = np.full(len(X_tr), np.nan)
                for tr_i, te_i in kf.split(X_tr):
                    # early stopping needs a validation set, which a fold has
                    # none of; drop it or XGBoost raises
                    fold_params = {k: v for k, v in self._base_params().items()
                                   if k != "early_stopping_rounds"}
                    m = self._make_regressor(fold_params)
                    kw = {"sample_weight": w_tr.iloc[tr_i]} if w_tr is not None else {}
                    m.fit(X_tr.iloc[tr_i], y_tr.iloc[tr_i], **kw)
                    oof[te_i] = m.predict(X_tr.iloc[te_i])
                res = np.abs(y_tr.values - oof)
                res = res[np.isfinite(res)]
                n = len(res)
                k = int(np.ceil((n + 1) * (1 - CONFORMAL_ALPHA)))
                self.conformal_q = float(np.sort(res)[min(k, n) - 1])
                self.conformal_mode = "cross-conformal"
                report.n_calib = n
                logger.info(f"    Conformal [cross-conformal OOF, n={n:,}]: "
                            f"q_hat={self.conformal_q:.4f}")
            except Exception as e:  # pragma: no cover
                logger.warning(f"    Cross-conformal fallback failed: {e}")
        report.conformal_q = self.conformal_q if self.conformal_q is not None else np.nan
        report.conformal_mode = self.conformal_mode

        self.is_trained = True
        report.fit_seconds = time.time() - t0
        report.best_params = dict(self.best_params)
        return report

    # ---------------------------------------------------------------- predict
    def predict(self, X: pd.DataFrame) -> Dict[str, pd.Series]:
        assert self.is_trained, "Model not trained."
        out: Dict[str, pd.Series] = {}
        pred_log = self.total_model.predict(X)
        total = pd.Series(np.expm1(pred_log), index=X.index)
        out["total_btu"] = total

        sqft = pd.to_numeric(X[SQFT_COL], errors="coerce").clip(lower=1) \
            if SQFT_COL in X.columns else pd.Series(2000.0, index=X.index)
        out["eui"] = total / sqft

        shares: Dict[str, pd.Series] = {}
        for task, sm in self.share_models.items():
            shares[task] = pd.Series(np.clip(sm.predict(X), 0, 1), index=X.index)
        if self.conserve_energy and shares:
            ssum = sum(shares.values())
            over = ssum > 1.0
            if over.any():
                for task in shares:
                    shares[task] = shares[task].where(~over,
                                                      shares[task] / ssum.clip(lower=1e-6))
            shares["other"] = (1.0 - sum(shares.values())).clip(lower=0)

        for task, s in shares.items():
            out[f"share_{task}"] = s
            if task == "other":
                continue
            out[f"eui_{task}_share"] = s * total / sqft
            out[f"btu_{task}"] = s * total
            if self.task_routing.get(task) == "direct" and task in self.direct_task_models:
                out[f"eui_{task}"] = pd.Series(
                    np.maximum(self.direct_task_models[task].predict(X), 0.0), index=X.index)
            else:
                out[f"eui_{task}"] = out[f"eui_{task}_share"]
        for task, dm in self.direct_task_models.items():
            if f"eui_{task}" not in out:
                out[f"eui_{task}"] = pd.Series(np.maximum(dm.predict(X), 0.0), index=X.index)
        if self.direct_eui_model is not None:
            out["eui_direct_ablation"] = pd.Series(
                np.maximum(self.direct_eui_model.predict(X), 0.0), index=X.index)
        return out

    def predict_intervals(self, X: pd.DataFrame) -> pd.DataFrame:
        """90% interval for total kBtu: conformal (preferred) or quantiles."""
        pred_log = pd.Series(self.total_model.predict(X), index=X.index)
        if self.conformal_q is not None:
            lo = np.expm1(pred_log - self.conformal_q).clip(lower=0)
            hi = np.expm1(pred_log + self.conformal_q)
        elif 0.1 in self.quantile_models and 0.9 in self.quantile_models:
            lo = np.expm1(self.quantile_models[0.1].predict(X)).clip(lower=0)
            hi = np.expm1(self.quantile_models[0.9].predict(X))
        else:
            return pd.DataFrame(index=X.index)
        return pd.DataFrame({"q10": lo.values, "q50": np.expm1(pred_log).values,
                             "q90": hi.values}, index=X.index)

    def conformal_coverage_curve(self, X_calib: pd.DataFrame, y_calib,
                                 alphas=(0.20, 0.10, 0.05)) -> Dict[float, float]:
        """Nominal-vs-empirical coverage on the calibration split, for the
        reliability figure (empirical coverage at each nominal level)."""
        out = {}
        if self.conformal_q is None:
            return out
        y_c = np.log1p(pd.to_numeric(y_calib, errors="coerce").clip(lower=0))
        ok = y_c.notna() & np.isfinite(y_c)
        pred = self.total_model.predict(X_calib.loc[y_c.index[ok]])
        yv = y_c.loc[ok].values
        for a in alphas:
            q = self.conformal_q * np.sqrt(-np.log(a) / -np.log(CONFORMAL_ALPHA)) \
                if a != CONFORMAL_ALPHA else self.conformal_q
            cov = float(np.mean((yv >= pred - q) & (yv <= pred + q)))
            out[1.0 - a] = cov
        return out

    # ------------------------------------------------------------ persistence
    def save(self, path):
        state = {
            "total_model": self.total_model, "share_models": self.share_models,
            "direct_task_models": self.direct_task_models,
            "task_routing": self.task_routing, "task_val_scores": self.task_val_scores,
            "quantile_models": self.quantile_models,
            "direct_eui_model": self.direct_eui_model,
            "conformal_q": self.conformal_q, "conformal_mode": self.conformal_mode,
            "best_params": self.best_params, "feature_names": self.feature_names,
            "conserve_energy": self.conserve_energy, "use_xgb": self.use_xgb,
            "monotone": self.monotone,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        logger.info(f"Model saved: {path}")

    @classmethod
    def load(cls, path) -> "EnergyConservingModel":
        with open(path, "rb") as f:
            state = pickle.load(f)
        m = cls(use_xgb=state.get("use_xgb", True),
                conserve_energy=state.get("conserve_energy", True),
                monotone=state.get("monotone", False))
        m.total_model = state["total_model"]
        m.share_models = state["share_models"]
        m.direct_task_models = state.get("direct_task_models", {})
        m.task_routing = state.get("task_routing", {})
        m.task_val_scores = state.get("task_val_scores", {})
        m.quantile_models = state["quantile_models"]
        m.direct_eui_model = state.get("direct_eui_model")
        m.conformal_q = state["conformal_q"]
        m.conformal_mode = state.get("conformal_mode", "unknown")
        m.best_params = state.get("best_params", {})
        m.feature_names = state.get("feature_names", [])
        m.is_trained = True
        return m
