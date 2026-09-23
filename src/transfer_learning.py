"""
Transfer Learning — v8.0
========================
Teacher/student knowledge distillation from ResStock 2025.1 (simulated,
~550k buildings) to RECS 2020 (surveyed, ~18.5k households).

v7.2 bug: the step silently did NOTHING (12 s, no logs) — the source-target
column check used the pre-rename name ("eui_total") after renaming to
"EUI_TOTAL", so ``source_y`` was None. v8.0:
  * harmonizes ResStock strings to RECS numeric codes,
  * injects climate-zone degree days (ResStock has no HDD/CDD fields),
  * recomputes the SAME physics features on ResStock,
  * trains a real LightGBM teacher on log1p(total kBtu) with holdout metrics,
  * adds TL_pred_logbtu / TL_pred_eui meta-features to RECS splits.

Leakage-free: the teacher never sees any RECS target; RECS rows are only
inputs at prediction time.
"""

import logging
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import (
    SQFT_COL, HDD_COL, CDD_COL, STATE_COL, CLIMATE_COL, WEIGHT_COL,
    BA_HDD_CDD, RS_TO_RECS_CODES, LGB_BASE_PARAMS, RANDOM_SEED, N_JOBS,
)
from physics_features import PhysicsFeatures

logger = logging.getLogger("transfer_learning")

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:  # pragma: no cover
    LIGHTGBM_AVAILABLE = False

KWH_TO_KBTU = 3.412


def harmonize_resstock(df: pd.DataFrame,
                       physics: Optional[PhysicsFeatures] = None,
                       seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Bring ResStock onto the RECS feature schema (best-effort)."""
    out = df.copy()
    rng = np.random.default_rng(seed)

    # string -> RECS numeric codes
    for recs_col, mapping in RS_TO_RECS_CODES.items():
        if recs_col in out.columns:
            v = out[recs_col]
            try:
                _is_num = pd.api.types.is_numeric_dtype(v.dtype)
            except TypeError:
                _is_num = False
            if not _is_num:
                out[recs_col] = v.astype(str).str.strip().str.lower().map(mapping)

    # HVAC equipment string -> EQUIPM numeric (RECS semantics: 3 = heat pump,
    # 12 = electric resistance, 7 = boiler, 2 = central warm-air furnace).
    # ResStock names cover both the generator ("Heat Pump") and the real
    # 2025.1 release ("Electricity ASHP", "Natural Gas Fuel Furnace", ...).
    if "EQUIPM" in out.columns:
        v = out["EQUIPM"]
        try:
            _is_num = pd.api.types.is_numeric_dtype(v.dtype)
        except TypeError:
            _is_num = False
        if not _is_num:
            s = v.astype(str).str.strip().str.lower()
            out["EQUIPM"] = pd.Series(np.select(
                [s.str.contains("ashp|heat pump|mshp|gshp"),
                 s.str.contains("baseboard|resistance|electric furnace|electric boiler|room heater|wall furnace|space heater"),
                 s.str.contains("boiler"),
                 s.str.contains("furnace")],
                [3.0, 12.0, 7.0, 2.0], default=np.nan), index=out.index)

    # envelope strings -> RECS physics inputs (DRAFTY / ADQINSUL / TYPEGLASS) so
    # that envelope upgrades are VISIBLE to the physics features and teachers
    def _first_num(v):
        m = re.search(r"(\d+\.?\d*)", str(v))
        return float(m.group(1)) if m else np.nan

    if "infiltration" in out.columns:           # "10 ACH50" -> DRAFTY 1..4
        ach = out["infiltration"].map(_first_num)
        out["DRAFTY"] = pd.Series(np.select(
            [ach >= 13, ach >= 8.5, ach >= 5.5], [1.0, 2.0, 3.0], default=4.0),
            index=out.index).where(ach.notna())

    roof_src = "insul_roof" if "insul_roof" in out.columns else (
        "insul_ceiling" if "insul_ceiling" in out.columns else None)
    if roof_src:                                 # "R-38" -> ADQINSUL 1..4
        r = out[roof_src].map(_first_num)
        out["ADQINSUL"] = pd.Series(np.select(
            [r < 19, r < 30, r < 38], [1.0, 2.0, 3.0], default=4.0),
            index=out.index).where(r.notna())

    if "windows" in out.columns:                 # -> TYPEGLASS 1/2/3
        s = out["windows"].astype(str).str.strip().str.lower()
        tg = pd.Series(np.select(
            [s.str.contains("single"), s.str.contains("triple")],
            [1.0, 3.0], default=2.0), index=out.index)
        tg = tg.mask(s.str.contains("low-e|low e"), 3.0)   # coating proxy (U≈0.35)
        out["TYPEGLASS"] = tg

    # numeric coercions for harmonized columns
    for c in (SQFT_COL, "STORIES", "BEDROOMS", "NHSLDMEM", "FUELHEAT", "EQUIPM"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # vintage strings ("1970s", "<1940") -> YEAR_BUILT
    if "vintage" in out.columns:
        def _vintage_year(v):
            s = str(v).strip()
            if s.startswith("<"):
                return 1935.0
            digits = "".join(ch for ch in s if ch.isdigit())
            return float(digits[:4]) + 5 if len(digits) >= 4 else np.nan
        out["YEAR_BUILT"] = out["vintage"].map(_vintage_year)

    # climate-zone degree days (ResStock has no survey HDD/CDD)
    if CLIMATE_COL in out.columns:
        hd = out[CLIMATE_COL].map(lambda z: BA_HDD_CDD.get(str(z).strip(), (None, None))[0])
        cd = out[CLIMATE_COL].map(lambda z: BA_HDD_CDD.get(str(z).strip(), (None, None))[1])
        out[HDD_COL] = pd.to_numeric(hd, errors="coerce") * rng.uniform(0.85, 1.15, len(out))
        out[CDD_COL] = pd.to_numeric(cd, errors="coerce") * rng.uniform(0.85, 1.15, len(out))

    # stories strings ("One", "Two") -> numeric
    if "STORIES" in out.columns:
        st_map = {"one": 1.0, "two": 2.0, "three": 3.0}
        v = out["STORIES"]
        mapped = v.astype(str).str.strip().str.lower().map(st_map)
        out["STORIES"] = pd.to_numeric(v, errors="coerce").fillna(mapped)

    # physics features on the harmonized frame
    physics = physics or PhysicsFeatures()
    out = physics.compute_features(out, verbose=False)
    return out


class TransferLearner:
    """ResStock teacher -> RECS KD meta-features."""

    def __init__(self, seed: int = RANDOM_SEED, n_jobs: int = N_JOBS):
        self.seed = seed
        self.n_jobs = n_jobs
        self.teacher = None
        self.teacher_features: List[str] = []
        self.medians: Optional[pd.Series] = None
        self.metrics: Dict = {}
        self.fitted = False
        self.upgrade_teachers: Dict[int, object] = {}

    # ------------------------------------------------- delta (savings) teachers
    def fit_delta_teachers(self, resstock: Dict[int, pd.DataFrame],
                           physics: Optional[PhysicsFeatures] = None,
                           n_estimators: int = 600) -> Dict[int, object]:
        """Train one SAVINGS model per upgrade: delta_i = (E_base - E_up)/E_base
        regressed on the baseline household features.

        Rationale: ResStock upgrades are true physical interventions inside the
        simulation, so paired per-building deltas are causal-flavoured
        conditional average treatment effects (CATE). Learning the delta
        directly is far more robust than comparing two independent teachers:
        within a single upgrade file the intervention features are nearly
        constant, so an upgrade-level teacher cannot learn their effect.

        Pairing order: bldg_id when available; positional when row counts are
        equal (ResStock releases are row-aligned); otherwise coarsened cell
        matching on stock characteristics.
        """
        self.delta_teachers = {}
        if not self.fitted or not LIGHTGBM_AVAILABLE:
            return self.delta_teachers
        base = resstock.get(0)
        if base is None or "TOTAL_ENERGY_KWH" not in base.columns:
            return self.delta_teachers
        hb = harmonize_resstock(base, physics=physics)
        eb = pd.to_numeric(hb["TOTAL_ENERGY_KWH"], errors="coerce")
        Xb = pd.DataFrame({c: (pd.to_numeric(hb[c], errors="coerce")
                               if c in hb.columns else np.nan)
                           for c in self.teacher_features}).fillna(self.medians)
        Xb = Xb[self.teacher_features]

        for u, df in resstock.items():
            if u == 0 or "TOTAL_ENERGY_KWH" not in df.columns:
                continue
            try:
                eu = pd.to_numeric(df["TOTAL_ENERGY_KWH"], errors="coerce")
                if "bldg_id" in df.columns and "bldg_id" in base.columns:
                    pos = pd.Series(np.arange(len(base)), index=base["bldg_id"].values)
                    hit = df["bldg_id"].map(pos)
                    mask = hit.notna() & eu.notna()
                    idx_b = hit[mask].to_numpy(dtype=int)
                    e_b, e_u = eb.iloc[idx_b].to_numpy(), eu[mask].to_numpy()
                    Xp = Xb.iloc[idx_b].reset_index(drop=True)
                    mode = f"bldg_id (n={mask.sum():,})"
                elif len(df) == len(base):
                    mask = (eu.notna() & eb.notna()).to_numpy()
                    e_b, e_u = eb.to_numpy()[mask], eu.to_numpy()[mask]
                    Xp = Xb.loc[mask].reset_index(drop=True)
                    mode = f"positional (n={mask.sum():,})"
                else:
                    Xp, e_b, e_u = self._cell_matched_pairs(base, df, eb, eu, Xb)
                    if Xp is None:
                        logger.warning(f"  Delta teacher {u}: files not pairable; skipped.")
                        continue
                    mode = f"cell-matched (n={len(Xp):,})"
                ok = (e_b > 2000) & (e_u >= 0)
                delta = np.clip((e_b[ok] - e_u[ok]) / e_b[ok] * 100.0, -50.0, 90.0)
                Xp = Xp.loc[ok].reset_index(drop=True)
                if len(delta) < 500:
                    logger.warning(f"  Delta teacher {u}: insufficient pairs; skipped.")
                    continue
                params = dict(LGB_BASE_PARAMS)
                params.update({"n_estimators": n_estimators, "n_jobs": self.n_jobs,
                               "random_state": self.seed})
                m = LGBMRegressor(**params)
                m.fit(Xp, delta)
                self.delta_teachers[u] = m
                logger.info(f"  Delta teacher {u}: savings model trained, {mode}, "
                            f"mean delta={float(np.mean(delta)):.1f}%")
            except Exception as e:  # pragma: no cover
                logger.warning(f"  Delta teacher {u} failed: {e}")
        return self.delta_teachers

    def _cell_matched_pairs(self, base, df, eb, eu, Xb):
        """Coarsened cell matching for upgrade files whose row subsets differ.

        v8.1 root-cause fix: v8.0 looked for the CANONICAL lowercase ResStock
        names ("sqft", "typehuq", "fuelheat", ...) but ``load_resstock`` has
        already renamed those columns to RECS-style names (TOTSQFT_EN,
        TYPEHUQ, FUELHEAT, ...) via RESSTOCK_RECS_MAP. Fewer than three keys
        survived the ``c in df.columns`` check, the function returned None and
        every subset upgrade (10/11/15) was silently skipped — the reason the
        paper's Table 5 had only two teacher rows. Keys are now resolved
        through the config alias map, coarsened (sqft in 500 ft2 bins) and
        NaN-tolerant (cell means are imputed instead of dropping cells).
        """
        from config import RS_MATCH_KEY_ALIASES, RS_CELL_MIN_CELLS

        def resolve(d):
            out = {}
            for logical, aliases in RS_MATCH_KEY_ALIASES.items():
                for a in aliases:
                    if a in d.columns:
                        out[logical] = a
                        break
            return out

        kb_map, ku_map = resolve(base), resolve(df)
        cand = [k for k in ("sqft", "vintage", "typehuq", "ba_climate",
                            "fuelheat", "occupants") if k in kb_map and k in ku_map]
        if len(cand) < 3:
            logger.warning(f"    cell matching: only {len(cand)} common keys "
                           f"({cand}); need >= 3")
            return None, None, None

        def keys(d, m):
            k = []
            for logical in cand:
                v = d[m[logical]]
                if logical == "sqft":
                    v = (pd.to_numeric(v, errors="coerce") / 500.0).round()
                k.append(v.astype(str).fillna("NA").values)
            return k

        kb, ku = keys(base, kb_map), keys(df, ku_map)
        gb_e = eb.groupby(kb, observed=True).mean()
        gu_e = eu.groupby(ku, observed=True).mean()
        Xb_cell = Xb.groupby(kb, observed=True).mean()
        # impute cell means instead of dropna(): a single NaN teacher feature
        # per cell previously annihilated almost every cell
        Xb_cell = Xb_cell.fillna(self.medians).fillna(0.0)
        common = gb_e.dropna().index.intersection(gu_e.dropna().index)
        common = common.intersection(Xb_cell.index)
        if len(common) < RS_CELL_MIN_CELLS:
            logger.warning(f"    cell matching: {len(common)} common cells "
                           f"(< {RS_CELL_MIN_CELLS} required) on keys {cand}")
            return None, None, None
        Xc = Xb_cell.loc[common, self.teacher_features].reset_index(drop=True)
        return Xc, gb_e.loc[common].to_numpy(), gu_e.loc[common].to_numpy()

    def upgrade_savings_pct(self, X: pd.DataFrame, upgrade: int) -> Optional[pd.Series]:
        """Per-household predicted savings % from the simulation-derived delta
        (CATE) teacher for the given upgrade."""
        if not self.fitted or upgrade not in getattr(self, "delta_teachers", {}):
            return None
        Z = pd.DataFrame(index=X.index)
        for c in self.teacher_features:
            Z[c] = pd.to_numeric(X[c], errors="coerce") if c in X.columns else np.nan
        Z = Z.fillna(self.medians)[self.teacher_features]
        return pd.Series(self.delta_teachers[upgrade].predict(Z), index=X.index)

    # ------------------------------------------------------------------ fit
    def fit_teacher(self, source_df: pd.DataFrame,
                    candidate_features: List[str],
                    holdout_frac: float = 0.10) -> bool:
        if not LIGHTGBM_AVAILABLE:
            logger.warning("LightGBM not available; transfer learning skipped.")
            return False

        df = source_df
        if "TOTALBTU" not in df.columns:
            if "TOTAL_ENERGY_KWH" in df.columns:
                df = df.copy()
                df["TOTALBTU"] = pd.to_numeric(df["TOTAL_ENERGY_KWH"],
                                               errors="coerce") * KWH_TO_KBTU
            elif "sqft" in df.columns and "EUI_TOTAL" in df.columns:
                df = df.copy()
                df["TOTALBTU"] = (pd.to_numeric(df["EUI_TOTAL"], errors="coerce") *
                                  pd.to_numeric(df[SQFT_COL], errors="coerce") * KWH_TO_KBTU)
            else:
                logger.warning("Teacher: no usable target column; skipped.")
                return False

        y = np.log1p(pd.to_numeric(df["TOTALBTU"], errors="coerce").clip(lower=0))
        feats = [c for c in candidate_features if c in df.columns]
        X = df[feats].apply(pd.to_numeric, errors="coerce")
        good = [c for c in X.columns if X[c].notna().mean() > 0.5 and X[c].std() > 0]
        X = X[good]
        valid = y.notna() & np.isfinite(y)
        X, y = X.loc[valid], y.loc[valid]
        if len(y) < 1000 or not good:
            logger.warning(f"Teacher: insufficient data (n={len(y)}, feats={len(good)}); skipped.")
            return False

        self.medians = X.median().fillna(0.0)
        X = X.fillna(self.medians)
        self.teacher_features = good

        n_hold = int(len(X) * holdout_frac)
        X_tr, y_tr = X.iloc[:-n_hold], y.iloc[:-n_hold]
        X_hold, y_hold = X.iloc[-n_hold:], y.iloc[-n_hold:]

        logger.info(f"Training ResStock teacher: n={len(X_tr):,}, features={len(good)}")
        params = dict(LGB_BASE_PARAMS)
        params.update({"n_estimators": 800, "n_jobs": self.n_jobs,
                       "random_state": self.seed})
        self.teacher = LGBMRegressor(**params)
        self.teacher.fit(X_tr, y_tr)

        pred_hold = self.teacher.predict(X_hold)
        ss_res = float(((y_hold - pred_hold) ** 2).sum())
        ss_tot = float(((y_hold - y_hold.mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        rmse_kbtu = float(np.sqrt(np.mean((np.expm1(y_hold) - np.expm1(pred_hold)) ** 2)))
        self.metrics = {"teacher_holdout_r2_log": r2, "teacher_holdout_rmse_kbtu": rmse_kbtu,
                        "n_train": len(X_tr), "n_features": len(good)}
        logger.info(f"  Teacher holdout: R2(log)={r2:.4f}, RMSE={rmse_kbtu:,.0f} kBtu")
        self.fitted = True
        return True

    # ------------------------------------------------------------- transform
    def add_meta_features(self, X: pd.DataFrame, sqft: Optional[pd.Series] = None
                          ) -> pd.DataFrame:
        """Append TL_pred_logbtu / TL_pred_eui (leakage-free: teacher never
        saw RECS targets)."""
        if not self.fitted:
            return X
        out = X.copy()
        Z = pd.DataFrame(index=out.index)
        for c in self.teacher_features:
            Z[c] = pd.to_numeric(out[c], errors="coerce") if c in out.columns else np.nan
        Z = Z.fillna(self.medians)
        pred_log = pd.Series(self.teacher.predict(Z[self.teacher_features]),
                             index=out.index)
        out["TL_pred_logbtu"] = pred_log
        if sqft is None and SQFT_COL in out.columns:
            sqft = pd.to_numeric(out[SQFT_COL], errors="coerce")
        if sqft is not None:
            out["TL_pred_eui"] = np.expm1(pred_log) / sqft.clip(lower=1)
        else:
            out["TL_pred_eui"] = np.nan
        logger.info(f"  KD meta-features added: TL_pred_logbtu mean={pred_log.mean():.3f}")
        return out
