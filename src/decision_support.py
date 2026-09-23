"""
Decision Support / Counterfactual Retrofit Analysis — v8.1
==========================================================
Retrofit scenarios edit RAW survey attributes; the physics features are then
RECOMPUTED (not hand-patched), features rebuilt through the SAME fitted
FeatureEngineer (transform-only), and the trained model predicts the new
consumption.

v8.1 (reviewer finding #7 — Table 5 / air-sealing -8.69 %):
  * ResStock upgrade deltas are now estimated with a MATCHED-CELL estimator.
    When no bldg_id exists, the v8.0 code compared population means of two
    files that cover DIFFERENT subsets of the stock; when the upgrade file
    covers only high-consumption households (air sealing targets the leaky
    ones), the naive mean ratio is composition-biased and can come out
    NEGATIVE even though the measure saves energy. The matched estimator
    computes baseline/upgraded means inside common
    (climate x building-type x heating-fuel) cells and aggregates with
    baseline-cell weights, which removes the composition effect.
  * Both estimates (naive and matched) plus coverage and an explanatory note
    are reported side by side, so Table 5 is never "half empty" or silent
    about a pathology again.
"""

import logging
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

from config import RS_MATCH_KEY_ALIASES, RS_CELL_MIN_CELLS

logger = logging.getLogger("decision_support")


# scenario name -> (raw-attribute edit function, matching ResStock upgrade id)
def _hvac_reference(df):
    """Efficient modern furnace (reference HVAC): upgrade only homes with old
    equipment (EQUIPM in {4,5,7}); homes already efficient stay unchanged."""
    d = df.copy()
    if "EQUIPM" in d.columns:
        eq = pd.to_numeric(d["EQUIPM"], errors="coerce")
        d["EQUIPM"] = np.where(eq.isin([4, 5, 7]), 3, eq)
    return d


def _hvac_ashp(df):
    """Cold-climate ASHP: switch main heating to electric heat pump
    (RECS-2020 semantics: FUELHEAT=5, EQUIPM=3, HEATPUMP=1)."""
    d = df.copy()
    if "FUELHEAT" in d.columns:
        d["FUELHEAT"] = 5
    if "EQUIPM" in d.columns:
        d["EQUIPM"] = 3
    if "HEATPUMP" in d.columns:
        d["HEATPUMP"] = 1
    return d


def _air_sealing(df):
    d = df.copy()
    if "DRAFTY" in d.columns:
        d["DRAFTY"] = 4
    return d


def _attic_insulation(df):
    d = df.copy()
    if "ADQINSUL" in d.columns:
        d["ADQINSUL"] = 4
    return d


def _estar_windows(df):
    d = df.copy()
    if "TYPEGLASS" in d.columns:
        d["TYPEGLASS"] = 2
    return d


SCENARIOS = {
    "Reference HVAC 2025": (_hvac_reference, 3),
    "Cold Climate ASHP": (_hvac_ashp, 4),
    "Air Sealing": (_air_sealing, 10),
    "Attic Insulation": (_attic_insulation, 11),
    "ENERGY STAR Windows": (_estar_windows, 15),
}


# ---------------------------------------------------------------------------
# matched-cell delta estimation
# ---------------------------------------------------------------------------
def _resolve_keys(df: pd.DataFrame) -> Dict[str, str]:
    """Map logical match keys -> actual column names (data_loader renames the
    canonical ResStock names to RECS-style names on load)."""
    out = {}
    for logical, aliases in RS_MATCH_KEY_ALIASES.items():
        for a in aliases:
            if a in df.columns:
                out[logical] = a
                break
    return out


def matched_cell_delta(base: pd.DataFrame, up: pd.DataFrame,
                       energy_col: str = "TOTAL_ENERGY_KWH",
                       min_cells: int = RS_CELL_MIN_CELLS) -> Dict:
    """Baseline-weighted delta over common (climate x type x fuel) cells.

    Returns {"delta_pct", "naive_pct", "n_cells", "coverage_pct", "note"}.
    ``delta_pct`` is None when too few cells overlap (the caller must then
    fall back to the naive estimate AND say so).
    """
    e_b = pd.to_numeric(base[energy_col], errors="coerce")
    e_u = pd.to_numeric(up[energy_col], errors="coerce")
    naive = float((1.0 - e_u.mean() / e_b.mean()) * 100) \
        if (e_b.mean() and np.isfinite(e_b.mean())) else np.nan
    coverage = float(len(up) / max(len(base), 1) * 100)

    kb = _resolve_keys(base)
    ku = _resolve_keys(up)
    common_keys = [k for k in ("ba_climate", "typehuq", "fuelheat") if k in kb and k in ku]
    note = (f"matched on cells: {common_keys}; upgrade file covers "
            f"{coverage:.0f}% of baseline rows")
    if len(common_keys) < 2:
        return {"delta_pct": None, "naive_pct": naive, "n_cells": 0,
                "coverage_pct": coverage,
                "note": note + "; too few match keys — naive estimate only"}

    gb = base.assign(_e=e_b.to_numpy()).groupby([kb[k] for k in common_keys],
                                                observed=True)
    gu = up.assign(_e=e_u.to_numpy()).groupby([ku[k] for k in common_keys],
                                              observed=True)
    m_b, n_b = gb["_e"].mean(), gb["_e"].size()
    m_u = gu["_e"].mean()
    common = m_b.dropna().index.intersection(m_u.dropna().index)
    if len(common) < min_cells:
        return {"delta_pct": None, "naive_pct": naive, "n_cells": int(len(common)),
                "coverage_pct": coverage,
                "note": note + f"; only {len(common)} common cells (< {min_cells})"}
    w = n_b.loc[common].to_numpy(dtype=float)
    delta = float((1.0 - np.average(m_u.loc[common].to_numpy(), weights=w)
                   / np.average(m_b.loc[common].to_numpy(), weights=w)) * 100)
    return {"delta_pct": delta, "naive_pct": naive, "n_cells": int(len(common)),
            "coverage_pct": coverage, "note": note}


def shared_support_delta(base: pd.DataFrame, up: pd.DataFrame,
                         energy_col: str = "TOTAL_ENERGY_KWH",
                         q_lo: float = 0.05, q_hi: float = 0.95,
                         min_cells: int = RS_CELL_MIN_CELLS) -> Dict:
    """Diagnosis-corrected delta for upgrades whose file covers a SELECTED
    subpopulation.

    Matching on observed stock characteristics cannot remove composition bias
    when the upgrade's eligibility depends on a variable the match keys do not
    carry (air sealing targets the leaky homes; leakiness is not a match key,
    so every cell still mixes leaky and tight homes). What does identify the
    effect is the intervention's own physical driver, which IS published in
    the upgrade file. This estimator therefore:

      1. reads the driver of the measure from the columns that DIFFER between
         the baseline and upgrade files (e.g. ``in.infiltration`` for air
         sealing, ``in.insulation_roof`` for attic insulation);
      2. restricts the baseline to the driver values the upgrade file actually
         contains (the diagnosed subpopulation);
      3. returns the baseline-weighted mean-ratio delta inside matched cells
         of that restricted baseline.

    ``delta_pct`` is None when no differing driver column is numeric-or-
    categorical-usable, in which case the caller falls back to the cell-matched
    estimate and must report the coverage caveat.
    """
    e_b = pd.to_numeric(base[energy_col], errors="coerce")
    e_u = pd.to_numeric(up[energy_col], errors="coerce")

    drivers = []
    for c in up.columns:
        if c == energy_col or c not in base.columns or c == "upgrade":
            continue
        if pd.api.types.is_numeric_dtype(up[c]) and pd.api.types.is_numeric_dtype(base[c]):
            ub = pd.to_numeric(up[c], errors="coerce")
            bb = pd.to_numeric(base[c], errors="coerce")
            # a measure that fixes a characteristic pins it to one value that
            # must already exist in the baseline, while the baseline still varies
            if ub.notna().sum() and bb.notna().sum() and ub.nunique() == 1 and \
               bb.nunique() > 1 and float(ub.dropna().iloc[0]) in set(
                   bb.dropna().unique().tolist()):
                drivers.append(c)
        else:
            su = set(map(str, up[c].dropna().unique()))
            sb = set(map(str, base[c].dropna().unique()))
            if su and su < sb:
                drivers.append(c)
        if len(drivers) >= 3:
            break
    logger.debug(f"shared_support_delta: drivers detected = {drivers}")
    if not drivers:
        return {"delta_pct": None, "drivers": [], "n_base_restricted": int(len(base)),
                "note": "no differing driver column found"}

    mask = pd.Series(True, index=base.index)
    for c in drivers:
        allowed = set(map(str, up[c].dropna().unique()))
        mask &= base[c].astype(str).isin(allowed)
    base_r = base.loc[mask]
    e_br = e_b.loc[mask]
    if len(base_r) < 200:
        return {"delta_pct": None, "drivers": drivers,
                "n_base_restricted": int(len(base_r)),
                "note": f"diagnosed subpopulation too small (n={len(base_r)})"}

    kb, ku = _resolve_keys(base_r), _resolve_keys(up)
    common_keys = [k for k in ("ba_climate", "typehuq", "fuelheat")
                   if k in kb and k in ku]
    if len(common_keys) < 2:
        delta = float((1.0 - e_u.mean() / e_br.mean()) * 100)
        return {"delta_pct": delta, "drivers": drivers,
                "n_base_restricted": int(len(base_r)), "n_cells": 0,
                "note": f"diagnosed on {drivers}; no match keys"}

    gb = base_r.assign(_e=e_br.to_numpy()).groupby([kb[k] for k in common_keys],
                                                   observed=True)
    gu = up.assign(_e=e_u.to_numpy()).groupby([ku[k] for k in common_keys],
                                              observed=True)
    m_b, n_b, m_u = gb["_e"].mean(), gb["_e"].size(), gu["_e"].mean()
    common = m_b.dropna().index.intersection(m_u.dropna().index)
    if len(common) < min(20, min_cells):
        delta = float((1.0 - e_u.mean() / e_br.mean()) * 100)
        return {"delta_pct": delta, "drivers": drivers,
                "n_base_restricted": int(len(base_r)), "n_cells": int(len(common)),
                "note": f"diagnosed on {drivers}; few cells"}
    w = n_b.loc[common].to_numpy(dtype=float)
    delta = float((1.0 - np.average(m_u.loc[common].to_numpy(), weights=w)
                   / np.average(m_b.loc[common].to_numpy(), weights=w)) * 100)
    return {"delta_pct": delta, "drivers": drivers,
            "n_base_restricted": int(len(base_r)), "n_cells": int(len(common)),
            "note": f"diagnosed on {drivers}; baseline restricted to "
                    f"{len(base_r):,}/{len(base):,} rows"}


def measure_resstock_deltas(resstock: Dict[int, pd.DataFrame]) -> Dict[int, Dict]:
    """Measured savings of each ResStock upgrade vs baseline.

    v8.1 returns a dict per upgrade:
        {"matched_pct", "naive_pct", "n_cells", "coverage_pct", "note"}
    so every downstream table can show BOTH estimators and the coverage that
    explains any gap between them. When a bldg_id exists, per-building matched
    deltas are used (order-independent, composition-free by construction).
    """
    out: Dict[int, Dict] = {}
    base = resstock.get(0)
    if base is None:
        return out
    key = "TOTAL_ENERGY_KWH" if "TOTAL_ENERGY_KWH" in base.columns else None
    if key is None:
        return out
    from config import RESSTOCK_COL_MAP
    id_aliases = RESSTOCK_COL_MAP.get("bldg_id", ["bldg_id"])
    base_id = next((a for a in id_aliases if a in base.columns), None)
    if base_id:
        base_map = pd.DataFrame({"_id": base[base_id].values,
                                 "_b": pd.to_numeric(base[key], errors="coerce").values})
    else:
        logger.warning("  measure_resstock_deltas: no bldg_id column — using "
                       "matched-cell population deltas (composition-corrected)")

    for u, df in resstock.items():
        if u == 0 or key not in df.columns:
            continue
        e = pd.to_numeric(df[key], errors="coerce")
        if base_id and any(a in df.columns for a in id_aliases):
            u_id = next(a for a in id_aliases if a in df.columns)
            m = base_map.merge(pd.DataFrame({"_id": df[u_id].values, "_u": e.values}),
                               on="_id", how="inner")
            b, uu = m["_b"], m["_u"]
            ok = b.notna() & uu.notna() & (b > 0) & (uu >= 0)
            if ok.sum() > 100:
                per = float(((b[ok] - uu[ok]) / b[ok]).mean() * 100)
                naive = float((1.0 - e.mean() / pd.to_numeric(base[key],
                             errors="coerce").mean()) * 100)
                out[u] = {"matched_pct": per, "naive_pct": naive,
                          "reported_pct": per, "n_cells": int(ok.sum()),
                          "coverage_pct": float(len(df) / len(base) * 100),
                          "note": f"per-building matched on {u_id} "
                                  f"(n={int(ok.sum()):,}); reported = paired "
                                  f"per-household mean, naive = population means"}
                logger.info(f"  ResStock delta upgrade {u}: paired={per:+.1f}% "
                            f"naive={naive:+.1f}% (per-building on {u_id}, "
                            f"n={int(ok.sum()):,}, coverage "
                            f"{len(df) / len(base) * 100:.0f}%)")
                continue
        res = matched_cell_delta(base, df, energy_col=key)
        rec = {"matched_pct": res["delta_pct"], "naive_pct": res["naive_pct"],
               "n_cells": res["n_cells"], "coverage_pct": res["coverage_pct"],
               "note": res["note"]}
        # subset upgrade files need the diagnosis-corrected estimator
        if res["coverage_pct"] < 95.0:
            ss = shared_support_delta(base, df, energy_col=key)
            rec["shared_support_pct"] = ss["delta_pct"]
            rec["shared_support_drivers"] = ss.get("drivers", [])
            if ss["delta_pct"] is not None:
                rec["note"] += " | " + ss["note"]
        best = (rec.get("shared_support_pct")
                if rec.get("shared_support_pct") is not None
                else rec["matched_pct"])
        rec["reported_pct"] = best if best is not None else rec["naive_pct"]
        out[u] = rec
        logger.info(f"  ResStock delta upgrade {u}: reported="
                    f"{rec['reported_pct']:.1f}% "
                    f"(matched={rec['matched_pct']}, naive={rec['naive_pct']:.1f}%; "
                    f"{rec['note']})")
        if (best is not None and rec["naive_pct"] is not None
                and np.sign(best) != np.sign(rec["naive_pct"])):
            logger.warning(f"    [COMPOSITION BIAS] upgrade {u}: naive population "
                           f"estimate ({rec['naive_pct']:+.1f}%) and the corrected "
                           f"estimate ({best:+.1f}%) disagree in SIGN — the upgrade "
                           f"file covers {res['coverage_pct']:.0f}% of the stock; "
                           "the table reports the corrected estimate and the note "
                           "explains the gap.")
    return out


def _delta_pct(deltas: Optional[Dict[int, Dict]], up_id: int) -> Optional[float]:
    if not deltas or up_id not in deltas:
        return None
    d = deltas[up_id]
    if isinstance(d, dict):
        return d.get("reported_pct",
                     d["matched_pct"] if d.get("matched_pct") is not None
                     else d.get("naive_pct"))
    return float(d)          # backward-compat with v8.0 float dicts


def _delta_note(deltas: Optional[Dict[int, Dict]], up_id: int) -> str:
    if not deltas or up_id not in deltas or not isinstance(deltas[up_id], dict):
        return ""
    return deltas[up_id].get("note", "")


def _delta_naive(deltas: Optional[Dict[int, Dict]], up_id: int) -> Optional[float]:
    return _delta_field(deltas, up_id, "naive_pct")


def _delta_field(deltas: Optional[Dict[int, Dict]], up_id: int,
                 field: str) -> Optional[float]:
    if not deltas or up_id not in deltas or not isinstance(deltas[up_id], dict):
        return None
    return deltas[up_id].get(field)


def teacher_counterfactual_table(tl, X_ref: pd.DataFrame,
                                 resstock_deltas: Optional[Dict[int, Dict]] = None
                                 ) -> pd.DataFrame:
    """Simulation-trained counterfactual transfer: per-upgrade teacher savings
    on the SAME households (the RECS test matrix expressed in the teacher's
    shared feature space)."""
    rows = []
    for name, (_, up_id) in SCENARIOS.items():
        sav = tl.upgrade_savings_pct(X_ref, up_id)
        if sav is None:
            rows.append({"scenario": name, "resstock_upgrade": up_id,
                         "teacher_mean_savings_pct": None,
                         "note": "delta teacher not pairable for this upgrade"})
            continue
        row = {"scenario": name, "resstock_upgrade": up_id,
               "teacher_mean_savings_pct": float(sav.mean()),
               "teacher_median_savings_pct": float(sav.median()),
               "teacher_p10_savings_pct": float(sav.quantile(0.10)),
               "teacher_p90_savings_pct": float(sav.quantile(0.90)),
               "note": ""}
        msg = (f"  [teacher] {name:<22}: mean {row['teacher_mean_savings_pct']:5.1f}% "
               f"(median {row['teacher_median_savings_pct']:5.1f}%, "
               f"p10 {row['teacher_p10_savings_pct']:5.1f}, "
               f"p90 {row['teacher_p90_savings_pct']:5.1f})")
        ref = _delta_pct(resstock_deltas, up_id)
        if ref is not None:
            row["resstock_delta_pct"] = ref
            row["resstock_delta_matched_pct"] = _delta_field(resstock_deltas, up_id, "matched_pct")
            row["resstock_delta_naive_pct"] = _delta_naive(resstock_deltas, up_id)
            row["gap_pp"] = row["teacher_mean_savings_pct"] - ref
            row["note"] = _delta_note(resstock_deltas, up_id)
            msg += f"  | measured {ref:5.1f}% (gap {row['gap_pp']:+.1f} pp)"
        logger.info(msg)
        rows.append(row)
    return pd.DataFrame(rows)


class DecisionSupport:
    """Counterfactual retrofit analysis with physics recompute."""

    def __init__(self, physics_engine, feature_engineer, build_frame_fn: Callable,
                 selected_features, transfer_learner=None):
        self.physics = physics_engine
        self.fe = feature_engineer
        self.build_frame = build_frame_fn
        self.selected = list(selected_features)
        self.tl = transfer_learner

    def _predict_total(self, model, raw_df: pd.DataFrame,
                       revert_raw_from: Optional[pd.DataFrame] = None) -> pd.Series:
        d = self.physics.compute_features(raw_df, verbose=False)
        if revert_raw_from is not None:
            for c in revert_raw_from.columns:
                if not c.startswith("PH_") and c in d.columns:
                    d[c] = revert_raw_from[c].values
        X = self.build_frame(d)
        Xt = self.fe.transform(X)
        Xt = Xt.reindex(columns=self.selected, fill_value=0.0)
        need_tl = [c for c in model.feature_names if c.startswith("TL_pred")]
        if need_tl and self.tl is not None and self.tl.fitted:
            Xt = self.tl.add_meta_features(Xt)
        cols = model.feature_names if model.feature_names else self.selected
        Xt = Xt.reindex(columns=cols, fill_value=0.0)
        return pd.Series(np.expm1(model.total_model.predict(Xt)), index=raw_df.index)

    def run_scenarios(self, model, raw_df: pd.DataFrame,
                      resstock_deltas: Optional[Dict[int, Dict]] = None,
                      max_rows: int = 1500, physics_only: bool = True) -> pd.DataFrame:
        base_raw = raw_df.iloc[:max_rows].copy()
        logger.info(f"Counterfactual scenarios on {len(base_raw):,} households "
                    f"(mode: {'physics-only' if physics_only else 'full raw-edit'})...")
        base_pred = self._predict_total(model, base_raw)
        rows = []
        for name, (edit_fn, up_id) in SCENARIOS.items():
            try:
                edited = edit_fn(base_raw)
                new_pred = self._predict_total(
                    model, edited,
                    revert_raw_from=base_raw if physics_only else None)
                sav = (base_pred - new_pred) / base_pred.clip(lower=1e-6) * 100.0
                row = {
                    "scenario": name,
                    "resstock_upgrade": up_id,
                    "mean_savings_pct": float(sav.mean()),
                    "median_savings_pct": float(sav.median()),
                    "p10_savings_pct": float(sav.quantile(0.10)),
                    "p90_savings_pct": float(sav.quantile(0.90)),
                    "note": "",
                }
                msg = (f"  {name:<24}: mean savings = {row['mean_savings_pct']:5.1f}%  "
                       f"(median {row['median_savings_pct']:5.1f}%)")
                ref = _delta_pct(resstock_deltas, up_id)
                if ref is not None:
                    row["resstock_delta_pct"] = ref
                    row["resstock_delta_matched_pct"] = _delta_field(resstock_deltas, up_id, "matched_pct")
                    row["resstock_delta_naive_pct"] = _delta_naive(resstock_deltas, up_id)
                    row["gap_pp"] = row["mean_savings_pct"] - ref
                    row["note"] = _delta_note(resstock_deltas, up_id)
                    msg += (f"\n    vs ResStock matched: {ref:5.1f}%  "
                            f"(delta {row['gap_pp']:+.1f} pp)")
                logger.info(msg)
                rows.append(row)
            except Exception as e:
                logger.warning(f"  Scenario '{name}' failed: {e}")
        return pd.DataFrame(rows)
