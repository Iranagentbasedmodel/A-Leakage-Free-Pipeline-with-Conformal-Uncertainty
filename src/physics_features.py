"""
Physics-Informed Features — v8.0
================================
Deterministic, row-local transforms derived from building heat-balance physics
(steady-state UA x degree-day model, ASHRAE fundamentals). Because every
feature depends only on the same row's survey attributes, computing them on
the full frame before the split is leakage-safe.

v8.0 additions vs v7.2:
  * PH_heating_eff / PH_cooling_eff / PH_dhw_eff / PH_ceiling_insul /
    PH_duct_leakage are actually created (v7 counterfactuals referenced
    non-existent columns -> silent no-op).
  * New interactions: HDD x sqft, CDD x sqft, vintage x HDD, log(sqft).
  * ``compute_features`` first drops existing PH_* columns, so it can be
    re-run after counterfactual edits of raw attributes (retrofit analysis).
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd

from config import SQFT_COL, HDD_COL, CDD_COL, YEAR_BUILT_COL

logger = logging.getLogger("physics_features")

PH_PREFIX = "PH_"

# --- lookup tables (engineering priors; documented in the report) -----------
WALL_U = {1: 0.30, 2: 0.18, 3: 0.25, 4: 0.20, 5: 0.22, 6: 0.35, 7: 0.28, 99: 0.22}
GLASS_U = {1: 1.00, 2: 0.55, 3: 0.35}           # single / double / triple
INSUL_FACTOR = {1: 1.25, 2: 1.00, 3: 0.80, 4: 0.65}   # ADQINSUL poor->good
DRAFTY_ACH50 = {1: 14.0, 2: 10.0, 3: 6.5, 4: 4.0}     # all/most/some/none
HEATING_EFF = {1: 0.80, 2: 0.80, 3: 0.90, 4: 0.90, 5: 2.30, 6: 0.78, 7: 0.60,
               8: 0.98, 9: 0.80, 10: 2.30, 11: 0.80, 12: 0.85, 13: 0.98}
# EQUIPM codes (RECS): 3/5 = central/warm-air furnace, 4=steam, 7=boiler?,
# 10 = heat pump (effective COP ~2.3), 12 = electric resistance (0.98) ...
FUEL_EFF_ADJ = {1: 0.80, 2: 0.75, 3: 0.78, 5: 0.98, 7: 0.55}
COOLING_EER = {1: 3.2, 2: 3.0, 3: 2.6}          # central / room / none-ish


class PhysicsFeatures:
    """Compute physics-informed features from raw survey attributes."""

    def __init__(self, ceiling_height_ft: float = 9.0):
        self.ceil_h = ceiling_height_ft

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def _num(df: pd.DataFrame, col: str, default: float) -> pd.Series:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").fillna(default)
        return pd.Series(default, index=df.index, dtype=float)

    @staticmethod
    def _map(df: pd.DataFrame, col: str, mapping: Dict, default: float) -> pd.Series:
        if col in df.columns:
            v = pd.to_numeric(df[col], errors="coerce")
            return v.map(mapping).fillna(default)
        return pd.Series(default, index=df.index, dtype=float)

    # -- main -----------------------------------------------------------------
    def compute_features(self, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        out = df.drop(columns=[c for c in df.columns if c.startswith(PH_PREFIX)],
                      errors="ignore").copy()
        if verbose:
            logger.info("Computing physics-informed features v8.0...")

        sqft = self._num(out, SQFT_COL, 2000.0).clip(lower=50)
        stories = self._num(out, "STORIES", 1.5).clip(lower=1, upper=6)
        hdd = self._num(out, HDD_COL, 4500.0).clip(lower=0)
        cdd = self._num(out, CDD_COL, 1200.0).clip(lower=0)
        occ = self._num(out, "NHSLDMEM", 2.5).clip(lower=1)
        year = self._num(out, YEAR_BUILT_COL, 1980.0)

        # ---- geometry
        height = stories * self.ceil_h
        footprint = sqft / stories
        perimeter = 4.0 * np.sqrt(footprint.clip(lower=25))
        wall_area = perimeter * height
        roof_area = footprint
        floor_area = footprint
        volume = sqft * self.ceil_h
        surface = wall_area + roof_area + floor_area

        out["PH_building_height"] = height
        out["PH_footprint_area"] = footprint
        out["PH_building_perimeter"] = perimeter
        out["PH_wall_area"] = wall_area
        out["PH_roof_area"] = roof_area
        out["PH_building_volume"] = volume
        out["PH_surface_area"] = surface
        out["PH_surface_volume_ratio"] = surface / volume.clip(lower=1)
        out["PH_surface_factor"] = surface / sqft
        out["PH_log_sqft"] = np.log1p(sqft)
        out["PH_age"] = (2020 - year).clip(lower=0)

        # ---- envelope U-values
        u_wall_base = self._map(out, "WALLTYPE", WALL_U, 0.22)
        insul_f = self._map(out, "ADQINSUL", INSUL_FACTOR, 1.0)
        u_wall = u_wall_base * insul_f
        u_roof = 0.08 * insul_f
        u_floor = pd.Series(0.10, index=out.index)
        u_glass = self._map(out, "TYPEGLASS", GLASS_U, 0.65)
        n_win = self._num(out, "WINDOWS", 5.0).clip(lower=1, upper=12)
        win_frac = (0.06 + 0.02 * n_win).clip(0.08, 0.30)
        win_area = wall_area * win_frac

        out["PH_U_wall"] = u_wall
        out["PH_U_roof"] = u_roof
        out["PH_U_window"] = u_glass
        out["PH_window_area"] = win_area
        out["PH_window_wall_ratio"] = win_frac
        out["PH_ceiling_insul"] = insul_f          # used by counterfactuals

        # ---- UA components (Btu/h/F)
        ua_wall = u_wall * (wall_area - win_area).clip(lower=0)
        ua_win = u_glass * win_area
        ua_roof = u_roof * roof_area
        ua_floor = u_floor * floor_area

        ach50 = self._map(out, "DRAFTY", DRAFTY_ACH50, 9.0)
        ach_nat = ach50 / 20.0
        ua_inf = 0.018 * volume * ach_nat
        duct_leak = ((year < 1990).astype(float) * 0.15 + 0.10) * (ua_wall + ua_roof)
        ua_total = ua_wall + ua_win + ua_roof + ua_floor + ua_inf

        out["PH_UA_wall"] = ua_wall
        out["PH_UA_window"] = ua_win
        out["PH_UA_roof"] = ua_roof
        out["PH_UA_floor"] = ua_floor
        out["PH_UA_infiltration"] = ua_inf
        out["PH_UA_total"] = ua_total
        out["PH_ACH_natural"] = ach_nat
        out["PH_duct_leakage"] = duct_leak         # used by counterfactuals

        # ---- equipment efficiencies (used by counterfactuals)
        fuelheat = self._num(out, "FUELHEAT", 1.0)
        equipm = self._num(out, "EQUIPM", 3.0)
        is_electric = fuelheat == 5
        # Heat-pump detection, RECS-2020 semantics:
        #   HEATPUMP==1 is the dedicated RECS flag; EQUIPM==3 is the RECS code for
        #   "heat pump" (EQUIPM==10 kept for backward-compat with the v8 test-data
        #   generator). Never treat a non-electric EQUIPM==3 home as a heat pump.
        hp_flag = self._num(out, "HEATPUMP", np.nan)
        is_hp = is_electric & (equipm.isin([3, 10]) | (hp_flag == 1))
        # equipment-age/type modulation of the fuel baseline efficiency
        equip_factor = equipm.map({3: 1.00, 4: 0.85, 5: 0.92, 7: 0.88,
                                   10: 1.00, 12: 1.00}).fillna(0.95)
        heat_eff = np.where(
            is_hp, 2.30,
            np.where(is_electric, 0.98,
                     fuelheat.map({1: 0.80, 2: 0.78, 3: 0.78, 7: 0.55}).fillna(0.80)
                     * equip_factor))
        heat_eff = pd.Series(heat_eff, index=out.index)
        aircond = self._num(out, "AIRCOND", 1.0)
        typeac = self._num(out, "TYPEAC", 2.0)
        cool_eff = typeac.map(COOLING_EER).fillna(2.8) * np.where(aircond == 1, 1.0, 0.0)
        dhw_eff = np.where(fuelheat == 5, 0.92, 0.62)

        out["PH_heating_eff"] = heat_eff.clip(lower=0.3)
        out["PH_cooling_eff"] = pd.Series(cool_eff, index=out.index)
        out["PH_dhw_eff"] = pd.Series(dhw_eff, index=out.index)
        out["PH_has_cooling"] = (aircond == 1).astype(float)

        # ---- thermal loads (kBtu, steady-state UA x DD x 24)
        heat_load = ua_total * hdd * 24.0 / 1000.0
        cool_load = (ua_total * 0.8 + ua_inf * 0.5) * cdd * 24.0 / 1000.0
        dhw_load = occ * 22.0 * 8.34 * 70.0 * 365.0 / 1000.0
        solar_gain = win_area * cdd * 0.30 / 1000.0
        base_load = (sqft * 3.5 + occ * 800.0) / 1000.0

        out["PH_heating_load"] = heat_load
        out["PH_cooling_load"] = cool_load
        out["PH_dhw_load"] = dhw_load
        out["PH_solar_gain"] = solar_gain
        out["PH_base_load"] = base_load

        # ---- interactions
        out["PH_HDD_x_UA"] = hdd * ua_total / 1000.0
        out["PH_CDD_x_UA"] = cdd * ua_total / 1000.0
        out["PH_HDD_x_sqft"] = hdd * sqft / 1000.0
        out["PH_CDD_x_sqft"] = cdd * sqft / 1000.0
        out["PH_HDD_x_CDD"] = hdd * cdd / 1000.0
        out["PH_vintage_x_HDD"] = out["PH_age"] * hdd / 1000.0
        out["PH_occ_x_sqft"] = occ * sqft / 1000.0

        # ---- aggregate physics scores
        out["PH_envelope_quality"] = 1.0 / (1.0 + ua_total / sqft.clip(lower=1) * 100.0)
        eff_h = out["PH_heating_eff"].clip(lower=0.3)
        eff_c = out["PH_cooling_eff"].clip(lower=0.1).replace(0, np.nan)
        est_total = (heat_load / eff_h + cool_load / eff_c.fillna(3.0).fillna(0) +
                     dhw_load / out["PH_dhw_eff"].clip(lower=0.3) + base_load)
        out["PH_eui_estimate"] = est_total / sqft
        out["PH_heating_share_est"] = (heat_load / eff_h) / est_total.clip(lower=1)
        out["PH_cooling_share_est"] = (cool_load / eff_c.fillna(3.0).fillna(0)) / est_total.clip(lower=1)
        out["PH_dhw_share_est"] = (dhw_load / out["PH_dhw_eff"].clip(lower=0.3)) / est_total.clip(lower=1)
        out["PH_efficiency_score"] = (out["PH_envelope_quality"] *
                                      eff_h.clip(upper=1.0) * insul_f.map({v: k for k, v in INSUL_FACTOR.items()})
                                      .fillna(2.0) / 4.0)

        n_new = sum(1 for c in out.columns if c.startswith(PH_PREFIX))
        if verbose:
            if "EUI" in out.columns:
                e = pd.to_numeric(out["EUI"], errors="coerce")
                p = out["PH_eui_estimate"]
                ok = e.notna() & p.notna()
                if ok.sum() > 30:
                    r = np.corrcoef(e[ok], p[ok])[0, 1]
                    logger.info(f"  PH_eui_estimate vs EUI correlation: {r:.4f}")
            logger.info(f"  Physics features created: {n_new}")
            for c in [c for c in out.columns if c.startswith(PH_PREFIX)][:10]:
                logger.info(f"    {c}: mean={out[c].mean():.4f}, valid={out[c].notna().sum():,}")
        return out
