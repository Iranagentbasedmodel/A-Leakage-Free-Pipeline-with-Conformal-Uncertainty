"""
Configuration Module — v8.0
============================
Central configuration for the Physics-Informed Energy-Conserving pipeline.
All paths are resolved relative to the project root (the folder that contains
``src/``), so the project is drop-in portable: on the user's machine the root
is e.g. ``D:\\...\\building_energy_pipeline``.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (auto-resolved from this file's location -> portable across systems)
# ---------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent
BASE_DIR = SRC_DIR.parent
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = BASE_DIR / "docs"
LOGS_DIR = BASE_DIR / "logs"
OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
MODELS_DIR = OUTPUTS_DIR / "models"
TABLES_DIR = OUTPUTS_DIR / "tables"
TOOLS_DIR = BASE_DIR / "tools"

for _d in (DATA_DIR, DOCS_DIR, LOGS_DIR, OUTPUTS_DIR, FIGURES_DIR, MODELS_DIR, TABLES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Data files
RECS_FILE = DATA_DIR / "recs2020_public_v7.csv"
TMY3_FILE = DATA_DIR / "tmy3.csv"
TMY3_META_FILE = DATA_DIR / "TMY3_StationsMeta.csv"
SPATIAL_LOOKUP_FILE = DATA_DIR / "spatial_tract_lookup_table.csv"
RECS_CODEBOOK = DATA_DIR / "RECS 2020 Codebook for Public File - v7.xlsx"
RESSTOCK_FILES = {u: DATA_DIR / f"upgrade{u}.csv.gz" for u in (0, 3, 4, 10, 11, 15)}

# ---------------------------------------------------------------------------
# Global settings
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
N_JOBS = max(1, min(8, (os.cpu_count() or 4)))
PIPELINE_VERSION = "8.1"
JOURNAL_TARGET = "Energy and Buildings (Q1, IF=8.0)"

# --- four-way split (v8.1) ---------------------------------------------------
# Reviewer finding #2: split-conformal calibration on the validation split was
# invalid because that same split drove early stopping, per-task routing and
# (optionally) Optuna tuning. Split conformal requires an exchangeable
# calibration sample that played NO role in fitting or selection. v8.1 holds
# out a dedicated calibration split; the validation split keeps its model-
# selection role. Sizes are fractions of the full data:
#   train 70 % | val 10 % | calib 5 % | test 15 %
TEST_SIZE = 0.15
VAL_SIZE = 0.10            # model selection (early stopping / routing / Optuna)
CALIB_SIZE = 0.05          # UNTOUCHED conformal calibration sample
MAX_FEATURES = 250

# --- multi-seed repetition (reviewer finding #3) -----------------------------
# Every headline comparison must carry seed-to-seed uncertainty. The pipeline
# retrains the compared models under an identical fixed-parameter protocol for
# each seed and reports mean +/- 95 % CI plus the paired difference.
SEED_LIST = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]
N_SEEDS_DEFAULT = 10
ENSEMBLE_SEEDS_DEFAULT = 3        # stacking is ~4x costlier per seed
EFFECT_THRESHOLD_PP = 0.10        # smallest TL gain treated as "measurable"

# --- leakage audit v8.1 (reviewer finding #6) --------------------------------
LINEAR_COMBO_MAX_FEATURES = 4     # greedy forward-OLS group size for test G
LINEAR_COMBO_R2_LIMIT = 0.95      # group R2 above this = suspicious composition
LINEAR_COMBO_MIN_SAMPLES = 200

# --- ResStock delta matching (reviewer finding #7) ---------------------------
# Matched-cell keys for population-level upgrade deltas when no bldg_id exists.
# Names are resolved through an alias map at call time because data_loader
# renames canonical ResStock columns to RECS-style names on load.
RS_MATCH_KEY_ALIASES = {
    "ba_climate": ["ba_climate", "BA_climate"],
    "typehuq": ["typehuq", "TYPEHUQ"],
    "fuelheat": ["fuelheat", "FUELHEAT"],
    "vintage": ["vintage"],
    "sqft": ["sqft", "TOTSQFT_EN"],
    "occupants": ["occupants", "NHSLDMEM"],
}
RS_CELL_MIN_CELLS = 100           # minimum common cells for a matched estimate.
                                  # The shipped climate x type x fuel space has
                                  # ~193 non-empty cells, so 200 was unreachable
                                  # and every matched estimate fell back to naive;
                                  # 100 cells x ~200 households/cell is stable.

# --- publication figures (reviewer finding: weak figures) ---------------------
FIG_FORMATS = ("pdf", "svg", "png")   # vector first; png for quick previews
FIG_DPI = 300
FIG_FONT_SIZE = 9.0

# ---------------------------------------------------------------------------
# RECS key columns
# ---------------------------------------------------------------------------
ID_COL = "DOEID"
WEIGHT_COL = "NWEIGHT"
SQFT_COL = "TOTSQFT_EN"           # conditioned square footage (legit feature)
STATE_COL = "state_postal"
CLIMATE_COL = "BA_climate"        # Building America climate zone (string)
CLIMATE_SIMPLE = "BA_climate_SIMPLE"
HDD_COL = "HDD65"                 # actual heating degree days (survey period)
CDD_COL = "CDD65"
HDD30_COL = "HDD30YR_PUB"
CDD30_COL = "CDD30YR_PUB"
YEAR_RANGE_COL = "YEARMADERANGE"  # vintage category -> YEAR_BUILT
YEAR_BUILT_COL = "YEAR_BUILT"

TARGET_TOTAL_COL = "TOTALBTU"     # total site energy, thousand Btu (kBtu)
TARGET_EUI_COL = "EUI"            # kBtu / sqft

ENDUSE_TASKS = ["heating", "cooling", "dhw", "lighting"]
ENDUSE_BTU_COLS = {               # site energy by end use (kBtu)
    "heating": "TOTALBTUSPH",     # space heating, all fuels
    "cooling": "BTUELCOL",        # space cooling (electric ~ all cooling)
    "dhw": "TOTALBTUWTH",         # water heating, all fuels
    "lighting": "BTUELLGT",       # lighting (electric)
}
ENDUSE_EUI_COLS = {t: f"EUI_{t.upper()}" for t in ENDUSE_TASKS}
ENDUSE_SHARE_COLS = {t: f"SHARE_{t.upper()}" for t in ENDUSE_TASKS}

# RECS missing-value codes (see RECS 2020 codebook)
MISSING_CODES = [-2, -3, -4, -5, -6, -7, -8, -9, -99, -999, 999999999]

# Leakage guard: any column whose name contains one of these tokens is an
# energy-consumption / cost / intensity column and must never be a feature.
LEAKAGE_PATTERNS = [
    "BTU", "KWH", "THERMS", "GALLON", "CUFEET", "DOL",
    "WOODAMT", "PELLETS", "EUI", "INTENSITY", "SHARE",
]
# Columns kept out of the feature matrix (id / weight are handled separately)
ALWAYS_EXCLUDE = [ID_COL, WEIGHT_COL]

# Vintage midpoints for YEARMADERANGE (RECS 2020 codebook)
YEARMADERANGE_MIDPOINTS = {
    1: 1935, 2: 1945, 3: 1955, 4: 1965, 5: 1975, 6: 1985,
    7: 1995, 8: 2005, 9: 2012, 10: 2017, 11: 2020,
}

# ---------------------------------------------------------------------------
# TMY3 (flexible name matching; preferred aliases below)
# ---------------------------------------------------------------------------
TMY3_STATION_ALIASES = ["station_number", "Station number", "USAF", "station"]
TMY3_DRYBULB_ALIASES = ["Dry-bulb (C)", "Dry-bulb(C)", "dry_bulb", "DryBulb", "Tdry"]
TMY3_GHI_ALIASES = ["GHI (W/m^2)", "GHI(W/m^2)", "GHI", "ghi"]
TMY3_RH_ALIASES = ["RHum (%)", "RHum(%)", "RH", "rhum"]
TMY3_WSPD_ALIASES = ["Wspd (m/s)", "Wspd(m/s)", "Wind Speed", "wspd"]
TMY3_DEW_ALIASES = ["Dew-point (C)", "Dew-point(C)", "dew_point", "Tdew"]
TMY3_DATE_ALIASES = ["Date (MM/DD/YYYY)", "Date", "date"]
TMY3_TIME_ALIASES = ["Time (HH:MM)", "Time", "time"]

TMY3_META_STATE_ALIASES = ["State", "state", "STATE"]
TMY3_META_CLASS_ALIASES = ["Class", "class", "Station Class", "CLASS"]
TMY3_META_LAT_ALIASES = ["Latitude", "Lat", "latitude", "LATITUDE"]
TMY3_META_LON_ALIASES = ["Longitude", "Lon", "longitude", "LONGITUDE"]
TMY3_META_ELEV_ALIASES = ["Elevation", "Elev", "elevation", "ELEVATION"]

# Station class weights (Class I = best solar data per TMY3 manual)
TMY3_CLASS_WEIGHTS = {"I": 1.0, "II": 0.7, "III": 0.4}

# ---------------------------------------------------------------------------
# ResStock 2025.1  (naming: "in.<char>" and "out.<metric>..<unit>")
# ---------------------------------------------------------------------------
RESSTOCK_COL_MAP = {
    "sqft": ["in.sqft..ft2", "in.sqft"],
    "typehuq": ["in.geometry_building_type_recs", "in.geometry_building_type"],
    "vintage": ["in.vintage"],
    "stories": ["in.geometry_stories"],
    "bedrooms": ["in.bedrooms"],
    "occupants": ["in.occupants"],
    "fuelheat": ["in.heating_fuel"],
    "hvac_heat_type": ["in.hvac_heating_type"],
    "hvac_heat_eff": ["in.hvac_heating_efficiency"],
    "hvac_cool_type": ["in.hvac_cooling_type"],
    "hvac_cool_eff": ["in.hvac_cooling_efficiency"],
    "insul_wall": ["in.insulation_wall"],
    "insul_roof": ["in.insulation_roof"],
    "insul_ceiling": ["in.insulation_ceiling"],
    "infiltration": ["in.infiltration"],
    "windows": ["in.windows"],
    "ba_climate": ["in.building_america_climate_zone"],
    "ashrae_zone": ["in.ashrae_iecc_climate_zone_2004"],
    "census_region": ["in.census_region"],
    "census_division": ["in.census_division"],
    "bldg_id": ["bldg_id", "in.bldg_id", "id", "building_id", "bldgid",
                "in.building_id", "sample_id"],      # building id for upgrade matching
    "state": ["in.state"],
    "county": ["in.county"],
    "puma": ["in.puma"],
    "weather_city": ["in.weather_file_city"],
    "wh_fuel": ["in.water_heater_fuel"],
    "wh_eff": ["in.water_heater_efficiency"],
    "has_pv": ["in.has_pv"],
    "ev": ["in.electric_vehicle_ownership"],
    "lighting": ["in.lighting"],
    "ceiling_fan": ["in.ceiling_fan"],
    "mech_vent": ["in.mechanical_ventilation"],
    "total_energy_kwh": ["out.site_energy.total.energy_consumption..kwh",
                         "out.site_energy.total.energy_consumption.kwh"],
    "elec_kwh": ["out.electricity.total.energy_consumption..kwh",
                 "out.electricity.total.energy_consumption.kwh"],
    "gas_kwh": ["out.natural_gas.total.energy_consumption..kwh",
                "out.natural_gas.total.energy_consumption.kwh"],
    "eui_total": ["out.site_energy.total.energy_consumption_intensity..kwh_per_ft2",
                  "out.site_energy.total.energy_consumption_intensity.kwh_per_ft2"],
    "bill_usd": ["out.utility_bills.total_bill..usd"],
    "co2e_kg": ["out.emissions.total.lrmer_mid_case_25..co2e_kg"],
    "upgrade": ["upgrade"],
    "upgrade_name": ["in.upgrade_name"],
    "weight": ["weight"],
}

# canonical ResStock name -> RECS-standard name (harmonization)
RESSTOCK_RECS_MAP = {
    "sqft": SQFT_COL,
    "typehuq": "TYPEHUQ",
    "stories": "STORIES",
    "bedrooms": "BEDROOMS",
    "occupants": "NHSLDMEM",
    "fuelheat": "FUELHEAT",
    "hvac_heat_type": "EQUIPM",
    "ba_climate": CLIMATE_COL,
    "state": STATE_COL,
    "weight": WEIGHT_COL,
    "eui_total": "EUI_TOTAL",
    "total_energy_kwh": "TOTAL_ENERGY_KWH",
}

# Building America climate zone -> representative degree days (ResStock has no
# survey HDD/CDD fields; coarse standard values, documented in the report).
BA_HDD_CDD = {
    "Very-Cold": (9000, 350), "Cold": (6500, 700), "Marine": (4200, 450),
    "Mixed-Humid": (4300, 1500), "Mixed-Dry": (4500, 1200),
    "Hot-Humid": (1600, 3200), "Hot-Dry": (1800, 3000), "Subarctic": (12000, 200),
}

# ResStock string -> RECS numeric code (best-effort harmonization)
RS_TO_RECS_CODES = {
    "TYPEHUQ": {
        "single-family detached": 2, "single-family attached": 3,
        "mobile home": 1, "multi-family with 2 - 4 units": 4,
        "multi-family with 5+ units": 5,
    },
    "FUELHEAT": {
        "electricity": 5, "natural gas": 1, "propane": 3,
        "fuel oil": 2, "wood": 7, "other fuel": 9, "none": 0,
    },
}

# ---------------------------------------------------------------------------
# Model hyper-parameters (CPU-friendly: i7-6700HQ, 4C/8T, 16GB RAM)
# ---------------------------------------------------------------------------
XGB_BASE_PARAMS = {
    "n_estimators": 2000,
    "learning_rate": 0.03,
    "max_depth": 7,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "n_jobs": N_JOBS,
    "random_state": RANDOM_SEED,
    "early_stopping_rounds": 100,
}

LGB_BASE_PARAMS = {
    "n_estimators": 2000,
    "learning_rate": 0.03,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_jobs": N_JOBS,
    "random_state": RANDOM_SEED,
    "verbose": -1,
}

QUANTILES = [0.1, 0.5, 0.9]
CONFORMAL_ALPHA = 0.10          # -> 90% prediction intervals
COVERAGE_TOL_PP = 3.0           # |observed - nominal| beyond this logs a
                                # warning: the guarantee is monitored, not
                                # merely assumed (finding #2)

# Performance targets (honest, literature-consistent on RECS survey data)
# Internal, aspirational engineering benchmarks inherited from v7. They are
# NOT journal acceptance criteria: published regression R2 on cross-sectional
# survey microdata (RECS) typically spans ~0.55-0.75, and the ResStock
# simulation teacher itself plateaus near R2(log)~0.74 on real files.
# Reaching 0.85 on RECS with survey-only inputs would itself be evidence of
# leakage (see leakage_audit checks B/E/G). Kept only as an internal gauge.
TARGET_R2_TOTAL = 0.85          # internal benchmark, total site energy
TARGET_R2_EUI = 0.70            # internal benchmark, energy intensity

# --------------------------------------------------------------------------
# Physics-derived monotonicity directions for the AGGREGATE total-energy
# model (log1p(TOTALBTU)). +1 = energy is non-decreasing in the feature,
# -1 = non-increasing. Only features whose direction is unambiguous from
# building physics are listed; everything else stays unconstrained (0).
#
# Scope note: these signs are valid for TOTAL energy only. End-use share /
# direct-task models are deliberately left unconstrained because the sign of
# e.g. HDD65 differs per end use (more HDD -> larger heating share but
# smaller cooling share).
#
# Names that do not appear in a given run's selected matrix are ignored,
# so the map is safe across feature-selection outcomes and data versions.
PHYSICS_MONOTONE_SIGNS = {
    # --- conditioned size / geometry: more envelope & volume -> more energy
    "TOTSQFT_EN": +1, "PH_log_sqft": +1, "PH_footprint_area": +1,
    "PH_wall_area": +1, "PH_roof_area": +1, "PH_building_volume": +1,
    "PH_surface_area": +1, "PH_building_height": +1,
    "PH_building_perimeter": +1, "PH_window_area": +1,
    # --- thermal conductance (U*A): leakier envelope -> more energy
    "PH_UA_total": +1, "PH_UA_wall": +1, "PH_UA_roof": +1,
    "PH_UA_window": +1, "PH_UA_floor": +1, "PH_UA_infiltration": +1,
    "PH_U_wall": +1, "PH_U_roof": +1, "PH_U_window": +1,
    "PH_ACH_natural": +1, "PH_duct_leakage": +1,
    # --- weather severity (both degree-day measures raise TOTAL energy)
    "HDD65": +1, "CDD65": +1, "TMY3_HDD65": +1, "TMY3_CDD65": +1,
    "PH_HDD_x_sqft": +1, "PH_CDD_x_sqft": +1, "PH_HDD_x_UA": +1,
    "PH_CDD_x_UA": +1, "PH_HDD_x_CDD": +1,
    # --- loads & occupancy
    "PH_heating_load": +1, "PH_cooling_load": +1, "PH_dhw_load": +1,
    "PH_base_load": +1, "PH_occ_x_sqft": +1, "PH_eui_estimate": +1,
    "NHSLDMEM": +1, "NUMADULTS": +1, "NUMCHILD": +1,
    # --- efficiency / insulation: better equipment & envelope -> less energy
    "PH_heating_eff": -1, "PH_cooling_eff": -1, "PH_dhw_eff": -1,
    "PH_ceiling_insul": -1, "PH_envelope_quality": -1,
    "PH_efficiency_score": -1,
}
