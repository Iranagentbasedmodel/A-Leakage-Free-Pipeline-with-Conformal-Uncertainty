"""
Data Loader — v8.0
==================
Loads RECS 2020 microdata, TMY3 weather, TMY3 station metadata and ResStock
2025.1 upgrade files.

v8.0 fixes vs v7.2:
  * CRITICAL: TMY3 hourly merge joined ``Tmy3_Lat`` (latitude floats) against
    ``station_number`` -> 0 matches -> all 17 climate features were NaN.
    Now: hourly data are aggregated per station, stations are aggregated to
    STATE level (class-weighted) and merged to RECS via ``state_postal``.
  * The impossible tract-level spatial merge (RECS public file has no
    county/PUMA) is removed for RECS.
  * NWEIGHT is extracted as sample weights (kept out of the feature matrix).
  * End-use SHARE targets are created for the energy-conserving architecture.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from config import (
    RECS_FILE, TMY3_FILE, TMY3_META_FILE, RESSTOCK_FILES,
    ID_COL, WEIGHT_COL, SQFT_COL, STATE_COL, CLIMATE_COL, CLIMATE_SIMPLE,
    HDD_COL, CDD_COL, HDD30_COL, CDD30_COL,
    YEAR_RANGE_COL, YEAR_BUILT_COL, YEARMADERANGE_MIDPOINTS,
    TARGET_TOTAL_COL, TARGET_EUI_COL, ENDUSE_BTU_COLS, ENDUSE_EUI_COLS,
    ENDUSE_SHARE_COLS, MISSING_CODES,
    TMY3_STATION_ALIASES, TMY3_DRYBULB_ALIASES, TMY3_GHI_ALIASES,
    TMY3_RH_ALIASES, TMY3_WSPD_ALIASES, TMY3_DEW_ALIASES,
    TMY3_META_STATE_ALIASES, TMY3_META_CLASS_ALIASES,
    TMY3_META_LAT_ALIASES, TMY3_META_LON_ALIASES, TMY3_META_ELEV_ALIASES,
    TMY3_CLASS_WEIGHTS, RESSTOCK_COL_MAP, RESSTOCK_RECS_MAP,
)

logger = logging.getLogger("data_loader")


def _find_col(df: pd.DataFrame, aliases) -> Optional[str]:
    """Case-insensitive alias lookup."""
    low = {c.lower().strip(): c for c in df.columns}
    for a in aliases:
        if a in df.columns:
            return a
        if a.lower().strip() in low:
            return low[a.lower().strip()]
    return None


def _read_csv_smart(path, **kw) -> pd.DataFrame:
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False, **kw)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, low_memory=False, **kw)


# ---------------------------------------------------------------------------
# RECS 2020
# ---------------------------------------------------------------------------
def load_recs(nrows: Optional[int] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """Load RECS 2020 public microdata. Returns (df, weights)."""
    logger.info(f"Loading RECS 2020 from: {RECS_FILE}")
    df = _read_csv_smart(RECS_FILE, nrows=nrows)
    df.columns = [c.strip() for c in df.columns]
    logger.info(f"RECS loaded: {len(df):,} rows x {df.shape[1]:,} columns")

    # --- vintage midpoint
    if YEAR_RANGE_COL in df.columns:
        yr = pd.to_numeric(df[YEAR_RANGE_COL], errors="coerce")
        df[YEAR_BUILT_COL] = yr.map(YEARMADERANGE_MIDPOINTS)
        logger.info(f"Mapped {YEAR_BUILT_COL} for {df[YEAR_BUILT_COL].notna().sum():,} records")

    # --- missing codes -> NaN
    n_missing = 0
    num_cols = df.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        m = df[c].isin(MISSING_CODES)
        if m.any():
            n_missing += int(m.sum())
            df.loc[m, c] = np.nan
    logger.info(f"Replaced {n_missing:,} missing-code values with NaN")

    # --- EUI + end-use targets
    sqft = pd.to_numeric(df.get(SQFT_COL), errors="coerce").clip(lower=1)
    total = pd.to_numeric(df.get(TARGET_TOTAL_COL), errors="coerce")
    df[TARGET_EUI_COL] = total / sqft
    logger.info(f"EUI calculated for {df[TARGET_EUI_COL].notna().sum():,} records")

    for task, btu_col in ENDUSE_BTU_COLS.items():
        eui_col, share_col = ENDUSE_EUI_COLS[task], ENDUSE_SHARE_COLS[task]
        if btu_col in df.columns:
            btu = pd.to_numeric(df[btu_col], errors="coerce")
            df[eui_col] = btu / sqft
            df[share_col] = (btu / total).where(total > 0)
            logger.info(f"  {eui_col}: from {btu_col}, n={btu.notna().sum():,} "
                        f"(mean={df[eui_col].mean():.2f})")
            logger.info(f"  {share_col}: mean={df[share_col].mean():.3f}")
        else:
            df[eui_col] = np.nan
            df[share_col] = np.nan
            logger.warning(f"  {btu_col} not found; {eui_col} set to NaN")

    # --- simplified climate zone (for stratification / LOGO)
    if CLIMATE_COL in df.columns:
        def _simple(z):
            z = str(z).lower()
            if "cold" in z and "very" not in z:
                return "Cold"
            if "very" in z or "subarctic" in z:
                return "Cold"
            if "hot" in z:
                return "Hot"
            if "marine" in z:
                return "Mixed"
            if "mixed" in z:
                return "Mixed"
            return "Mixed"
        df[CLIMATE_SIMPLE] = df[CLIMATE_COL].map(_simple)
    else:
        df[CLIMATE_SIMPLE] = "Mixed"

    if HDD_COL in df.columns:
        logger.info(f"Added RECS climate features: HDD mean="
                    f"{pd.to_numeric(df[HDD_COL], errors='coerce').mean():.1f}, "
                    f"CDD mean={pd.to_numeric(df[CDD_COL], errors='coerce').mean():.1f}")

    # --- EUI validation report (sanity check vs EIA published stats)
    eui = df[TARGET_EUI_COL].dropna()
    logger.info("=" * 50)
    logger.info("EUI VALIDATION REPORT")
    logger.info(f"  Count: {len(eui):,} | Mean: {eui.mean():.2f} | Median: {eui.median():.2f}")
    logger.info(f"  1st %ile: {eui.quantile(0.01):.2f} | 99th %ile: {eui.quantile(0.99):.2f}")
    ok = 20 <= eui.median() <= 80
    logger.info(f"  [{'PASS' if ok else 'CHECK'}] Median EUI ({eui.median():.2f}) "
                f"{'within' if ok else 'outside'} expected range (~42)")
    logger.info("=" * 50)
    logger.info("RECS loading complete.")

    weights = pd.to_numeric(df.get(WEIGHT_COL), errors="coerce").fillna(1.0)
    return df, weights


# ---------------------------------------------------------------------------
# TMY3
# ---------------------------------------------------------------------------
def load_tmy3_meta() -> Optional[pd.DataFrame]:
    if not TMY3_META_FILE.exists():
        logger.warning(f"TMY3 metadata not found: {TMY3_META_FILE}")
        return None
    meta = _read_csv_smart(TMY3_META_FILE)
    st_col = _find_col(meta, TMY3_STATION_ALIASES)
    if st_col and st_col != "station_number":
        meta = meta.rename(columns={st_col: "station_number"})
    logger.info(f"TMY3 metadata loaded: {len(meta):,} stations")
    return meta


def load_tmy3_hourly() -> Optional[pd.DataFrame]:
    """Load hourly TMY3 and aggregate to per-station annual climate features.

    Returns one row per station_number with degree-days (base 18.3C ~ 65F),
    solar, humidity and wind statistics — the "17 climate features".
    """
    if not TMY3_FILE.exists():
        logger.warning(f"Hourly TMY3 not found: {TMY3_FILE}")
        return None
    logger.info(f"Loading hourly TMY3 from: {TMY3_FILE} (this may take a while)...")

    head = pd.read_csv(TMY3_FILE, nrows=5, low_memory=False)
    st_col = _find_col(head, TMY3_STATION_ALIASES)
    t_col = _find_col(head, TMY3_DRYBULB_ALIASES)
    ghi_col = _find_col(head, TMY3_GHI_ALIASES)
    rh_col = _find_col(head, TMY3_RH_ALIASES)
    wspd_col = _find_col(head, TMY3_WSPD_ALIASES)
    dew_col = _find_col(head, TMY3_DEW_ALIASES)
    if st_col is None or t_col is None:
        logger.error("TMY3 hourly: station/dry-bulb column not found.")
        return None

    usecols = [c for c in {st_col, t_col, ghi_col, rh_col, wspd_col, dew_col} if c]
    df = pd.read_csv(TMY3_FILE, usecols=usecols, low_memory=False)
    df = df.rename(columns={st_col: "station_number"})
    logger.info(f"TMY3 hourly loaded: {len(df):,} rows")

    t = pd.to_numeric(df[t_col], errors="coerce")
    base_c = 18.3  # 65F
    df["_hdd"] = (base_c - t).clip(lower=0) / 24.0   # degC-days per hour record
    df["_cdd"] = (t - base_c).clip(lower=0) / 24.0

    grp = df.groupby("station_number")
    g = pd.DataFrame({"TMY3_HDD65": grp["_hdd"].sum(),
                      "TMY3_CDD65": grp["_cdd"].sum()})
    for col, name in ((t_col, "TDB"), (ghi_col, "GHI"), (rh_col, "RH"), (wspd_col, "WSPD")):
        if col:
            df[f"_{name}"] = pd.to_numeric(df[col], errors="coerce")
            v = grp[f"_{name}"]
            g[f"TMY3_{name}_mean"] = v.mean()
            g[f"TMY3_{name}_std"] = v.std()
            g[f"TMY3_{name}_max"] = v.max()

    # extra physics-oriented aggregates
    g["TMY3_HDD_CDD_ratio"] = g["TMY3_HDD65"] / g["TMY3_CDD65"].clip(lower=1)
    g["TMY3_heating_months"] = (g["TMY3_HDD65"] / 30.0).clip(0, 12)
    g["TMY3_cooling_months"] = (g["TMY3_CDD65"] / 30.0).clip(0, 12)
    if "TMY3_GHI_mean" in g.columns:
        g["TMY3_solar_gain_index"] = g["TMY3_GHI_mean"] * g["TMY3_GHI_max"].clip(lower=1) / 1000.0
    g["TMY3_n_hours"] = df.groupby("station_number").size()

    g = g.reset_index()
    logger.info(f"TMY3 aggregated: {len(g):,} stations, {g.shape[1] - 1} climate features")
    return g


def merge_tmy3_with_recs(recs_df: pd.DataFrame,
                         meta: Optional[pd.DataFrame],
                         station_agg: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Attach state-level TMY3 climate features to RECS rows via state_postal.

    Aggregation is weighted by station class (I > II > III, per TMY3 manual).
    """
    df = recs_df.copy()
    if meta is None or station_agg is None or STATE_COL not in df.columns:
        logger.warning("TMY3 merge skipped (missing meta/hourly/state column).")
        return df

    m = meta.copy()
    state_col = _find_col(m, TMY3_META_STATE_ALIASES)
    class_col = _find_col(m, TMY3_META_CLASS_ALIASES)
    lat_col = _find_col(m, TMY3_META_LAT_ALIASES)
    lon_col = _find_col(m, TMY3_META_LON_ALIASES)
    elev_col = _find_col(m, TMY3_META_ELEV_ALIASES)
    if state_col is None:
        logger.warning("TMY3 meta has no State column; merge skipped.")
        return df

    keep = ["station_number", state_col]
    for c in (class_col, lat_col, lon_col, elev_col):
        if c:
            keep.append(c)
    m = m[keep].drop_duplicates("station_number")
    m = m.merge(station_agg, on="station_number", how="inner")
    logger.info(f"TMY3 meta+hourly: {len(m):,}/{len(meta):,} stations have hourly aggregates")

    m["_w"] = m[class_col].astype(str).str.strip().str[0].map(TMY3_CLASS_WEIGHTS).fillna(0.5) \
        if class_col else 0.5
    m["_w"] = m["_w"] * m.get("TMY3_n_hours", pd.Series(1.0, index=m.index)).clip(lower=1)

    feat_cols = [c for c in m.columns if c.startswith("TMY3_") and c != "TMY3_n_hours"]
    for c in (lat_col, lon_col, elev_col):
        if c:
            m[c] = pd.to_numeric(m[c], errors="coerce")
    geo = {lat_col: "TMY3ST_Lat", lon_col: "TMY3ST_Lon", elev_col: "TMY3ST_Elev"}

    def _wavg(g, col):
        v = pd.to_numeric(g[col], errors="coerce")
        w = g["_w"]
        ok = v.notna() & (w > 0)
        return np.average(v[ok], weights=w[ok]) if ok.any() else np.nan

    rows = {}
    for st, grp in m.groupby(state_col):
        r = {}
        for c in feat_cols:
            r[c] = _wavg(grp, c)
        for src, dst in geo.items():
            if src:
                r[dst] = _wavg(grp, src)
        r["TMY3ST_n_stations"] = len(grp)
        rows[st] = r
    state_feat = pd.DataFrame(rows).T.reset_index().rename(columns={"index": STATE_COL})
    state_feat[STATE_COL] = state_feat[STATE_COL].astype(str).str.strip().str.upper()

    df[STATE_COL] = df[STATE_COL].astype(str).str.strip().str.upper()
    n_feat = state_feat.shape[1] - 1
    df = df.merge(state_feat, on=STATE_COL, how="left")
    matched = df[feat_cols[0]].notna().mean() * 100 if feat_cols else 0
    logger.info(f"TMY3 merged by state. Added {n_feat} features; matched records: {matched:.1f}%")

    # validation: TMY3-derived HDD should correlate with the survey HDD
    if "TMY3_HDD65" in df.columns and HDD_COL in df.columns:
        a = pd.to_numeric(df["TMY3_HDD65"], errors="coerce")
        b = pd.to_numeric(df[HDD_COL], errors="coerce")
        ok = a.notna() & b.notna()
        if ok.sum() > 30:
            r = np.corrcoef(a[ok], b[ok])[0, 1]
            logger.info(f"  [VALIDATION] corr(TMY3_HDD65, RECS HDD65) = {r:.3f} (expect > 0.5)")
    return df


# ---------------------------------------------------------------------------
# ResStock 2025.1
# ---------------------------------------------------------------------------
def _resstock_rename(df: pd.DataFrame) -> pd.DataFrame:
    """Rename ResStock columns to canonical names, tolerating naming variants."""
    low = {c.lower(): c for c in df.columns}
    ren = {}
    for canon, aliases in RESSTOCK_COL_MAP.items():
        for a in aliases:
            if a in df.columns:
                ren[a] = canon
                break
            if a.lower() in low:
                ren[low[a.lower()]] = canon
                break
    return df.rename(columns=ren)


def load_resstock(upgrades=(0,), nrows: Optional[int] = None) -> Dict[int, pd.DataFrame]:
    """Load ResStock upgrade files. Returns {upgrade_id: df} with canonical
    names plus the RECS-harmonized aliases needed downstream."""
    out = {}
    for u in upgrades:
        path = RESSTOCK_FILES.get(u)
        if path is None or not path.exists():
            logger.warning(f"ResStock upgrade {u} not found; skipped.")
            continue
        logger.info(f"Loading ResStock upgrade {u} from: {path}")
        wanted = set()
        for aliases in RESSTOCK_COL_MAP.values():
            wanted.update(aliases)
        chunks = []
        reader = pd.read_csv(path, compression="gzip", chunksize=100_000, low_memory=False)
        got = 0
        for chunk in reader:
            chunk = _resstock_rename(chunk)
            canon = [c for c in chunk.columns if c in RESSTOCK_COL_MAP]
            chunks.append(chunk[canon])
            got += len(chunk)
            if nrows and got >= nrows:
                break
        if not chunks:
            logger.warning(f"  upgrade {u}: no mapped columns found.")
            continue
        df = pd.concat(chunks, ignore_index=True)
        if nrows:
            df = df.iloc[:nrows]
        logger.info(f"  Retaining {df.shape[1]} mapped columns per chunk")
        logger.info(f"  Loaded: {len(df):,} rows x {df.shape[1]} columns")

        # harmonized RECS-style aliases (used by transfer learning)
        ren = {c: RESSTOCK_RECS_MAP[c] for c in df.columns if c in RESSTOCK_RECS_MAP}
        df = df.rename(columns=ren)
        out[u] = df
    logger.info(f"ResStock loading complete: {len(out)}/{len(upgrades)} files loaded")
    return out
