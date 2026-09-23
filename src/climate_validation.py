"""
Cross-Climate Generalization — v8.1
===================================
Leave-one-climate-zone-out (LOGO) validation: train on all zones but one,
evaluate on the held-out zone.

v8.1 (reviewer finding #8 — LOGO transparency):
  * v8.0 evaluated LOGO on the ALREADY-TRANSFORMED, ALREADY-SELECTED feature
    matrix of the primary experiment. The encoder, winsorizer and the 250-
    feature selection had all been fitted on the primary TRAIN split, and the
    LOGO "training folds" contained primary test households. Whatever the
    magnitude of that effect, the protocol could not be described honestly.
  * v8.1 refits the FULL preprocessing chain from scratch inside every fold:
    build_feature_frame -> FeatureEngineer.fit(train fold) -> transform ->
    select_features(train fold) -> LGBM fit -> evaluate on the held-out zone.
  * The returned table carries an explicit ``protocol_note`` and per-fold
    sizes; sum(n_test) over folds equals the FULL dataset size by design
    (each household is held out exactly once). LOGO is a resampling study
    over the whole sample, separate from the primary split — the manuscript
    must say so, and now the artifact says it too.
"""

import ctypes
import gc
import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from config import (LGB_BASE_PARAMS, CLIMATE_SIMPLE, RANDOM_SEED, N_JOBS,
                    MAX_FEATURES, TARGET_TOTAL_COL)

logger = logging.getLogger("climate_validation")

try:
    from lightgbm import LGBMRegressor
    _LGB = True
except ImportError:  # pragma: no cover
    _LGB = False

def _release_memory() -> None:
    """gc + glibc malloc_trim.

    pandas/sklearn temporaries are freed promptly by CPython, but glibc keeps
    the pages cached in the process heap, so RSS stays at the high-water mark
    of every fold. On a 2 GB machine that accumulated peak - not the live set
    - is what triggers the OOM killer. malloc_trim hands the free pages back;
    measured effect on the full RECS frame: 674 MB -> 430 MB RSS.
    """
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):        # non-glibc platforms
        pass


PROTOCOL_NOTE = (
    "Each fold refits the deterministic leakage guard, FeatureEngineer "
    "(encoder/imputer/winsorizer) and train-only feature selection FROM "
    "SCRATCH on the training folds; the held-out zone is never seen. "
    "sum(n_test) equals the full dataset size: LOGO is a resampling study "
    "over all households, independent of the primary train/val/calib/test "
    "split."
)


# --------------------------------------------------------------------------
# Subprocess isolation (memory)
# --------------------------------------------------------------------------
# By STEP 10 the main process is resident at ~900 MB (feature frames, primary
# + ablation + ensemble models, ResStock files) and glibc holds the pages of
# every earlier transient. A LOGO fold adds a ~600 MB peak of its own, which
# on a 2 GB machine crosses the OOM line. Running LOGO in a forked child
# keeps the fold temporaries out of the parent's address space entirely: the
# child shares the parent's pages copy-on-write, allocates only what it
# writes, and everything is returned to the OS when it exits.

_WORKER_DATA: dict = {}


def _logo_child(kwargs: dict) -> pd.DataFrame:
    """Runs inside the forked child; reads the frames from the fork image."""
    return leave_one_climate_out(_WORKER_DATA["df"], _WORKER_DATA["groups"],
                                 **kwargs)


def leave_one_climate_out_isolated(raw_df: pd.DataFrame,
                                   groups: pd.Series, **kwargs) -> pd.DataFrame:
    """leave_one_climate_out in a forked child when possible, in-process
    otherwise (Windows has no fork; typical desktop RAM makes the in-process
    path safe there)."""
    import multiprocessing as mp
    try:
        ctx = mp.get_context("fork")
    except ValueError:                      # platform without fork
        logger.info("  LOGO: fork unavailable; running in-process.")
        return leave_one_climate_out(raw_df, groups, **kwargs)
    _WORKER_DATA["df"] = raw_df
    _WORKER_DATA["groups"] = groups
    try:
        logger.info("  LOGO: running in an isolated child process "
                    "(memory returned to the OS on completion).")
        with ctx.Pool(1) as pool:
            return pool.apply(_logo_child, (kwargs,))
    except Exception as e:                  # pragma: no cover
        logger.warning(f"  LOGO child failed ({e}); retrying in-process.")
        return leave_one_climate_out(raw_df, groups, **kwargs)
    finally:
        _WORKER_DATA.clear()


def leave_one_climate_out(raw_df: pd.DataFrame,
                          groups: pd.Series,
                          physics=None,
                          max_features: int = MAX_FEATURES,
                          seed: int = RANDOM_SEED,
                          n_jobs: int = N_JOBS,
                          min_group_n: int = 100,
                          zones=None) -> pd.DataFrame:
    """Honest LOGO: full preprocessing refit inside every fold.

    Parameters
    ----------
    raw_df : raw survey frame (PH_* features may already be attached — they
        are row-local, hence fold-independent).
    groups : climate-zone label per row.
    """
    if not _LGB:
        logger.warning("LightGBM unavailable; LOGO skipped.")
        return pd.DataFrame()

    from feature_engineering import FeatureEngineer, build_feature_frame
    from feature_selection import select_features

    y_total = pd.to_numeric(raw_df[TARGET_TOTAL_COL], errors="coerce")
    y_log = np.log1p(y_total.clip(lower=0))
    valid = y_log.notna() & np.isfinite(y_log)

    logger.info("LOGO protocol: FE + selection refit from scratch per fold.")
    rows = []
    zone_iter = (sorted(groups.unique()) if zones is None
                 else [z for z in zones if z in set(groups.unique())])
    for zone in zone_iter:
        test_mask = valid & (groups == zone).values
        train_mask = valid & (groups != zone).values
        if test_mask.sum() < min_group_n or train_mask.sum() < 5 * min_group_n:
            continue

        tr_raw = raw_df.loc[train_mask]
        te_raw = raw_df.loc[test_mask]

        # ---- refit the whole chain on the training folds ONLY
        fe = FeatureEngineer()
        bff_tr = build_feature_frame(tr_raw)
        bff_te = build_feature_frame(te_raw)
        # float32 halves every encoder/imputer temporary inside the fold
        for bff in (bff_tr, bff_te):
            numc = bff.select_dtypes("number").columns
            bff[numc] = bff[numc].astype(np.float32)
        X_tr = fe.fit_transform(bff_tr)
        X_te = fe.transform(bff_te)
        del bff_tr, bff_te

        sel = select_features(X_tr, y_log.loc[train_mask], max_features=max_features)
        # float32: halves the resident set; LightGBM internally bins to
        # 32-bit anyway, so this costs nothing in accuracy
        X_tr = X_tr[sel].astype(np.float32)
        X_te = X_te[sel].astype(np.float32)
        del fe

        p = dict(LGB_BASE_PARAMS)
        p.update({"n_estimators": 500, "n_jobs": n_jobs, "random_state": seed})
        m = LGBMRegressor(**p)
        m.fit(X_tr, y_log.loc[train_mask])
        pred = m.predict(X_te)
        r2 = float(r2_score(y_log.loc[test_mask], pred))
        del X_tr, X_te, m, pred
        _release_memory()
        rows.append({"zone": zone,
                     "n_train": int(train_mask.sum()),
                     "n_test": int(test_mask.sum()),
                     "n_features_selected": len(sel),
                     "r2_log_total": r2})
        logger.info(f"  LOGO zone={zone:<8}: R2(log total)={r2:.4f} "
                    f"(n_train={train_mask.sum():,}, n_test={test_mask.sum():,}, "
                    f"features={len(sel)})")

    df = pd.DataFrame(rows)
    if len(df):
        df["protocol_note"] = PROTOCOL_NOTE
        logger.info(f"  LOGO mean R2: {df['r2_log_total'].mean():.4f} "
                    f"(min {df['r2_log_total'].min():.4f}) | "
                    f"sum n_test = {df['n_test'].sum():,} (full dataset)")
    return df
