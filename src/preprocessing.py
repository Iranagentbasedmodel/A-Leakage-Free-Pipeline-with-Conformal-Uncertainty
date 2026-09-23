"""
Preprocessing / Splitting — v8.1
================================
The split happens BEFORE any fitted preprocessing (leakage-safe order):
  raw df -> stratified train/val/calib/test -> FE.fit(train) -> transform(...).

v8.1 (reviewer finding #2): FOUR-way split.
    train  70 %  model fitting (primary, shares, direct-EUI, quantiles, stack)
    val    10 %  model SELECTION only: early stopping, per-task routing, Optuna
    calib   5 %  UNTOUCHED split-conformal calibration sample. Nothing is
                 fitted, tuned or selected on it, so the residual quantile
                 computed there satisfies the split-conformal exchangeability
                 assumption and the test-set coverage claim is valid.
    test   15 %  reported metrics only.
In v8.0 conformal calibration shared the validation split with early stopping
and routing; that invalidates the finite-sample coverage guarantee, which is
precisely the claim a reviewer can (and did) attack.

Stratification is by the simplified Building America climate zone so that all
splits are climate-balanced; groups for leave-one-climate-out (LOGO)
validation are provided as well.
"""

import logging
from typing import Dict

import pandas as pd
from sklearn.model_selection import train_test_split

from config import (CLIMATE_SIMPLE, RANDOM_SEED, TEST_SIZE, VAL_SIZE, CALIB_SIZE)

logger = logging.getLogger("preprocessing")


def _strat(frame: pd.DataFrame):
    if CLIMATE_SIMPLE in frame.columns:
        strat = frame[CLIMATE_SIMPLE]
        if strat.value_counts().min() >= 10:
            return strat
    return None


def stratified_split(df: pd.DataFrame,
                     test_size: float = TEST_SIZE,
                     val_size: float = VAL_SIZE,
                     calib_size: float = CALIB_SIZE,
                     seed: int = RANDOM_SEED) -> Dict[str, pd.DataFrame]:
    """70/10/5/15 stratified split (train/val/calib/test).

    The calibration split is produced by the SAME stratified mechanism but is
    never used for fitting or selection anywhere in the pipeline — grep for
    ``X_calib`` to verify: it appears only in ``model.fit`` (conformal step)
    and ``main`` (slicing/logging).
    """
    holdout = test_size + val_size + calib_size
    train_df, temp_df = train_test_split(
        df, test_size=holdout, random_state=seed, stratify=_strat(df))

    # temp -> test | rest
    rel_test = test_size / holdout
    test_df, rest_df = train_test_split(
        temp_df, test_size=rel_test, random_state=seed, stratify=_strat(temp_df))

    # rest -> val | calib
    rel_calib = calib_size / (val_size + calib_size)
    val_df, calib_df = train_test_split(
        rest_df, test_size=rel_calib, random_state=seed, stratify=_strat(rest_df))

    logger.info(f"Split (stratified by {CLIMATE_SIMPLE}): "
                f"train={len(train_df):,} | val={len(val_df):,} | "
                f"calib={len(calib_df):,} | test={len(test_df):,}")
    logger.info("  Roles: val=model selection (early stopping/routing/Optuna); "
                "calib=UNTOUCHED conformal calibration; test=reporting only.")
    return {"train": train_df, "val": val_df,
            "calib": calib_df, "test": test_df}


def climate_groups(df: pd.DataFrame) -> pd.Series:
    """Group labels for leave-one-climate-out validation."""
    if CLIMATE_SIMPLE in df.columns:
        return df[CLIMATE_SIMPLE].astype(str)
    return pd.Series("ALL", index=df.index)
