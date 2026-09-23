"""
Feature Engineering — v8.0
==========================
Split-aware preprocessing: the split happens FIRST (see preprocessing.py);
``FeatureEngineer.fit`` sees ONLY the training frame; validation/test are
processed with ``transform`` (no refitting). This fixes v7.2 leakage where
the ordinal encoder and the +/-5-sigma clipper were fitted on the full data.

Also: NWEIGHT never enters the feature matrix — it is returned separately as
sample weights via ``extract_targets``.
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from config import (
    LEAKAGE_PATTERNS, ALWAYS_EXCLUDE, WEIGHT_COL,
    TARGET_TOTAL_COL, TARGET_EUI_COL, ENDUSE_EUI_COLS, ENDUSE_SHARE_COLS,
)

logger = logging.getLogger("feature_engineering")

MAX_CAT_CARDINALITY = 15     # numeric cols with <= this many levels -> categorical


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic leakage-guard: drop energy/cost/intensity columns and
    id/weight. No fitting -> safe to apply identically to every split."""
    drop = []
    for c in df.columns:
        cu = c.upper()
        if c in ALWAYS_EXCLUDE:
            drop.append(c)
            continue
        if any(p in cu for p in LEAKAGE_PATTERNS):
            drop.append(c)
    return df.drop(columns=list(dict.fromkeys(drop)), errors="ignore")


class FeatureEngineer:
    """Train-fitted encoder + imputer + winsorizer with schema alignment."""

    def __init__(self, clip_q: float = 0.001):
        self.clip_q = clip_q
        self.encoder: Optional[OrdinalEncoder] = None
        self.cat_cols: list = []
        self.num_cols: list = []
        self.medians: Optional[pd.Series] = None
        self.clip_lo: Optional[pd.Series] = None
        self.clip_hi: Optional[pd.Series] = None
        self.schema_: list = []
        self.fitted = False

    # ------------------------------------------------------------------ fit
    def fit(self, X: pd.DataFrame) -> "FeatureEngineer":
        logger.info("=" * 60)
        logger.info("FEATURE ENGINEERING v8.0 (fit on train only)")
        logger.info("=" * 60)
        X = X.copy()
        self.schema_ = list(X.columns)

        for c in X.columns:
            # Robust dtype detection: pandas extension dtypes (StringDtype,
            # Int64, ArrowDtype...) crash np.issubdtype — use the pandas API.
            dt = X[c].dtype
            try:
                is_num = pd.api.types.is_numeric_dtype(dt)
            except TypeError:
                is_num = False
            if (not is_num) or X[c].nunique(dropna=True) <= MAX_CAT_CARDINALITY:
                self.cat_cols.append(c)
            else:
                self.num_cols.append(c)
        logger.info(f"Categorical: {len(self.cat_cols)}, Numeric: {len(self.num_cols)}")

        if self.cat_cols:
            cat = X[self.cat_cols].fillna("MISSING").astype(str)
            self.encoder = OrdinalEncoder(handle_unknown="use_encoded_value",
                                          unknown_value=-1,
                                          encoded_missing_value=-1)
            self.encoder.fit(cat)
            logger.info(f"Ordinal encoder fitted on {len(self.cat_cols)} categorical "
                        f"columns (train only)")

        if self.num_cols:
            num = X[self.num_cols].apply(pd.to_numeric, errors="coerce")
            self.medians = num.median().fillna(0.0)
            self.clip_lo = num.quantile(self.clip_q)
            self.clip_hi = num.quantile(1 - self.clip_q)
            # keep bounds finite; fall back to median when a bound is NaN
            self.clip_lo = self.clip_lo.fillna(self.medians)
            self.clip_hi = self.clip_hi.fillna(self.medians)

        self.fitted = True
        return self

    # ------------------------------------------------------------- transform
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        assert self.fitted, "FeatureEngineer must be fitted before transform."
        missing = [c for c in self.schema_ if c not in X.columns]
        extra = [c for c in X.columns if c not in self.schema_]
        if missing or extra:
            logger.warning(f"Schema alignment: +{len(missing)} missing filled, "
                           f"-{len(extra)} extra dropped "
                           f"(e.g. {missing[:3]}{'...' if len(missing) > 3 else ''})")
        Z = X.reindex(columns=self.schema_).copy()

        if self.cat_cols:
            cat = Z[self.cat_cols].fillna("MISSING").astype(str)
            Z[self.cat_cols] = self.encoder.transform(cat)
        if self.num_cols:
            num = Z[self.num_cols].apply(pd.to_numeric, errors="coerce")
            num = num.clip(lower=self.clip_lo, upper=self.clip_hi, axis=1)
            Z[self.num_cols] = num.fillna(self.medians)
        return Z.astype(np.float32)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)

    # --------------------------------------------------------------- targets
    @staticmethod
    def extract_targets(df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Targets + weights (weights are NOT features)."""
        t: Dict[str, pd.Series] = {}
        if TARGET_TOTAL_COL in df.columns:
            t[TARGET_TOTAL_COL] = pd.to_numeric(df[TARGET_TOTAL_COL], errors="coerce")
        if TARGET_EUI_COL in df.columns:
            t[TARGET_EUI_COL] = pd.to_numeric(df[TARGET_EUI_COL], errors="coerce")
        for col in list(ENDUSE_EUI_COLS.values()) + list(ENDUSE_SHARE_COLS.values()):
            if col in df.columns:
                t[col] = pd.to_numeric(df[col], errors="coerce")
        if WEIGHT_COL in df.columns:
            t["_w"] = pd.to_numeric(df[WEIGHT_COL], errors="coerce").fillna(1.0)
        return t
