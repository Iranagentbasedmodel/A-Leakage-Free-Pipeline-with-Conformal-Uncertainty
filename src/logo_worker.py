"""
LOGO out-of-process worker
==========================
Runs the leave-one-climate-out study in a FRESH Python process.

Why a separate process instead of a fork: by STEP 10 the pipeline parent is
resident at ~1 GB, and a forked child copy-on-writes hundreds of MB of the
parent's pandas pages into its own private set as soon as it touches them
(observed: forked LOGO child grew to 1.65 GB anon-rss and got OOM-killed on
a 2 GB machine). A fresh process that loads RECS + TMY3 + physics features
from disk peaks around 650 MB total, which coexists with the waiting parent.

The worker recomputes exactly the frame the in-process LOGO would have seen:
load_recs -> TMY3 merge -> physics features -> leave_one_climate_out, all
deterministic under the fixed RANDOM_SEED.

Usage (called by main.py, or directly):
    python src/logo_worker.py --out outputs/tables/logo_climate.csv \
        [--max-features 250] [--n-jobs 4]
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import CLIMATE_SIMPLE, MAX_FEATURES, N_JOBS  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="LOGO out-of-process worker")
    ap.add_argument("--out", default=None, help="CSV path for the LOGO table "
                    "(required unless --list-zones)")
    ap.add_argument("--max-features", type=int, default=MAX_FEATURES)
    ap.add_argument("--n-jobs", type=int, default=N_JOBS)
    ap.add_argument("--zone", action="append", default=None,
                    help="run only this climate zone (repeatable). One zone "
                         "per process resets the memory high-water between "
                         "folds; folds are independent, so the numbers are "
                         "identical to a single-process LOGO.")
    ap.add_argument("--nrows", type=int, default=None,
                    help="limit RECS rows (smoke tests only; paper runs use "
                         "the full sample)")
    ap.add_argument("--list-zones", action="store_true",
                    help="print the climate zones present in the sample and "
                         "exit (keeps the zone-derivation logic in one place)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")

    from data_loader import (load_recs, load_tmy3_meta, load_tmy3_hourly,
                             merge_tmy3_with_recs)

    if args.list_zones:
        df, _ = load_recs(nrows=args.nrows)
        for z in sorted(df[CLIMATE_SIMPLE].dropna().unique().tolist()):
            print(z)
        return 0
    if not args.out:
        ap.error("--out is required unless --list-zones")
    from physics_features import PhysicsFeatures
    from climate_validation import leave_one_climate_out

    df, _ = load_recs(nrows=args.nrows)
    meta, hourly = load_tmy3_meta(), load_tmy3_hourly()
    if meta is not None and hourly is not None:
        df = merge_tmy3_with_recs(df, meta, hourly)
    df = PhysicsFeatures().compute_features(df, verbose=False)

    out = leave_one_climate_out(df, df[CLIMATE_SIMPLE],
                                max_features=args.max_features,
                                n_jobs=args.n_jobs,
                                zones=args.zone)
    if out is None or not len(out):
        logging.getLogger("logo_worker").error("LOGO produced no folds")
        return 1
    out.to_csv(args.out, index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
