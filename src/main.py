"""
Main Pipeline — v8.1
====================
Physics-Informed Energy-Conserving pipeline for residential site-energy
prediction (RECS 2020 + TMY3 + ResStock 2025.1).

v8.1 integrates every protocol fix demanded in review:
  * FOUR-WAY split; split-conformal calibration on an untouched calib split
    (finding #2 — the coverage guarantee is now valid);
  * FAIR paired TL ablation under identical hyper-parameters, plus MULTI-SEED
    repetition with paired 95 % CIs and pre-registered verdicts for every
    headline comparison (findings #1 and #3 — the TL claim can now only be
    stated at the strength the data supports);
  * a RUN MANIFEST that exports the paper tables (one number per cell) from
    this single run (finding #4 — no more "0.721 0.730" cells, no more
    250-vs-252 or 90.0-vs-90.1 drift);
  * the hardened leakage audit with its JSON report, published pattern list,
    linear-combination test and real fitted-transformer verification
    (finding #6);
  * matched-cell ResStock deltas with side-by-side naive estimates and notes
    (finding #7 — the air-sealing sign flip is explained, not hidden);
  * honest LOGO with full per-fold refit and a protocol note (finding #8);
  * vector figures from a single style profile (figure findings).

Leakage-safe order of operations:
  load -> physics (row-local) -> SPLIT -> FE.fit(train) -> select(train)
  -> transfer (teacher on ResStock only) -> train -> ensemble -> evaluate
  -> seeds -> LOGO -> leakage audit -> counterfactuals -> manifest.

Run (from the project root):
    python src/main.py --fast
    python src/main.py                      # full protocol (10 seeds)
    python src/main.py --optimize --n-trials 40
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    BASE_DIR, LOGS_DIR, MODELS_DIR, TABLES_DIR, PIPELINE_VERSION, JOURNAL_TARGET,
    RANDOM_SEED, N_JOBS, MAX_FEATURES, SQFT_COL, WEIGHT_COL,
    TARGET_TOTAL_COL, TARGET_EUI_COL, ENDUSE_TASKS, ENDUSE_EUI_COLS,
    ENDUSE_SHARE_COLS, TARGET_R2_TOTAL, TARGET_R2_EUI,
    SEED_LIST, N_SEEDS_DEFAULT, ENSEMBLE_SEEDS_DEFAULT, CONFORMAL_ALPHA,
)
from data_loader import (load_recs, load_tmy3_meta, load_tmy3_hourly,
                         merge_tmy3_with_recs, load_resstock)
from physics_features import PhysicsFeatures
from feature_engineering import FeatureEngineer, build_feature_frame
from feature_selection import select_features
from preprocessing import stratified_split, climate_groups
from transfer_learning import TransferLearner, harmonize_resstock
from model_training import EnergyConservingModel
from ensemble_models import HonestStacking
from evaluation import (comprehensive_metrics, interval_coverage,
                        population_calibration, conformal_coverage_curve)
from leakage_audit import run_audit
from climate_validation import (leave_one_climate_out,
                                leave_one_climate_out_isolated)
from decision_support import (DecisionSupport, measure_resstock_deltas,
                              teacher_counterfactual_table)
from results_manifest import RunManifest

logger = logging.getLogger("MainPipeline")


def setup_logging(level: str = "INFO") -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"pipeline_v8_{stamp}.log"
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    handlers = [logging.StreamHandler(sys.stdout),
                logging.FileHandler(log_file, encoding="utf-8")]
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format=fmt, handlers=handlers, force=True)
    return log_file


def parse_args():
    p = argparse.ArgumentParser(description="Energy-Conserving Pipeline v8.1")
    p.add_argument("--recs-nrows", type=int, default=None)
    p.add_argument("--monotone", action="store_true",
                    help="apply physics-derived monotonicity constraints to "
                         "the aggregate total-energy model and the stacking "
                         "bases (domain priors only; cannot leak)")
    p.add_argument("--resstock-nrows", type=int, default=None)
    p.add_argument("--max-features", type=int, default=MAX_FEATURES)
    p.add_argument("--optimize", action="store_true")
    p.add_argument("--n-trials", type=int, default=60)
    p.add_argument("--optuna-timeout", type=int, default=30)
    p.add_argument("--skip-ensemble", action="store_true")
    p.add_argument("--skip-transfer", action="store_true")
    p.add_argument("--skip-counterfactual", action="store_true")
    p.add_argument("--skip-seeds", action="store_true",
                   help="skip the multi-seed repetition (NOT recommended: the "
                        "paper's comparison CIs come from it)")
    p.add_argument("--n-seeds", type=int, default=N_SEEDS_DEFAULT)
    p.add_argument("--ensemble-seeds", type=int, default=ENSEMBLE_SEEDS_DEFAULT)
    p.add_argument("--no-weights", action="store_true")
    p.add_argument("--fast", action="store_true", help="smoke-test capacity")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def _run_logo_workers(logo_csv: Path, args) -> pd.DataFrame:
    """LOGO via one fresh worker process per climate zone.

    Runs at pipeline START, while this parent is still ~300 MB. The workers
    rebuild RECS+TMY3+physics from disk, so they need nothing from the
    parent; conversely the parent (later resident at ~1 GB, with pymalloc
    arenas that malloc_trim cannot return) would OOM-kill any concurrent
    ~1 GB worker. Folds are independent, so per-zone processes produce the
    same numbers as a single-process LOGO with a lower peak each.
    """
    import subprocess
    worker = Path(__file__).resolve().parent / "logo_worker.py"
    zcmd = [sys.executable, str(worker), "--list-zones"]
    if args.recs_nrows:
        zcmd += ["--nrows", str(args.recs_nrows)]
    zr = subprocess.run(zcmd, capture_output=True, text=True)
    zones = zr.stdout.split()
    if zr.returncode != 0 or not zones:
        logger.error(f"  LOGO zone scan failed (rc={zr.returncode}); Table 4 "
                     f"will be empty. stderr: {zr.stderr[-300:]}")
        return pd.DataFrame()
    parts = []
    for zone in zones:
        zcmd = [sys.executable, str(worker), "--out", str(logo_csv),
                "--max-features", str(args.max_features),
                "--n-jobs", str(N_JOBS), "--zone", zone]
        if args.recs_nrows:
            zcmd += ["--nrows", str(args.recs_nrows)]
        proc = subprocess.run(zcmd, capture_output=True, text=True)
        if proc.returncode == 0 and logo_csv.exists():
            parts.append(pd.read_csv(logo_csv))
            logger.info(f"  LOGO worker OK for zone {zone}")
        else:
            logger.error(f"  LOGO worker FAILED for zone {zone} (rc="
                         f"{proc.returncode}); stderr tail: "
                         f"{proc.stderr[-400:] if proc.stderr else '-'}")
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(logo_csv, index=False)
    return out


def main() -> int:
    args = parse_args()
    log_file = setup_logging(args.log_level)
    t_start = time.time()
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest = RunManifest(PIPELINE_VERSION, stamp=run_stamp)

    n_seeds = 3 if args.fast else args.n_seeds
    ens_seeds = 2 if args.fast else args.ensemble_seeds

    logger.info("=" * 70)
    logger.info(f"  PHYSICS-INFORMED ENERGY-CONSERVING PIPELINE v{PIPELINE_VERSION}")
    logger.info(f"  Target: {JOURNAL_TARGET}")
    logger.info("=" * 70)
    logger.info(f"  RECS nrows: {args.recs_nrows or 'ALL'} | Max features: {args.max_features}")
    logger.info(f"  ResStock upgrades: [0, 3, 4, 10, 11, 15] | Transfer: {not args.skip_transfer}")
    logger.info(f"  Weights: {not args.no_weights} | Optimize: {args.optimize} | Fast: {args.fast}")
    logger.info(f"  Seeds: {n_seeds} (ensemble: {ens_seeds}) | N_JOBS: {N_JOBS}")
    logger.info("=" * 70)

    # ---------------------------------------------------------------- STEP 0
    # LOGO runs FIRST in one fresh worker per zone, while this process is
    # still small; results are reported later in STEP 10b (Table 4).
    logo_csv = TABLES_DIR / f"logo_climate_{run_stamp}.csv"
    logger.info("STEP 0: Leave-one-climate-out workers (fresh process per "
                "zone; results land in Table 4)...")
    logo_df = _run_logo_workers(logo_csv, args)
    if len(logo_df):
        logger.info(f"  LOGO done: mean R2 {logo_df['r2_log_total'].mean():.4f} "
                    f"| sum n_test = {logo_df['n_test'].sum():,} (full dataset)")

    # ---------------------------------------------------------------- STEP 1
    logger.info("STEP 1: Loading data...")
    df, weights = load_recs(nrows=args.recs_nrows)
    meta = load_tmy3_meta()
    station_agg = load_tmy3_hourly()
    df = merge_tmy3_with_recs(df, meta, station_agg)

    logger.info("STEP 1b: Loading ResStock...")
    resstock = load_resstock(upgrades=(0, 3, 4, 10, 11, 15),
                             nrows=args.resstock_nrows)

    # ---------------------------------------------------------------- STEP 2
    logger.info("STEP 2: Computing physics-informed features...")
    physics = PhysicsFeatures()
    df = physics.compute_features(df, verbose=True)

    try:
        from eda import run_eda
        run_eda(df, tag="recs")
    except Exception as e:
        logger.warning(f"EDA skipped ({e})")

    # ---------------------------------------------------------------- STEP 3
    logger.info("STEP 3: Four-way split (before any fitted preprocessing)...")
    splits = stratified_split(df)
    raw_train = splits["train"]
    raw_val = splits["val"]
    raw_calib = splits["calib"]
    raw_test = splits["test"]

    def _targets(raw):
        return FeatureEngineer.extract_targets(raw)

    tgt_train, tgt_val = _targets(raw_train), _targets(raw_val)
    tgt_calib, tgt_test = _targets(raw_calib), _targets(raw_test)
    manifest.update({"n_train": len(raw_train), "n_val": len(raw_val),
                     "n_calib": len(raw_calib), "n_test": len(raw_test)})

    # ---------------------------------------------------------------- STEP 4
    logger.info("STEP 4: Feature engineering (fit on train only)...")
    fe = FeatureEngineer()
    X_train = fe.fit_transform(build_feature_frame(raw_train))
    X_val = fe.transform(build_feature_frame(raw_val))
    X_calib = fe.transform(build_feature_frame(raw_calib))
    X_test = fe.transform(build_feature_frame(raw_test))
    logger.info(f"Final feature matrix: {X_train.shape[0]:,} x {X_train.shape[1]:,}")

    w_train = w_val = None
    if not args.no_weights and "_w" in tgt_train:
        w_train = tgt_train["_w"] / tgt_train["_w"].mean()
        w_val = tgt_val["_w"] / tgt_val["_w"].mean()
        logger.info(f"  Survey weights enabled (NWEIGHT, normalized). "
                    f"Train sum={w_train.sum():,.0f}")

    # ---------------------------------------------------------------- STEP 5
    logger.info("STEP 5: Feature selection (train only)...")
    y_total_train = tgt_train[TARGET_TOTAL_COL]
    y_log_train = np.log1p(y_total_train.clip(lower=0))
    selected = select_features(X_train, y_log_train, max_features=args.max_features)
    # float32 everywhere downstream: halves the resident set of every matrix
    # and every GBM histogram; tree splits are unaffected in practice
    X_train = X_train[selected].astype(np.float32)
    X_val = X_val[selected].astype(np.float32)
    X_calib = X_calib[selected].astype(np.float32)
    X_test = X_test[selected].astype(np.float32)
    n_selected = len(selected)
    # finding #4: report BOTH counts, consistently, everywhere:
    logger.info(f"  Feature matrix after selection: {n_selected} features "
                f"(+2 TL meta-features appended later = {n_selected + 2} final, "
                f"when transfer learning is active)")
    manifest.update({"n_features_selected": n_selected})

    # ---------------------------------------------------------------- STEP 6
    tl = TransferLearner()
    tl_used = False
    if not args.skip_transfer and 0 in resstock:
        logger.info("STEP 6: Transfer learning (ResStock teacher -> RECS meta-features)...")
        src = harmonize_resstock(resstock[0], physics=physics)
        numeric_candidates = [c for c in selected if c not in set(fe.cat_cols)]
        if tl.fit_teacher(src, candidate_features=numeric_candidates):
            sq = lambda X: pd.to_numeric(X[SQFT_COL], errors="coerce") if SQFT_COL in X.columns else None
            X_train = tl.add_meta_features(X_train, sq(X_train))
            X_val = tl.add_meta_features(X_val, sq(X_val))
            X_calib = tl.add_meta_features(X_calib, sq(X_calib))
            X_test = tl.add_meta_features(X_test, sq(X_test))
            tl_used = True
            manifest.set("teacher_metrics", tl.metrics)
    else:
        logger.info("STEP 6: Transfer learning skipped.")
    manifest.set("n_features_final", int(X_train.shape[1]))
    manifest.set("tl_used", bool(tl_used))

    # ---------------------------------------------------------------- STEP 7
    logger.info("STEP 7: Energy-conserving multi-task training...")
    shares_train = pd.DataFrame({c: tgt_train[c] for c in ENDUSE_SHARE_COLS.values()
                                 if c in tgt_train})
    shares_val = pd.DataFrame({c: tgt_val[c] for c in ENDUSE_SHARE_COLS.values()
                               if c in tgt_val})

    def _eui_dict(raw):
        d = {f"EUI_{t.upper()}": raw[f"EUI_{t.upper()}"] for t in ENDUSE_TASKS
             if f"EUI_{t.upper()}" in raw.columns}
        if TARGET_EUI_COL in raw.columns:
            d["AGGREGATE"] = raw[TARGET_EUI_COL]
        return d

    eui_train, eui_val = _eui_dict(raw_train), _eui_dict(raw_val)

    model = EnergyConservingModel(random_seed=RANDOM_SEED, n_jobs=N_JOBS,
                                  use_xgb=True, monotone=args.monotone)
    report = model.fit(X_train, y_total_train, shares_train,
                       X_val, tgt_val[TARGET_TOTAL_COL], shares_val,
                       eui_train=eui_train, eui_val=eui_val,
                       w_train=w_train, w_val=w_val,
                       X_calib=X_calib, y_total_calib=tgt_calib[TARGET_TOTAL_COL],
                       optimize=(args.optimize and not args.fast),
                       n_trials=args.n_trials, timeout_min=args.optuna_timeout)
    manifest.update({"conformal_mode": report.conformal_mode,
                     "conformal_q_log": report.conformal_q,
                     "conformal_n_calib": report.n_calib,
                     "routing": report.routing,
                     "task_val_scores": model.task_val_scores})
    if args.fast:
        logger.info("  [FAST mode] smoke-test capacity")

    # ---- FAIR paired ablation: identical hyper-parameters, minus TL columns
    ablation_r2 = None
    if tl_used:
        logger.info("  Ablation: re-training primary model WITHOUT TL meta-features "
                    "under the IDENTICAL fitted protocol (fixed params)...")
        cols_no_tl = [c for c in X_train.columns if not c.startswith("TL_pred")]
        primary_params = dict(model.total_model.get_params())
        primary_params = {k: v for k, v in primary_params.items()
                          if k not in ("n_jobs", "random_state", "objective",
                                       "early_stopping_rounds")}
        n_iter = getattr(model.total_model, "best_iteration", None)
        if n_iter:
            primary_params["n_estimators"] = int(n_iter)
        ab_model = EnergyConservingModel(random_seed=RANDOM_SEED, n_jobs=N_JOBS,
                                         use_xgb=True, monotone=args.monotone)
        ab_model.fit(X_train[cols_no_tl], y_total_train, shares_train,
                     X_val[cols_no_tl], tgt_val[TARGET_TOTAL_COL], shares_val,
                     w_train=w_train, w_val=w_val,
                     fixed_params=primary_params, optimize=False)
        pred_ab = ab_model.predict(X_test[cols_no_tl])
        m_ab = comprehensive_metrics(tgt_test[TARGET_TOTAL_COL].values,
                                     pred_ab["total_btu"], name="no-TL ablation")
        ablation_r2 = m_ab.get("r2")
        logger.info(f"  Ablation (no TL, identical protocol): test R2={ablation_r2:.4f}")
    manifest.set("primary_ablation_noTL_r2", ablation_r2)
    # The ablation model (~2k boosted trees) and the training targets are
    # dead after this point; freeing them here keeps the SHAP/figures stage
    # (and the later audit) off the OOM cliff on small-RAM machines.
    from climate_validation import _release_memory
    ab_model = None
    shares_train = shares_val = None
    eui_train = eui_val = None
    y_total_train = y_total_val = None
    _release_memory()

    # ---------------------------------------------------------------- STEP 7.5
    seed_stats = None
    if not args.skip_seeds:
        logger.info("STEP 7.5: Multi-seed repetition (headline comparison CIs)...")
        try:
            from seed_repeat import run_seed_repetition
            y_log_val = np.log1p(tgt_val[TARGET_TOTAL_COL].clip(lower=0))
            seed_stats = run_seed_repetition(
                X_train=X_train, y_log_train=y_log_train,
                X_val=X_val, y_log_val=y_log_val,
                X_test=X_test, y_test_total=tgt_test[TARGET_TOTAL_COL],
                w_train=w_train,
                seeds=SEED_LIST[:n_seeds],
                ensemble_seeds=ens_seeds,
                run_ensemble=not args.skip_ensemble)
            seed_stats["raw"].to_csv(TABLES_DIR / f"seed_repetition_{run_stamp}.csv")
            manifest.set("seed_summary", seed_stats["summary"])
            manifest.set("seed_comparisons", seed_stats["comparisons"])
        except Exception as e:
            logger.warning(f"  Seed repetition failed ({e})")

    # ---------------------------------------------------------------- STEP 8
    ensemble = None
    if not args.skip_ensemble:
        logger.info("STEP 8: Honest stacking ensemble...")
        ensemble = HonestStacking(seed=RANDOM_SEED, n_jobs=N_JOBS,
                                  monotone=args.monotone,
                                  n_folds=3 if args.fast else 5)
        y_log_val = np.log1p(tgt_val[TARGET_TOTAL_COL].clip(lower=0))
        ens_metrics = ensemble.fit(X_train, y_log_train, X_val, y_log_val,
                                   w_train=w_train)
        if ens_metrics:
            logger.info(f"  Ensemble trained: {ensemble.learners}")
            manifest.set("ensemble_meta_coef", ens_metrics.get("meta_coef"))
            manifest.set("ensemble_oof_scores", ens_metrics.get("oof_scores"))

    # ---------------------------------------------------------------- STEP 9
    logger.info("STEP 9: Evaluation on held-out TEST set...")
    pred = model.predict(X_test)
    y_test_total = tgt_test[TARGET_TOTAL_COL]
    w_test = tgt_test.get("_w")

    m_total = comprehensive_metrics(y_test_total.values, pred["total_btu"],
                                    weights=w_test, name="total_btu")
    intervals = model.predict_intervals(X_test)
    cov = interval_coverage(y_test_total.values,
                            intervals["q10"].values, intervals["q90"].values) \
        if len(intervals) else {}
    pop = population_calibration(y_test_total.values, pred["total_btu"], w_test) \
        if w_test is not None else {}

    # coverage curve (finding #2 evidence): calib residuals -> test coverage
    y_calib_log = np.log1p(pd.to_numeric(tgt_calib[TARGET_TOTAL_COL],
                                         errors="coerce").clip(lower=0))
    calib_resid = np.abs(y_calib_log.values
                         - model.total_model.predict(X_calib)) if model.conformal_q else []
    y_test_log = np.log1p(pd.to_numeric(y_test_total, errors="coerce").clip(lower=0))
    curve = conformal_coverage_curve(y_test_log.values,
                                     model.total_model.predict(X_test),
                                     calib_resid) if len(calib_resid) else {}

    logger.info(f"  total_btu   : R2={m_total.get('r2', np.nan):.4f} "
                f"RMSE={m_total.get('rmse', np.nan):,.0f} kBtu "
                f"cov90={cov.get('coverage', np.nan):.3f} "
                f"R2w={m_total.get('r2_weighted', np.nan):.4f}")
    logger.info(f"  conformal calibration: mode={model.conformal_mode} "
                f"(n_calib={report.n_calib:,}); test coverage curve: "
                + ", ".join(f"{int(k*100)}%->{v*100:.1f}%" for k, v in curve.items()))
    # the guarantee is monitored, not assumed: flag any level whose observed
    # coverage deviates from nominal by more than COVERAGE_TOL_PP
    from config import COVERAGE_TOL_PP
    for lvl, obs in curve.items():
        dev_pp = (obs - lvl) * 100
        if abs(dev_pp) > COVERAGE_TOL_PP:
            logger.warning(f"  [CONFORMAL] nominal {lvl*100:.0f}% interval covered "
                           f"{obs*100:.1f}% on test ({dev_pp:+.1f} pp, "
                           f"tolerance {COVERAGE_TOL_PP} pp; n_calib="
                           f"{report.n_calib:,}, n_test={len(y_test_log):,}) - "
                           "report the observed coverage alongside the claim")
    if pop:
        logger.info(f"  population calibration error: {pop['calibration_error_pct']:+.2f}%")

    m_eui = comprehensive_metrics(tgt_test[TARGET_EUI_COL].values, pred["eui"],
                                  name="eui_derived")
    logger.info(f"  eui derived : R2={m_eui.get('r2', np.nan):.4f} "
                f"RMSE={m_eui.get('rmse', np.nan):.2f} kBtu/sqft")
    m_eui_direct = {}
    if "eui_direct_ablation" in pred:
        m_eui_direct = comprehensive_metrics(tgt_test[TARGET_EUI_COL].values,
                                             pred["eui_direct_ablation"],
                                             name="eui_direct")
        logger.info(f"  eui direct  : R2={m_eui_direct.get('r2', np.nan):.4f}  "
                    f"(ablation, v7-style)")

    enduse_metrics = {}
    for task in ENDUSE_TASKS:
        col = ENDUSE_EUI_COLS[task]
        if col in tgt_test and f"eui_{task}" in pred:
            mm = comprehensive_metrics(tgt_test[col].values, pred[f"eui_{task}"],
                                       name=f"eui_{task}")
            enduse_metrics[task] = mm
            logger.info(f"  eui_{task:<8}: R2={mm.get('r2', np.nan):.4f} "
                        f"RMSE={mm.get('rmse', np.nan):.2f}")

    m_ens = {}
    if ensemble is not None and ensemble.is_trained:
        ens_log_pred = ensemble.predict(X_test)
        ens_total = np.expm1(ens_log_pred)
        m_ens = comprehensive_metrics(y_test_total.values, ens_total, name="ensemble")
        logger.info(f"  ensemble    : R2={m_ens.get('r2', np.nan):.4f} "
                    f"RMSE={m_ens.get('rmse', np.nan):,.0f} kBtu")

    # ---- ASHRAE G14 note (finding #5): the standard governs single-building
    # calibration against metered data; it does not apply to cross-sectional
    # survey prediction. NMBE/CV(RMSE) are reported descriptively only.
    g14_note = ("ASHRAE Guideline 14 applies to single-building calibration "
                "against metered data and is NOT applicable to cross-sectional "
                "survey prediction; NMBE/CV(RMSE) are reported descriptively.")

    # ---- Table 2 (manifest; one number per cell)
    manifest.add_row("table2_performance", {
        "target_model": "Total energy - primary multi-task",
        "r2": round(m_total.get("r2"), 4), "rmse_kbtu": round(m_total.get("rmse"), 0),
        "note": ""})
    if m_ens:
        manifest.add_row("table2_performance", {
            "target_model": "Total energy - stacking ensemble",
            "r2": round(m_ens.get("r2"), 4), "rmse_kbtu": round(m_ens.get("rmse"), 0),
            "note": ""})
    manifest.add_row("table2_performance", {
        "target_model": "EUI (derived)", "r2": round(m_eui.get("r2"), 4),
        "rmse_kbtu_per_sqft": round(m_eui.get("rmse"), 2), "note": ""})
    if m_eui_direct:
        manifest.add_row("table2_performance", {
            "target_model": "EUI (direct, ablation)",
            "r2": round(m_eui_direct.get("r2"), 4), "note": ""})
    for task, mm in enduse_metrics.items():
        manifest.add_row("table2_performance", {
            "target_model": f"{task.capitalize()} EUI", "r2": round(mm.get("r2"), 4),
            "rmse_kbtu_per_sqft": round(mm.get("rmse"), 2), "note": ""})

    manifest.update({
        "primary_test_r2_total": m_total.get("r2"),
        "primary_test_rmse_total": m_total.get("rmse"),
        "primary_test_r2_weighted": m_total.get("r2_weighted"),
        "primary_test_r2_eui": m_eui.get("r2"),
        "ensemble_test_r2_total": m_ens.get("r2") if m_ens else None,
        "ensemble_test_rmse_total": m_ens.get("rmse") if m_ens else None,
        "conformal_coverage_90_test": cov.get("coverage"),
        "conformal_coverage_90_reported_pct": round(cov.get("coverage") * 100, 1)
            if cov.get("coverage") is not None else None,
        "conformal_mean_width_kbtu": cov.get("mean_width"),
        "conformal_coverage_curve_test": {str(k): v for k, v in curve.items()},
        "population_calibration_error_pct": pop.get("calibration_error_pct") if pop else None,
        "nmbe_pct": m_ens.get("nmbiased_pct") if m_ens else m_total.get("nmbiased_pct"),
        "cvrmse_pct": m_ens.get("cvrmse_pct") if m_ens else m_total.get("cvrmse_pct"),
        "ashrae_g14_note": g14_note,
    })

    # ---------------------------------------------------------------- figures
    try:
        from publication_outputs import (prediction_figures, coverage_figure,
                                         architecture_schematic)
        ens_r2 = m_ens.get("r2") if m_ens else None
        ens_rmse = m_ens.get("rmse") if m_ens else None
        prediction_figures(y_test_total.values, pred["total_btu"].values,
                           tag="total_btu_test", unit="kBtu",
                           r2=m_total.get("r2"), rmse=m_total.get("rmse"),
                           n=len(y_test_total))
        if ens_r2 is not None:
            prediction_figures(y_test_total.values,
                               np.expm1(ensemble.predict(X_test)).values,
                               tag="ensemble_test", unit="kBtu",
                               r2=ens_r2, rmse=ens_rmse, n=len(y_test_total))
        coverage_figure(curve, tag="conformal",
                        mean_width=cov.get("mean_width"))
        architecture_schematic()
    except Exception as e:
        logger.warning(f"figures skipped ({e})")

    shap_payload = None
    try:
        from shap_analysis import run_shap
        # 800 rows is plenty for a stable top-10 ranking and keeps the
        # TreeExplainer footprint small on constrained machines
        shap_payload = run_shap(model, X_train, tag="total", max_rows=800)
        if shap_payload:
            manifest.set("shap_top10_single_run", shap_payload["values"])
    except Exception as e:
        logger.warning(f"shap skipped ({e})")

    # ---- persist + free (runs BEFORE the final-results block and audit)
    # By this point SHAP is done, so the fitted models, df and resstock are
    # either persistable or reloadable. On a 2 GB machine this is the
    # difference between finishing and an OOM kill right after SHAP
    # (measured: parent resident ~1.4 GB at this point, killed seconds
    # into the final block while every heavyweight was still held).
    model_pkl = MODELS_DIR / f"energy_conserving_model_v8_{run_stamp}.pkl"
    try:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model.save(model_pkl)
        if ensemble is not None and ensemble.is_trained:
            ensemble.save(MODELS_DIR / f"stacking_ensemble_v8_{run_stamp}.pkl")
        models_persisted = True
    except Exception as e:
        models_persisted = False
        logger.warning(f"  early model persistence failed ({e}); will retry at end")
    # Free EVERY heavyweight before the LOGO worker starts, not after: the
    # worker is a fresh ~1.2 GB process and must coexist with this parent on
    # a 2 GB machine (measured: parent 1.85 GB + worker 1.15 GB = OOM kill).
    # Each of these is either dead by now or reloadable from disk in seconds.
    df_columns = list(df.columns)
    dims = {"train": len(X_train), "val": len(X_val), "calib": len(X_calib),
            "test": len(X_test), "features_final": int(X_train.shape[1])}
    ab_model = None
    ensemble = None
    df = None
    resstock = None                       # reloaded after LOGO for step 12
    if models_persisted:
        model = None                      # reloaded from model_pkl after LOGO
    _release_memory()

    # ---- final results block (log)
    logger.info("=" * 60)
    logger.info("FINAL RESULTS (test set)")
    logger.info("=" * 60)
    rows = [{
        "metric": "total_btu (primary)", "r2": m_total.get("r2"),
        "target": TARGET_R2_TOTAL,
        "status": ("meets internal benchmark"
                   if m_total.get("r2", 0) >= TARGET_R2_TOTAL
                   else "below internal benchmark")},
        {"metric": "eui (derived)", "r2": m_eui.get("r2"), "target": TARGET_R2_EUI,
         "status": ("meets internal benchmark"
                    if m_eui.get("r2", 0) >= TARGET_R2_EUI
                    else "below internal benchmark")},
    ]
    for task, mm in enduse_metrics.items():
        rows.append({"metric": f"eui_{task}", "r2": mm.get("r2"),
                     "target": TARGET_R2_EUI,
                     "status": ("meets internal benchmark"
                                if mm.get("r2", 0) >= TARGET_R2_EUI
                                else "below internal benchmark")})
    if m_ens:
        rows.append({"metric": "total_btu (stacked ensemble)", "r2": m_ens.get("r2"),
                     "target": TARGET_R2_TOTAL,
                     "status": ("meets internal benchmark"
                                if m_ens.get("r2", 0) >= TARGET_R2_TOTAL
                                else "below internal benchmark")})
    for r in rows:
        logger.info(f"  {r['metric']:<32}: R2={r['r2']:.4f} "
                    f"(internal benchmark {r['target']}) [{r['status']}]")
    r2_note = (
        "The 0.85/0.70 figures are internal aspirational benchmarks inherited "
        "from v7, not acceptance criteria. Published regression R2 on "
        "cross-sectional survey microdata (RECS) typically spans ~0.55-0.75; "
        "the ResStock simulation teacher plateaus near R2(log)~0.74 on real "
        "files, which bounds what any student of these features can reach. "
        "R2=0.85 on RECS with survey-only inputs would itself be evidence of "
        "leakage (audit checks B/E/G exist precisely to catch that). Report "
        "the three honest views together: unweighted test R2, population-"
        "weighted R2, and log-space OOF/validation R2, each with its "
        "definition.")
    logger.info("  NOTE: " + r2_note)
    manifest.set("r2_context_note", r2_note)
    tables_final = pd.DataFrame(rows)

    # ---- Table 3 (baselines/ablations with seed CIs where available)
    def _ci(name):
        s = (seed_stats or {}).get("summary", {}).get(name)
        if not s or "mean" not in s:
            return None
        return f"{s['mean']:.4f} [{s['ci95_lo']:.4f}, {s['ci95_hi']:.4f}]"

    t3 = [
        ("XGBoost simple (no TL)", _ci("xgb_noTL"), "same features minus TL_*"),
        ("LightGBM simple (no TL)", _ci("lgb_noTL"), "same features minus TL_*"),
        ("XGBoost simple (+TL)", _ci("xgb_TL"), "full features"),
        ("LightGBM simple (+TL)", _ci("lgb_TL"), "full features"),
    ]
    for nm, v, note in t3:
        if v:
            mean = v.split(" ")[0]
            manifest.add_row("table3_models", {
                "model": nm, "r2_mean": float(mean),
                "r2_ci95": v[v.index("["):], "note": note})
    if ablation_r2 is not None:
        manifest.add_row("table3_models", {
            "model": "Primary multi-task (no TL)", "r2_mean": round(ablation_r2, 4),
            "r2_ci95": "", "note": "paired ablation, identical protocol"})
    manifest.add_row("table3_models", {
        "model": "Primary multi-task (with TL)",
        "r2_mean": round(m_total.get("r2"), 4), "r2_ci95": "",
        "note": "full architecture"})
    if m_ens:
        manifest.add_row("table3_models", {
            "model": "Honest stacking ensemble", "r2_mean": round(m_ens.get("r2"), 4),
            "r2_ci95": (_ci("ensemble") or "").split(" ", 1)[-1] if _ci("ensemble") else "",
            "note": "seed-mean shown in CI column source"})
    if seed_stats:
        for k, v in seed_stats["comparisons"].items():
            manifest.add_row("table3_models", {
                "model": f"comparison: {v['name']}",
                "r2_mean": round(v["mean_pp"] / 100.0, 5),
                "r2_ci95": f"[{v['ci95_lo_pp']:+.2f}, {v['ci95_hi_pp']:+.2f}] pp",
                "note": f"verdict={v['verdict']} (sign {v['sign_consistency']})"})

    # ---------------------------------------------------------------- STEP 10
    # LOGO refits the whole chain per fold, which needs headroom. raw_calib
    # and the splits dict are dead after step 4; dropping them (and letting
    # LOGO work in float32, see climate_validation) is the difference between
    # finishing and being OOM-killed on a 2 GB machine.
    from climate_validation import _release_memory
    raw_calib = None
    splits = None
    # Persist the fitted artefacts NOW, then drop the heavyweights that no
    # later step needs. The LOGO child fork shares this address space: every
    # MB freed here is a MB the child cannot push the machine over with
    # (observed: parent ~1 GB + LOGO fold peak = OOM kill on a 2 GB box).
    # (model persistence + heavyweight frees moved up, right after SHAP —
    #  see the "persist + free" block there)

    # ---------------------------------------------------------------- STEP 10a
    # The leakage audit runs BEFORE the LOGO worker on purpose: it is the
    # last consumer of the big training frames, so auditing first lets us
    # free them and keep this parent small while the worker runs.
    logger.info("STEP 10a: Leakage audit (v8.1, JSON report)...")
    audit = run_audit(
        {"train": X_train, "val": X_val, "calib": X_calib, "test": X_test},
        y_log_train,
        fe_fitted_on_train=fe.fitted,
        selection_on_train=True,
        test_scores={"total_btu": m_total.get("r2", 0), "eui": m_eui.get("r2", 0)},
        raw_columns=df_columns,
        guarded_columns=list(build_feature_frame(raw_train).columns),
        y_raw_train=tgt_train[TARGET_TOTAL_COL],
        fe=fe, raw_train=raw_train)
    manifest.update({"leakage_audit_passed": audit["passed"],
                     "leakage_audit_json": audit.get("json_path"),
                     "leakage_audit_checks": audit["checks"],
                     "leakage_forbidden_patterns": audit["forbidden_patterns"],
                     "leakage_removed_by_guard": audit["removed_by_guard"]})

    # Step 12 only needs raw_test, X_test, fe, physics and tl (+ model and
    # resstock reloaded after LOGO); everything else goes now.
    X_train = None
    X_val = None
    X_calib = None
    raw_train = None
    raw_val = None
    y_log_train = None
    _release_memory()

    # ---------------------------------------------------------------- STEP 10b
    logger.info("STEP 10b: LOGO results (computed in STEP 0 worker "
                "processes)...")
    if len(logo_df):
        logger.info(f"  LOGO table at {logo_csv} | mean R2: "
                    f"{logo_df['r2_log_total'].mean():.4f} | sum n_test = "
                    f"{logo_df['n_test'].sum():,} (full dataset)")

    if len(logo_df):
        for _, r in logo_df.iterrows():
            manifest.add_row("table4_logo", {
                "held_out_regime": r["zone"], "n_test": int(r["n_test"]),
                "n_train": int(r["n_train"]),
                "n_features_selected": int(r["n_features_selected"]),
                "r2_log_total": round(float(r["r2_log_total"]), 4)})
        manifest.update({"logo_mean_r2": float(logo_df["r2_log_total"].mean()),
                         "logo_min_r2": float(logo_df["r2_log_total"].min()),
                         "logo_n_test_sum": int(logo_df["n_test"].sum()),
                         "logo_protocol_note": logo_df["protocol_note"].iloc[0]})
        try:
            from publication_outputs import logo_figure
            logo_figure(logo_df)
        except Exception as e:
            logger.warning(f"logo figure skipped ({e})")

    # reload what step 12 needs now that the LOGO worker has exited
    resstock = load_resstock(upgrades=(0, 3, 4, 10, 11, 15),
                             nrows=args.resstock_nrows)
    if model is None:
        model = EnergyConservingModel.load(model_pkl)
        logger.info("  primary model reloaded from disk for counterfactuals")
    _release_memory()

    # ---------------------------------------------------------------- STEP 12
    cf_df = tcf = None
    if not args.skip_counterfactual:
        logger.info("STEP 12: Counterfactual retrofit analysis...")
        rs_deltas = measure_resstock_deltas(resstock)
        manifest.set("resstock_measured_deltas", rs_deltas)
        ds = DecisionSupport(physics, fe, build_feature_frame, selected,
                             transfer_learner=tl if tl_used else None)
        cf_df = ds.run_scenarios(model, raw_test, resstock_deltas=rs_deltas,
                                 max_rows=800 if args.fast else 1500)
        if len(cf_df):
            p = TABLES_DIR / f"counterfactual_{run_stamp}.csv"
            cf_df.to_csv(p, index=False)
            logger.info(f"  Counterfactual table saved: {p}")

        if tl_used and resstock and len(resstock) > 1:
            logger.info("STEP 12b: Simulation-derived savings (CATE delta teachers)...")
            tl.fit_delta_teachers(resstock, physics=physics)
            tcf = teacher_counterfactual_table(tl, X_test, resstock_deltas=rs_deltas)
            if len(tcf):
                p = TABLES_DIR / f"counterfactual_teacher_{run_stamp}.csv"
                tcf.to_csv(p, index=False)
                logger.info(f"  Teacher counterfactual table saved: {p}")

        # Table 5: student + teacher + BOTH ResStock estimators + note
        for src_df, layer in ((cf_df, "student"), (tcf, "teacher")):
            if src_df is None or not len(src_df):
                continue
            for _, r in src_df.iterrows():
                manifest.add_row("table5_counterfactual", {
                    "scenario": r["scenario"], "layer": layer,
                    "upgrade": int(r["resstock_upgrade"]),
                    "mean_savings_pct": (round(float(r[k]), 2) if pd.notna(
                        r[k := ("mean_savings_pct" if layer == "student"
                                else "teacher_mean_savings_pct")]) else None),
                    "resstock_matched_pct": (round(float(r["resstock_delta_matched_pct"]), 2)
                                             if "resstock_delta_matched_pct" in r
                                             and pd.notna(r.get("resstock_delta_matched_pct"))
                                             else None),
                    "resstock_naive_pct": (round(float(r["resstock_delta_naive_pct"]), 2)
                                           if "resstock_delta_naive_pct" in r
                                           and pd.notna(r.get("resstock_delta_naive_pct"))
                                           else None),
                    "note": r.get("note", "")})
        try:
            from publication_outputs import counterfactual_figure
            counterfactual_figure(cf_df, tcf)
        except Exception as e:
            logger.warning(f"counterfactual figure skipped ({e})")

    # ---------------------------------------------------------------- persist
    # (models were already persisted before the LOGO fork; this only runs if
    #  that early save failed)
    if not models_persisted:
        try:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            model.save(MODELS_DIR / f"energy_conserving_model_v8_{run_stamp}.pkl")
        except Exception as e:
            logger.warning(f"model persistence failed ({e})")

    summary = {
        "version": PIPELINE_VERSION,
        "run_stamp": run_stamp,
        "n_train": dims["train"], "n_val": dims["val"],
        "n_calib": dims["calib"], "n_test": dims["test"],
        "n_features_selected": n_selected,
        "n_features_final": dims["features_final"],
        "tl_used": tl_used,
        "tl_ablation_r2": ablation_r2,
        "primary_test_r2_total": m_total.get("r2"),
        "primary_test_r2_eui": m_eui.get("r2"),
        "ensemble_test_r2_total": m_ens.get("r2") if m_ens else None,
        "conformal_mode": model.conformal_mode,
        "conformal_coverage_90": cov.get("coverage"),
        "population_calibration_error_pct": pop.get("calibration_error_pct") if pop else None,
        "logo_mean_r2": float(logo_df["r2_log_total"].mean()) if len(logo_df) else None,
        "leakage_audit_passed": audit["passed"],
        "leakage_audit_json": audit.get("json_path"),
        "seed_comparisons": {k: v["verdict"] for k, v in
                             ((seed_stats or {}).get("comparisons") or {}).items()},
        "routing": model.task_routing,
        "task_val_scores": model.task_val_scores,
        "runtime_seconds": round(time.time() - t_start, 1),
    }
    with open(TABLES_DIR / f"run_summary_{run_stamp}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    tables_final.to_csv(TABLES_DIR / f"final_results_{run_stamp}.csv", index=False)
    manifest.set("final_results", rows)
    manifest.save()

    logger.info("=" * 70)
    logger.info(f"# PIPELINE COMPLETE in {summary['runtime_seconds']}s")
    logger.info(f"  Leakage audit: {'PASS' if audit['passed'] else 'FAIL'} | "
                f"Log: {log_file}")
    logger.info(f"  Manifest: {manifest.dir}")
    logger.info("=" * 70)
    return 0 if audit["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
