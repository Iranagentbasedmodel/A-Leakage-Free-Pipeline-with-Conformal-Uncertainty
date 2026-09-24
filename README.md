# Building Energy Pipeline v8.1
### Residential energy-consumption prediction pipeline (RECS 2020 + TMY3 + ResStock)

Debugged and end-to-end-tested version 8 of the building-energy prediction
pipeline developed for a manuscript targeting the journal *Energy and Buildings*.

---

## 1. Repository layout

```
building_energy_pipeline/
├── src/                     # pipeline source code
│   ├── main.py              # entry point — runs the 12 stages (STEP 1..12)
│   ├── config.py            # settings and paths (auto-resolved from __file__)
│   ├── data_loader.py       # RECS / TMY3 / ResStock ingestion
│   ├── physics_features.py  # 47 physics-informed features (PH_*)
│   ├── feature_engineering.py
│   ├── feature_selection.py
│   ├── preprocessing.py
│   ├── transfer_learning.py # ResStock -> RECS knowledge distillation
│   ├── model_training.py    # energy-conserving architecture + adaptive routing
│   ├── ensemble_models.py   # honest stacking (OOF meta-features)
│   ├── evaluation.py
│   ├── leakage_audit.py     # automated data-leakage audit (tests A..F)
│   ├── climate_validation.py# leave-one-climate-out (LOGO) generalisation
│   ├── decision_support.py  # counterfactual retrofit scenarios
│   └── eda.py / publication_outputs.py / shap_analysis.py / model_selection.py
├── tools/
│   └── make_test_data.py    # synthetic test-data generator (structure-identical)
├── data/                    # <- place the raw data files here
├── logs/  outputs/  docs/
└── requirements.txt
```

> **Paths:** all paths are resolved automatically relative to `config.py`, so the
> folder can be placed anywhere (e.g. `D:\projects\building_energy_pipeline\`)
> without editing the code.

---

## 2. Installation (Windows)

```bat
cd D:\projects\building_energy_pipeline
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

## 3. Input data

Place the following files in `data\`:

| File | Required | Description |
|---|---|---|
| `recs2020_public_v7.csv` | yes | RECS 2020 microdata |
| `tmy3.csv` | yes | TMY3 hourly weather |
| `TMY3_StationsMeta.csv` | yes | TMY3 station metadata |
| `upgrade00.csv.gz` ... `upgrade15.csv.gz` | recommended | ResStock (baseline + upgrade scenarios) for transfer learning and scenario validation |
| `RECS 2020 Codebook ...xlsx` | optional | used by helper tooling only |
| `spatial_tract_lookup_table.csv` | optional | — |

Without ResStock files the pipeline still runs (transfer learning and
counterfactual validation are skipped automatically).

---

## 4. Running

### Quick smoke test (a few minutes on CPU)
```bat
venv\Scripts\python.exe src\main.py --fast --log-level INFO
```

### Full run on the real data (with hyperparameter optimisation)
```bat
venv\Scripts\python.exe src\main.py --optimize --n-trials 40 --log-level INFO
```

### Useful options
| Flag | Purpose |
|---|---|
| `--fast` | lighter models for a quick end-to-end check |
| `--recs-nrows N` | read only N RECS rows (limited-memory testing) |
| `--resstock-nrows N` | cap ResStock rows |
| `--max-features N` | feature budget (default 250) |
| `--optimize --n-trials T` | Optuna with T trials |
| `--skip-ensemble` / `--skip-transfer` / `--skip-counterfactual` | skip heavy stages |
| `--no-weights` | disable NWEIGHT sample weights |

---

## 5. Synthetic test data (no downloads needed)

```bat
venv\Scripts\python.exe tools\make_test_data.py --recs-rows 4000 --resstock-rows 8000 --tmy3-stations 12
```
then run `src\main.py --fast`. The synthetic data matches the real files in
column names, ranges and structure and contains realistic behavioural noise, so
R² values on it are a conservative lower bound.

---

## 6. Outputs

- `logs/pipeline_*.log` — full run log
- `outputs/tables/` — metrics, routing results, LOGO, leakage audit, counterfactuals
- `outputs/figures/` — publication-ready figures
- `outputs/models/` — persisted models
- `outputs/run_summary.json` — machine-readable run summary

---

## 7. Methodological highlights (v8.0)

1. **Energy-conserving multi-task architecture:** a primary model predicts
   `log(total site energy)`; four share models predict end-use fractions,
   clipped and renormalised so the shares sum to at most one (remainder =
   "other"); EUI is derived as total/sqft.
2. **Adaptive per-task routing:** for every end use both the share-derived and
   the direct-EUI route are trained; the better route is selected on validation R².
3. **Leakage-safe ordering:** physics features (row-local) -> split -> feature
   engineering fitted on train only -> feature selection on train only.
4. **Leakage-free transfer learning (knowledge distillation):** a LightGBM
   teacher trained on harmonised ResStock simulations contributes meta-features;
   the teacher never sees RECS targets.
5. **Honest stacking:** K-fold out-of-fold meta-features + RidgeCV with
   non-negativity-clipped coefficients; base learners retrained on 100 % of train.
6. **Calibrated uncertainty:** XGBoost quantile regression (q10/q50/q90) +
   split-conformal calibration on the validation split.
7. **Automated six-test leakage audit (A–F)** on every run.
8. **Leave-one-climate-out (LOGO)** generalisation test.
9. **Counterfactual retrofit decision support:** raw-attribute edits with full
   physics re-computation, externally validated against the measured deltas of
   the matching ResStock upgrade simulations (buildings matched on `bldg_id`).

---

## 8. System requirements

- Python 3.10+, CPU-only is sufficient (tested on i7-6700HQ / 16 GB RAM)
- `--fast` run: ~5 minutes; full run with Optuna: ~45-60 minutes
