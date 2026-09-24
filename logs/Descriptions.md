 Physics-Informed Energy-Conserving Pipeline - Capability & Option Map

> Source log: `pipeline_v8.1.log`  
> Focus: Coding capabilities, architecture, and configurable options, not numeric outcomes.  
> Pipeline version: `v8.1` 

---

## 0. High-Level Capability Map

```text
+---------------------------------------------------------------------+
| DATA LAYER                                                          |
| RECS 2020 | TMY3 | ResStock upgrades 0 3 4 10 11 15 |               |
| missing-code cleaning                                               |
+---------------------------------------------------------------------+
| FEATURE LAYER                                                       |
| Physics-informed features | climate features | interactions         |
+---------------------------------------------------------------------+
| LEAKAGE-SAFE PROTOCOL                                               |
| Train / Val / Calib / Test | train-only fitting | audit suite       |
+---------------------------------------------------------------------+
| MODEL LAYER                                                         |
| Multi-task | transfer learning | stacking | conformal | SHAP        |
+---------------------------------------------------------------------+
| DECISION LAYER                                                      |
| Counterfactuals | retrofit deltas | teacher savings | manifests     |
+---------------------------------------------------------------------+
```

## 1. Runtime Options & Configuration Switches

| Option | Value / State | What It Enables |
|---|---|---|
| RECS row selection | `nrows: ALL` | Full dataset usage |
| Max features | `250` | Feature-selection cap |
| ResStock upgrades | `[0, 3, 4, 10, 11, 15]` | Multi-upgrade simulation support |
| Transfer learning | `Transfer: True` | Teacher to student meta-features |
| Survey weights | `Weights: True` | Population-weighted learning and evaluation |
| Hyperparameter optimization | `Optimize: True` | Optuna-based tuning |
| Fast mode | `Fast: False` | Full, non-shortcut execution |
| Seeds | `10` (ensemble: `3`) | Multi-seed robustness checks |
| Parallel jobs | `N_JOBS: 8` | Parallel worker execution |

---

## 2. Data Loading, Processing & Validation

- RECS 2020 loader with column mapping.
- Missing-code replacement to `NaN`.
- EUI calculation for total and end-uses: heating, cooling, DHW, lighting.
- Climate feature generation: HDD, CDD.
- EUI validation report.
- TMY3 metadata and hourly loading.
- Hourly TMY3 aggregation to 19 climate features.
- TMY3 merge by state.
- Cross-source validation: HDD correlation between RECS and TMY3.
- ResStock chunked loading from `.csv.gz`.
- Column retention for mapped columns only.
- All ResStock upgrade files loaded.

Note: This layer shows a reusable multi-source data ingestion and validation framework.

---

## 3. Physics-Informed Feature Engineering

- Building geometry:
  - `PH_building_height`
  - `PH_footprint_area`
  - `PH_building_perimeter`
  - `PH_wall_area`
  - `PH_roof_area`
  - `PH_building_volume`
  - `PH_surface_area`
  - `PH_surface_volume_ratio`
  - `PH_surface_factor`
  - `PH_log_sqft`
- Interaction features:
  - `PH_occ_x_sqft`
  - `PH_HDD_x_sqft`
  - `PH_vintage_x_HDD`
- Load and insulation features:
  - `PH_base_load`
  - `PH_ceiling_insul`
- Physics-based EUI estimate:
  - `PH_eui_estimate`

Note: The pipeline is not purely data-driven; it injects domain physics into the feature space.

---

## 4. Leakage-Safe Splitting & Preprocessing

                    +---------------------+
                    |     FULL DATASET    |
                    +----------+----------+
                               |
        +----------------------+----------------------+
        |                      |                      |
        v                      v                      v
 +------------+        +------------+        +------------+        +---------+
 |   TRAIN    |        |    VAL     |        |   CALIB    |        |  TEST   |
 +------------+        +------------+        +------------+        +---------+
        |                      |                      |                 |
        v                      v                      v                 v
  Model fitting        Model selection        Conformal calibration    Reporting
  + preprocessing      + early stopping       + untouched              only
  + encoding           + routing              + no fitting
                       + Optuna

- Four-way split before any fitted preprocessing.
- Stratified by `BA_climate_SIMPLE`.
- Role separation:
  - `val` for model selection, early stopping, routing, Optuna.
  - `calib` for untouched conformal calibration.
  - `test` for reporting only.
- Feature engineering fit on train only.
- Ordinal encoder fit on train only.
- Survey weights enabled and normalized (`NWEIGHT`).

Note: This is a strong anti-leakage protocol embedded at the split level.

---

## 5. Feature Selection

- Drop all-NaN and constant columns.
- Selection on train split only.
- Configurable max feature cap.
- Tracks physics feature share in the selected set.
- Appends TL meta-features later when transfer learning is active.

---

## 6. Transfer Learning & Knowledge Distillation

- ResStock teacher trained on simulation data.
- KD meta-features generated:
  - `TL_pred_logbtu`
- Ablation without TL under identical fitted protocol and fixed hyperparameters.
- Delta teachers for savings and CATE:
  - positional matching
  - cell-matched
- Teacher versus measured comparison for retrofit savings.

Note: The pipeline supports both standard distillation and simulation-derived causal-style delta teachers.

---

## 7. Model Training & Energy-Conserving Multi-Task Modeling

- Primary target: `log1p(TOTALBTU)`.
- Share models for:
  - heating
  - cooling
  - DHW
  - lighting
- Routing logic: share model versus direct model.
- Monotonicity constraints on physics-signed features.
- Optuna hyperparameter optimization.
- Early stopping.
- Ablation baselines:
  - direct EUI
  - v7-style
- Ablation retraining without TL under fixed hyperparameters.
- Model save and load from disk.
- Final model training on the full training set.

### 7.1. Primary Model Training Details

- Train primary model on total energy target.
- Apply monotonicity constraints on signed physics features.
- Run Optuna for hyperparameter search.
- Early stopping based on validation set.
- Record best trial and best loss.
- Prepare model for conformal calibration.

### 7.2. Share Model Training Details

- Train separate share model for each end-use.
- Compute mean share per end-use.
- Compare two strategies:
  - share model
  - direct model
- Choose better route based on validation R2.
- Use chosen route for final prediction.

### 7.3. Stacking Ensemble Training

- Base learners: `xgb`, `lgb`, `catboost`.
- Generate OOF meta-features.
- Train meta-learner with non-negative coefficients.
- Retrain base models on 100% of training data.
- Evaluate on held-out validation set.
- Save ensemble as `.pkl`.

### 7.4. Teacher and Delta Teacher Training

- Train ResStock teacher on simulation data.
- Generate KD meta-features for student.
- Train delta teachers for savings:
  - positional matching
  - cell-matched
- Use delta teachers in counterfactual analysis.

### 7.5. Multi-Seed Training

- Repeat training with multiple seeds.
- Compare TL effect and stacking effect across seeds.
- Compute confidence intervals and sign counts.

---

## 8. Evaluation, Validation & Testing

### 8.1. Roles of the Splits

- `train`: model fitting and preprocessing.
- `val`: model selection, early stopping, routing, Optuna.
- `calib`: untouched conformal calibration.
- `test`: final reporting only.

### 8.2. Evaluation Metrics

- Unweighted R2.
- Population-weighted R2.
- RMSE.
- Log-space R2.
- Conformal interval coverage.
- Population calibration error.
- Per-end-use evaluation:
  - heating
  - cooling
  - DHW
  - lighting
- Derived EUI and direct EUI evaluation.
- Stacking ensemble evaluation.

### 8.3. Multi-Layer Evaluation

- Multi-seed repetition for headline comparison.
- Confidence intervals and sign counts for:
  - TL effect
  - stacking effect
- Leave-One-Climate-Out (LOGO) with fresh process per zone.
- Held-out test evaluation.
- Three honest views:
  - unweighted test R2
  - population-weighted R2
  - log-space OOF and validation R2
- Per-end-use evaluation.

### 8.4. Evaluation on the Test Set

- Test set as held-out and untouched.
- Report total energy R2.
- Report derived EUI R2.
- Report direct EUI R2.
- Report per-end-use R2.
- Report stacking ensemble R2.
- Report coverage curve at 80%, 90%, 95%.
- Report population calibration error.
- Generate evaluation figures.

### 8.5. Delta Teacher Evaluation

- Compare delta teacher savings against measured values.
- Report mean, median, 10th percentile, 90th percentile.
- Report gap between teacher and measured.
- Use in counterfactual tables.

Note: Evaluation is designed to avoid single-number overclaiming.

---

## 9. SHAP Analysis & Interpretability

- Run SHAP analysis on the primary model.
- Extract top features.
- Report top features.
- Generate SHAP figure as vector (PDF).
- Use SHAP to interpret feature importance.
- Integrate SHAP into publication outputs.
- Examine contribution of physics features and TL meta-features.
- Help identify direction of feature effects.
- Help document model decisions.

Note: SHAP is treated as part of interpretability and publication outputs.

---

## 10. Uncertainty & Conformal Prediction

- Conformal calibration with `calib-split`.
- Coverage curve at 80%, 90%, 95%.
- Population calibration error.
- Cross-conformal OOF in ablation.
- Intervals reported in log-space.

Note: Uncertainty is treated as a first-class output, not an afterthought.

---

## 11. Leakage Audit

- Forbidden columns in train, val, calib, test.
- Single-feature correlation with target.
- Exact train/test row duplicates.
- Fitted-transformer verification.
- Flag for implausibly perfect score.
- Feature selection on train only.
- Linear-combination reconstruction check.
- Target-derived column lineage check.
- JSON audit report.

Note: This is a formal, automated leakage-audit subsystem.

---

## 12. Publication Outputs & Reproducibility

- Vector figures (PDF):
  - prediction versus actual
  - coverage conformal
  - architecture schematic
  - LOGO climate
  - counterfactual
  - SHAP
- SHAP analysis and top-feature reporting.
- Save model and ensemble.
- Save manifest and paper tables.
- Timestamped logs.
- Record paths for outputs, models, tables, and figures.

Note: The pipeline is built for paper-ready, reproducible outputs.

---

## 13. Counterfactual & Decision Support

- Retrofit measures:
  - Reference HVAC 2025
  - Cold Climate ASHP
  - Air Sealing
  - Attic Insulation
  - ENERGY STAR Windows
- ResStock deltas:
  - matched-cell
  - naive
- Composition-bias warning when naive and matched disagree in sign.
- Counterfactual scenarios on sampled households.
- Physics-only mode.
- Delta teacher savings estimates.
- Counterfactual and teacher counterfactual tables.

Note: This layer connects prediction to decision-support and retrofit analysis.

---

## 14. Parallelization & Resource Management

- `N_JOBS=8`.
- LOGO executed as fresh process per zone.
- Chunked loading for large `.csv.gz` files.
- Multiple seeds and ensemble seeds.

Note: Designed for heavy tabular pipelines with controlled parallelism.

---

## 15. Logging & Versioning

- Pipeline version: `v8.1`.
- Feature engineering version: `v8.0`.
- Step-by-step logging: `STEP 0` to `STEP 12b`.
- `INFO` and `WARNING` levels.
- Paths for outputs, models, tables, figures.
- Final summary and leakage-audit status.

Note: Versioned, stepwise logging makes the run auditable.

---

## 16. Final Summary

| Capability | Present? |
|---|---|
| Configurable runtime switches | Yes |
| Multi-source data loading | Yes |
| Physics-informed features | Yes |
| Leakage-safe four-way split | Yes |
| Transfer learning and KD | Yes |
| Energy-conserving multi-task learning | Yes |
| Primary and share model training | Yes |
| Share versus direct routing | Yes |
| Physics monotonicity constraints | Yes |
| Optuna optimization | Yes |
| Early stopping | Yes |
| Ablation without TL | Yes |
| Validation and test evaluation | Yes |
| Multi-seed evaluation | Yes |
| LOGO evaluation | Yes |
| Conformal uncertainty | Yes |
| Honest stacking ensemble | Yes |
| SHAP analysis | Yes |
| Automated leakage audit | Yes |
| Publication-ready figures | Yes |
| Counterfactual decision support | Yes |
| Parallel execution | Yes |
| Versioned logging | Yes |

> Bottom line: This log describes a research-grade building-energy pipeline with configurable options, physics-aware feature engineering, multi-task training, model routing, physics constraints, hyperparameter optimization, early stopping, validation and test evaluation, multi-seed evaluation, LOGO, conformal calibration, honest stacking, SHAP analysis, leakage audit, counterfactual analysis, and publication-oriented reproducibility.
