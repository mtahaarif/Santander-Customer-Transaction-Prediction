# Santander Customer Transaction Prediction
An end-to-end machine learning pipeline built for the Kaggle competition [**Santander Customer Transaction Prediction**](https://www.kaggle.com/competitions/santander-customer-transaction-prediction), where the task is to predict which customers will make a specific future transaction, regardless of the amount, using 200 anonymized numerical features.

The project goes beyond a single notebook submission: it benchmarks six model families, performs exhaustive feature selection, tunes hyperparameters with Optuna, and explains the final model's predictions with SHAP — all backed by saved artifacts for full reproducibility.

**Final result: 0.89693 public / 0.89412 private leaderboard ROC-AUC**, versus an 0.8597 logistic-regression baseline.

----

## Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
  - [Phase 0 — EDA](#phase-0--exploratory-data-analysis)
  - [Phase 1 — Model Selection](#phase-1--model-selection)
  - [Phase 2 — Feature Selection](#phase-2--dynamic-feature-selection)
  - [Phase 3 — Hyperparameter Tuning](#phase-3--optuna-hyperparameter-tuning)
  - [Phase 4 — Champion Model & Threshold Tuning](#phase-4--champion-model--threshold-tuning)
  - [Phase 5 — Interpretability](#phase-5--interpretability-shap--permutation-importance)
  - [Phase 6 — Deployment Artifacts](#phase-6--deployment--artifacts)
- [Results Summary](#results-summary)
- [Reproducibility](#reproducibility)
- [Getting Started](#getting-started)
- [Key Findings](#key-findings)
- [Reports](#reports)

---

## Problem Statement

Santander's fraud/transaction team needs to identify, ahead of time, which customers are likely to perform a specific transaction — independent of the transaction amount. This is framed as a **binary classification** problem with:

- **200,000 rows** in both train and test sets
- **200 anonymized numerical features** (`var_0` … `var_199`)
- A **highly imbalanced target**: ~90% negative class (`target = 0`), ~10% positive class (`target = 1`)
- No missing values and no categorical features — every column is a real-valued float

The evaluation metric is **ROC-AUC**, and the primary modeling challenges are class imbalance and the sheer width of the feature space relative to signal strength.

## Dataset

The raw data (`train.csv`, `test.csv`) is **not included in this repository** (Kaggle competition rules / file size) — it must be downloaded from the [competition data page](https://www.kaggle.com/competitions/santander-customer-transaction-prediction/data) and placed so the notebooks' `DATA_DIR` path resolves correctly (by default `/kaggle/input/competitions/santander-customer-transaction-prediction` when run on Kaggle).

| Split | Rows | Columns |
|---|---|---|
| `train.csv` | 200,000 | `ID_code`, `target`, `var_0`–`var_199` |
| `test.csv` | 200,000 | `ID_code`, `var_0`–`var_199` |

## Project Structure

```
Santander-Customer-Transaction-Prediction/
├── santander-customer-transaction-prediction.ipynb   # Final, end-to-end pipeline (Phases 0-6)
├── generate the missing artifacts.ipynb              # Regenerates report artifacts (EDA plots, SHAP waterfalls, repro JSON)
│
├── prototyping/                                       # Iterative development notebooks
│   ├── model selection.ipynb                          # Benchmarks 6 model families on a holdout split
│   ├── Dynamic Feature Selection.ipynb                 # Exhaustive top-k gain-importance sweep (k = 1..200)
│   ├── Optuna hyperparameter tuning for LightGBM.ipynb # Bayesian hyperparameter search
│   └── Post-Hoc Model Interpretability.ipynb           # SHAP + permutation importance deep dive
│
├── results/                                            # Final run outputs
│   ├── submission.csv                                  # Kaggle submission file (ID_code, target probability)
│   ├── Kaggle Competition Leaderboard results.png       # Screenshot of the final leaderboard score
│   ├── artifacts/
│   │   ├── champion_model_full.pkl                     # Trained LightGBM champion model
│   │   ├── scaler.joblib                               # Fitted StandardScaler
│   │   ├── selected_features.pkl                       # List of the 180 selected feature names
│   │   ├── confusion_matrix.png
│   │   ├── permutation_importance.png
│   │   ├── shap_summary.png
│   │   ├── threshold_curve.png
│   │   └── pipeline_metadata.txt                       # Final feature count, OOF AUC, threshold, Optuna params
│   └── __results___files/                              # Inline plots exported from the notebook run
│
├── missing artifacts results/
│   └── final_report_artifacts/
│       ├── eda_correlation_heatmap.png
│       ├── eda_target_imbalance.png
│       ├── eda_var_81_distribution.png
│       ├── shap_local_waterfall_1.png / _2.png / _3.png # Per-customer local explanations
│       └── reproducibility.json                        # Environment + metric snapshot for the report
│
├── Project Report.pdf                                  # Written project report
└── CEP Machine Learning Project Spring 2026_final.pdf   # Course/assignment brief
```

## Methodology

The pipeline is organized into seven phases, implemented first as separate prototyping notebooks and then consolidated into the single production notebook `santander-customer-transaction-prediction.ipynb`.

### Phase 0 — Exploratory Data Analysis

- Verified there are **zero missing values** across all 200 features.
- Confirmed the target imbalance: roughly **89.95% class 0 vs. 10.05% class 1**.
- Plotted feature distributions (e.g. `var_0`–`var_5`, `var_81`) — most features are approximately Gaussian / bimodal-Gaussian mixtures with no obvious categorical structure.
- Computed a correlation heatmap over a feature subset (`var_0`–`var_19`) — pairwise correlations between anonymized features are close to **zero**, meaning most predictive signal comes from marginal (univariate) feature-target relationships rather than interactions, which favors tree-based models with strong per-feature splitting and count-based statistics.

Artifacts: `eda_correlation_heatmap.png`, `eda_target_imbalance.png`, `eda_var_81_distribution.png`.

### Phase 1 — Model Selection

Before committing to a single algorithm, six model families were benchmarked on an 80/20 stratified train/validation split with standardized features (`prototyping/model selection.ipynb`):

| Model | Validation ROC-AUC | Notes |
|---|---|---|
| Logistic Regression | 0.8597 | Baseline; strong given the linear-ish per-feature signal |
| Random Forest (balanced) | 0.7796 | Collapses on the minority class without careful tuning |
| XGBoost | 0.8574 | Competitive but behind LightGBM out-of-the-box |
| **LightGBM (balanced)** | **0.8667** | Best out-of-the-box performer, chosen as the champion family |
| Keras Neural Network (256→… , BatchNorm, Dropout, L2) | 0.8551 | Regularized MLP with class weighting; competitive but harder to tune/interpret |
| K-Nearest Neighbors (k=100) | 0.7298 | Weakest — curse of dimensionality over 200 features |

**LightGBM** was selected as the champion algorithm: it natively handles class imbalance via `class_weight='balanced'`, trains efficiently on GPU, and gave the best baseline ROC-AUC with room for further gains via feature selection and tuning.

### Phase 2 — Dynamic Feature Selection

Rather than assuming all 200 features are useful, an **exhaustive step-wise search** was run over Gain-based feature importances extracted from a LightGBM booster:

1. Rank all 200 features by LightGBM **gain importance**.
2. For every candidate count `k` (initially stepped by 10, later swept exhaustively from `k = 1` to `k = 200`), retrain on the top-`k` features using 3-fold stratified CV for speed.
3. Track validation ROC-AUC as a function of `k` and pick the `k` that maximizes it.

This identified an optimal subset of **180 features** as the mathematical peak — trimming 20 uninformative/noisy features without sacrificing (and slightly improving) generalization, while reducing model size and inference cost.

Artifact: feature-count-vs-AUC curve (`prototyping/Dynamic Feature Selection.ipynb`), `selected_features.pkl`.

### Phase 3 — Optuna Hyperparameter Tuning

With the lean 180-feature set fixed, [Optuna](https://optuna.org/) ran a Bayesian search over the LightGBM hyperparameter space (learning rate, `num_leaves`, `max_depth`, `min_child_samples`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`) using AUC as the objective, with early stopping and GPU acceleration.

**Best parameters found:**

```json
{
  "learning_rate": 0.02373565758661305,
  "num_leaves": 16,
  "max_depth": 10,
  "min_child_samples": 56,
  "subsample": 0.5219978247279289,
  "colsample_bytree": 0.5363049085869913,
  "reg_alpha": 1.1769679138751632e-08,
  "reg_lambda": 0.0021012573431610855,
  "class_weight": "balanced",
  "device": "gpu",
  "n_estimators": 3000,
  "random_state": 42
}
```

### Phase 4 — Champion Model & Threshold Tuning

The tuned LightGBM model was retrained with **5-fold stratified cross-validation** on the 180-feature lean dataset, producing out-of-fold (OOF) predictions used to:

- Report the unbiased OOF **ROC-AUC of 0.89483**.
- Average the 5 fold models' test-set probabilities into the final submission.
- Sweep the decision threshold against precision/recall/F1 to find an **operating threshold of 0.6744** (rather than the naive 0.5), better balancing precision and recall under the ~10% positive-class prior.

Artifacts: `confusion_matrix.png`, `threshold_curve.png`, `champion_model_full.pkl`.

### Phase 5 — Interpretability (SHAP & Permutation Importance)

To open up the "black box" LightGBM model, the pipeline computes:

- **Global SHAP summary plot** — ranks features by mean absolute SHAP value across a 2,000-row subsample, confirming which of the 180 features drive predictions most consistently.
- **Permutation importance** — an independent, model-agnostic cross-check of the SHAP ranking.
- **Local SHAP waterfall plots** for three individual customers — showing exactly which features pushed each prediction up or down, and how that can diverge from the global ranking (a feature globally minor can be locally decisive for a specific customer).

Artifacts: `shap_summary.png`, `permutation_importance.png`, `shap_local_waterfall_1.png`, `_2.png`, `_3.png`.

### Phase 6 — Deployment & Artifacts

The final phase persists everything needed to reproduce or serve the model without rerunning training:

- `scaler.joblib` — the fitted `StandardScaler` (fit once on the full training set in Phase 0).
- `champion_model_full.pkl` — the final LightGBM model trained on all data with the tuned hyperparameters.
- `selected_features.pkl` — the exact 180 feature names required at inference time.
- `pipeline_metadata.txt` — feature count, OOF AUC, decision threshold, and Optuna params in one place.
- `submission.csv` — the Kaggle-format `ID_code, target` submission file.
- `reproducibility.json` — competition/framework/validation-strategy/metrics snapshot used in the written report.
- `reproducibility_appendix.json` — captured Python/library versions and random seed for exact environment reproduction.

## Results Summary

| Metric | Value |
|---|---|
| Validation strategy | 5-fold stratified cross-validation |
| Final feature count | 180 (down from 200) |
| OOF ROC-AUC | **0.89483** |
| Public leaderboard ROC-AUC | **0.89693** |
| Private leaderboard ROC-AUC | **0.89412** |
| Optimal decision threshold | 0.6744 |
| Baseline (logistic regression) ROC-AUC | 0.8597 |
| Uplift over baseline | **+0.038** ROC-AUC |

The compliance check in the final notebook explicitly asserts `final_model_auc > baseline_logistic_auc`, guaranteeing the tuned LightGBM pipeline strictly outperforms the required linear baseline.

## Reproducibility

Environment captured at the time of the final run (`reproducibility.json` / `reproducibility_appendix.json`):

- **Framework:** LightGBM (GPU)
- **Validation:** 5-fold stratified CV
- **Random seed:** 42 (used consistently across `train_test_split`, `StratifiedKFold`, model `random_state`, and SHAP subsampling)
- Python, scikit-learn, LightGBM, SHAP, and Optuna versions are recorded automatically at run time and written to `reproducibility_appendix.json`.

All artifacts needed to reproduce inference (scaler, feature list, trained model) are saved under `results/artifacts/`, so predictions can be regenerated without rerunning the full training/tuning pipeline.

## Getting Started

1. **Set up the environment.** This project was developed on Kaggle notebooks with GPU acceleration. To run locally, create a Python 3.10+ environment with:
   ```bash
   pip install pandas numpy scikit-learn lightgbm xgboost shap optuna joblib matplotlib seaborn tensorflow
   ```
2. **Download the data.** Get `train.csv` and `test.csv` from the [competition page](https://www.kaggle.com/competitions/santander-customer-transaction-prediction/data) and update the `DATA_DIR` / `ARTIFACTS_DIR` constants at the top of the notebooks to point at your local paths (they default to Kaggle's `/kaggle/input/...` and `/kaggle/working/...` conventions).
3. **Run the pipeline.**
   - For the full, consolidated pipeline: run `santander-customer-transaction-prediction.ipynb` top to bottom (Phases 0–6 plus the EDA/compliance/explainability addenda).
   - To explore individual stages: run the notebooks in `prototyping/` in this order — `model selection.ipynb` → `Dynamic Feature Selection.ipynb` → `Optuna hyperparameter tuning for LightGBM.ipynb` → `Post-Hoc Model Interpretability.ipynb`.
4. **Inspect outputs.** Trained artifacts, plots, and the submission file are written to `results/artifacts/` and `results/submission.csv` (or `missing artifacts results/final_report_artifacts/` for the report-focused regeneration notebook).

## Key Findings

- **Class imbalance matters more than algorithm exoticism.** Simply using `class_weight='balanced'` moved LightGBM's recall on the positive class from near-zero (untuned Random Forest / KNN) to usable levels, and was a bigger lever than swapping model families.
- **Low pairwise feature correlation** means the 200 anonymized variables behave mostly independently — this favors gradient-boosted trees (which exploit per-feature splits) over models relying on learned feature interactions or distance metrics (KNN performed worst of all six models tested).
- **Not all 200 features carry signal.** An exhaustive sweep showed model quality peaks at 180 features — the remaining 20 add noise rather than signal, and removing them slightly improved OOF AUC while shrinking the model.
- **The default 0.5 threshold is suboptimal under imbalance.** Tuning the decision threshold (0.6744) against the precision/recall curve materially improves the practical precision/recall trade-off compared to the naive midpoint.
- **Global and local explanations diverge.** SHAP global summaries identify features that matter *on average*, but individual local waterfall plots show that features irrelevant globally can dominate a specific customer's prediction — important context for using this model as a decision-support tool rather than a black box.

## Reports

- [`Project Report.pdf`](./Project%20Report.pdf) — the full written report covering methodology, results, and discussion.
- [`CEP Machine Learning Project Spring 2026_final.pdf`](./CEP%20Machine%20Learning%20Project%20Spring%202026_final.pdf) — the original project/course brief this work was completed against.

---
om local workspace.
