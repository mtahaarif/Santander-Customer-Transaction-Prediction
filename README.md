# Santander Customer Transaction Prediction

This repository contains work for the Kaggle competition "Santander Customer Transaction Prediction". It includes notebooks, experiments, and artifacts used to train and evaluate models for predicting whether a customer will make a specific transaction.

Contents
- `prototyping/` — exploratory notebooks and experiments
- `results/` — outputs, submission CSV, and saved artifacts (scaler, pipeline metadata)
- `missing artifacts results/` — supporting reproducibility artifacts
- Notebooks at the repository root used for final analysis and artifact generation

Reproducibility
- A `scaler.joblib` and `pipeline_metadata.txt` are present under `results/artifacts/` to help reproduce preprocessing steps.

Quick start
1. Create a Python environment and install needed packages (e.g., `scikit-learn`, `lightgbm`, `pandas`, `joblib`).
2. Open the notebooks in `prototyping/` or the root notebooks and run cells to reproduce experiments.

If you'd like, I can also add a `.gitignore` to exclude heavy artifacts (models, datasets) and a `requirements.txt` listing exact dependencies.

---
Updated and pushed from local workspace.
