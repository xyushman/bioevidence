# 07 Forecasting: Trial Duration (and optional enrollment)

## Target
`duration_months = completion_date - start_date` for trials with status COMPLETED, both dates present and not imputed, duration between 1 and 180 months.
(Optional second target: `enrollment`.) Predict at trial registration time: use ONLY features known at start (no leakage from results/status updates).

## Features (`forecasting/features.py`)
phase, therapeutic area (from conditions), study type, allocation, masking, num_arms, intervention class counts, sponsor_class, number of countries,
number of sites if available, eligibility text length, number of inclusion/exclusion criteria (regex), primary endpoint category (from extractor, optional),
start year, planned enrollment. Optional text signal: SVD(50) of fine-tuned embedding of summary. Encode with target-safe encoders; handle rare categories.

## Split
Time-based: train start_date < 2018, val 2018-2019, test >= 2020 (adjust to keep >= 15% test; right-censoring caveat: recent trials that finished
quickly are overrepresented, state this limitation). No random split.

## Models
1. Baselines: global median; median by phase; Ridge regression.
2. Gradient boosting: LightGBM/XGBoost with early stopping on val; small Optuna search (<= 40 trials).
3. Small neural net (MLP in PyTorch, embeddings for categoricals) only if time permits.
Optionally quantile regression (10/50/90) for prediction intervals; report interval coverage.

## Metrics
MAE (months), RMSE, R², MAPE (guard against tiny targets), median AE; per-phase breakdown; calibration plot of predicted vs actual. Compare all vs baseline with bootstrap CIs.

## Explainability
SHAP summary + top-10 feature importance; 3 local explanations (fast, slow, mispredicted trials). Write a paragraph on what drives duration and where the model fails.

## Deliverables
`results/forecast/metrics.json`, `reports/forecasting.md` with plots (saved PNGs), saved model (`joblib`), `/forecast` API endpoint (optional) taking trial design fields.

## Acceptance criteria
Reproducible via `make forecast`; GBM beats median-by-phase baseline on test MAE (or an honest write-up why not); no feature leakage test (feature list asserted against an allowlist).
