# Model Strategy

## Scope

This document defines the practical final-model direction for the project and a clean integration path for future feature sets, especially the turn-level features from Member 3.

## What From The Reference Notebook Is Useful

The reference notebook is useful for these ideas:

- use grouped validation based on player identity rather than random row splits
- prioritize tree-based gradient boosting for structured tabular features
- tune only a small set of strong candidates instead of tuning many weak models
- compare feature blocks incrementally rather than throwing everything in at once

These parts should **not** be copied blindly:

- very broad feature engineering without checking whether each block helps under leakage-safe validation
- large notebook-style experimentation that is hard to reproduce and integrate
- early dependence on many extra libraries when the current project already has a working sklearn-based baseline
- any validation setup that weakens the grouped-human split logic already established in `src/validation.py`

## Current Baseline Context

Current validated metadata-only baseline:

- `ridge_onehot`
- average RMSE around `147.23`
- grouped validation by human nickname

This is good enough to serve as the control model for all later comparisons.

Current combined-feature Member 4 scaffold comparison:

- the final-model scaffold now uses metadata from `src/baseline_model.py`
- it also uses turn-level aggregates from `src/turn_features.py`
- this is the first real final-model comparison on the richer feature set

## Current Implemented Model Performance

This is a regression task, so the relevant validation metrics are:

- RMSE
- MAE

There is no meaningful classification-style accuracy metric here.

Current grouped-human validation results on the **metadata + turn-feature** set:

| Model | Avg RMSE | Avg MAE | Current Rank |
|---|---:|---:|---:|
| `xgboost` | `150.3441` | `118.8258` | 1 |
| `ridge_onehot` | `150.7944` | `118.1805` | 2 |
| `hist_gbm` | `155.2655` | `122.1826` | 3 |
| `lightgbm` | `157.7103` | `124.0040` | 4 |

Current selected final model:

- `xgboost`

Why it is selected currently:

- it has the best average RMSE on the combined feature set
- once turn-level features are included, the nonlinear boosted model finally edges out ridge
- `ridge_onehot` still has slightly better average MAE, so it remains the strongest control model and a good candidate for simple blending if needed

## Candidate Model Families Worth Trying Seriously

### 1. Histogram Gradient Boosting

Why it is worth trying:

- strong tabular baseline available directly in sklearn
- handles nonlinear interactions better than ridge
- easy to integrate with current environment
- good candidate once turn-level aggregates are merged in

Role in project:

- immediate next serious candidate
- likely best non-external-library model in the current environment

### 2. Regularized Linear Model

Why it stays in the stack:

- cheap and stable benchmark
- useful sanity check against feature leakage or overfitting
- often surprisingly competitive on engineered aggregates

Role in project:

- baseline control model
- fallback final model if more complex models become unstable

### 3. LightGBM

Why it is worth trying:

- usually among the strongest choices for tabular competition-style regression
- often outperforms linear models once rich aggregate features exist
- strong candidate for the final submission once turn-level features are merged

Current status:

- installed in the current environment
- implemented in `src/model_candidates.py`
- ready for metadata-only benchmarking now

### 4. XGBoost

Why it is worth trying:

- strong competition-style tree boosting model
- useful second nonlinear benchmark against LightGBM
- can react differently from LightGBM on mixed feature sets

Current status:

- installed in the current environment
- implemented in `src/model_candidates.py`
- ready for metadata-only benchmarking now

### 5. CatBoost

Why it is still on the shortlist:

- often strong when categorical variables matter
- can simplify some categorical handling

Why it is lower priority:

- not currently installed
- current feature space is still relatively simple
- should be tested only if the currently implemented candidates plateau

## Recommended Final Candidate Order

Practical order of effort:

1. `XGBoost`
2. `ridge_onehot`
3. `hist_gbm`
4. `LightGBM`
5. `CatBoost` only if time remains or the implemented candidates plateau

## Realistic Tuning Plan

Do **not** run broad hyperparameter sweeps immediately.

Use this sequence:

### Phase 1: combined-feature calibration

- keep the current grouped-human folds
- compare `ridge_onehot`, `hist_gbm`, `LightGBM`, and `XGBoost` on metadata + turn features
- confirm that the selected model stays ahead on average RMSE after reruns
- keep an eye on fold stability, not just the mean

### Phase 2: limited tuning of the best nonlinear model

For `LightGBM`, `XGBoost`, or `hist_gbm`, tune only:

- learning rate
- tree depth or leaf complexity
- number of boosting iterations
- minimum samples per leaf / child weight
- regularization strength

Keep search small and evidence-driven:

- 10 to 25 parameter combinations is enough at first
- require improvement in average RMSE and acceptable fold stability

### Phase 3: optional ensemble

Only consider an ensemble if:

- the top 2 models have materially different error patterns
- both are validated under the same grouped-human folds
- the blended score beats the best single model consistently

Simple averaging is enough. Do not build stacking unless there is clear evidence it helps.

## Metrics To Track Consistently

For every candidate model and feature set, record:

- average RMSE across folds
- average MAE across folds
- fold-wise RMSE spread
- fold-wise MAE spread
- notes on training speed and stability

The project should optimize for validation reliability, not just the single best fold.

## Recommendation For Likely Final Model Stack

Most likely final stack:

- strong metadata + turn-feature table
- `xgboost` as the current main nonlinear final model
- `ridge_onehot` as the control baseline and possible blend partner
- `hist_gbm` as the sklearn fallback nonlinear benchmark

Most likely final submission path:

1. keep grouped-human validation fixed
2. merge Member 3 feature table into the current train/test frames
3. tune `xgboost` first
4. compare tuned `xgboost` against untuned or lightly tuned ridge and hist_gbm
5. optionally average `xgboost` with ridge if validation supports it

## Training Scaffold

The implementation scaffold for this strategy is in:

- `src/model_candidates.py`

It is designed to:

- reuse the Member 1 grouped validation
- start from the current metadata train/test frames
- merge turn-level aggregates from `src/turn_features.py`
- compare multiple candidate models under the same fold structure
- fit a recommended final model and produce test predictions

## Dependency On Member 3

This work is **not blocked** by Member 3.

Member 3 output is now being used through `src/turn_features.py`, so the final-model scaffold is no longer metadata-only. Further tuning is still useful, but the current final-model comparison is now meaningful and directly based on the richer feature set.
