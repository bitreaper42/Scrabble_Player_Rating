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
- the current best validated submission model is now a weighted blend rather than a single model

## Current Implemented Model Performance

This is a regression task, so the relevant validation metrics are:

- RMSE
- MAE

There is no meaningful classification-style accuracy metric here.

Current grouped-human validation results on the **metadata + turn-feature** set:

| Model | Avg RMSE | Avg MAE | Current Rank |
|---|---:|---:|---:|
| `blend_ridge_hist` | `141.9355` | `109.6770` | 1 |
| `ridge_onehot` | `145.1150` | `111.4992` | 2 |
| `xgboost` | `151.2549` | `120.6188` | 3 |
| `hist_gbm` | `153.1473` | `119.6493` | 4 |
| `lightgbm` | `153.7146` | `119.3483` | 5 |

Current selected final model:

- `blend_ridge_hist`

Why it is selected currently:

- it has the best average RMSE on the combined feature set
- it also has the best average MAE among the final combined-feature candidates
- blending a strong linear model with a tuned nonlinear model works better than either model alone

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

1. `blend_ridge_hist`
2. `ridge_onehot`
3. `xgboost`
4. `hist_gbm`
5. `lightgbm`
5. `CatBoost` only if time remains or the implemented candidates plateau

## Realistic Tuning Plan

Do **not** run broad hyperparameter sweeps immediately.

Use this sequence:

### Phase 1: combined-feature calibration

- keep the current grouped-human folds
- compare `ridge_onehot`, `hist_gbm`, `LightGBM`, `XGBoost`, and simple blends on metadata + turn features
- confirm that the selected blended model stays ahead on average RMSE after reruns
- keep an eye on fold stability, not just the mean

### Phase 2: limited tuning of the best nonlinear model

For `LightGBM`, `XGBoost`, `hist_gbm`, or the blend components, tune only:

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
- `blend_ridge_hist` as the current submission-grade final model
- `ridge_onehot` as the control baseline
- `xgboost` and `hist_gbm` as the main single-model alternatives

Most likely final submission path:

1. keep grouped-human validation fixed
2. merge Member 3 feature table into the current train/test frames
3. keep the grouped-human validation fixed
4. compare the tuned blend against single-model alternatives
5. only replace the blend if a simpler model clearly beats it under the same folds

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

Member 3 output is now being used through `src/turn_features.py`, so the final-model scaffold is no longer metadata-only. The current selected final model is already a validated blended model on the richer feature set.
