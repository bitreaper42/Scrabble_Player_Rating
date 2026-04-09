# Final Model Report

## Modeling Objective

The objective is to predict the rating of the human player in each test-game row using game metadata and turn-level gameplay information.

## Reference Notebook Decisions

The reference notebook was useful for identifying broad modeling directions, but the final project pipeline was reduced to the parts that remained effective under leakage-safe validation:

- grouped validation by player identity
- comparison of compact, high-signal feature sets
- focus on strong tabular regression models

The final pipeline avoids excessive feature inflation and keeps the modeling stack reproducible.

## Final Feature Set

The final model uses a combined feature table built from:

- metadata from `train.csv`, `test.csv`, and `games.csv`
- turn-level aggregate features derived from `turns.csv`

The turn-based aggregates include:

- turn counts and non-play rates
- scoring profile features
- score progression features
- rack and move-shape features
- bingo-like move indicators

## Candidate Models Evaluated

The following regression models were evaluated on the combined feature set:

| Model | Avg RMSE | Avg MAE | Rank |
|---|---:|---:|---:|
| `blend_ridge_hist` | `141.9355` | `109.6770` | 1 |
| `ridge_onehot` | `145.1150` | `111.4992` | 2 |
| `xgboost` | `151.2549` | `120.6188` | 3 |
| `hist_gbm` | `153.1473` | `119.6493` | 4 |
| `lightgbm` | `153.7146` | `119.3483` | 5 |

This is a regression task, so RMSE and MAE are the relevant evaluation metrics.

## Selected Final Model

Selected model:

- `blend_ridge_hist`

Reason for selection:

- best average RMSE across grouped-human folds
- best average MAE across the evaluated combined-feature candidates
- stronger validation performance than any single model tested on the same feature set

## Final Model Structure

The selected model is a weighted blend of two separately trained regressors:

- a tuned ridge regression pipeline with one-hot encoded categorical features
- a tuned histogram gradient boosting regressor

This blend combines:

- the stability of the linear model
- the nonlinear pattern capture of the boosted-tree model

### Architecture Details

`blend_ridge_hist` is not a single native library model. It is an ensemble built from two full pipelines that are both trained on the same combined feature table.

#### Component 1: Ridge pipeline

Input handling:

- numeric features are imputed and scaled
- categorical features are imputed and one-hot encoded

Estimator:

- `Ridge(alpha=10.0)`

Role in the ensemble:

- captures strong linear relationships between rating and engineered features
- works especially well with sparse one-hot encoded categorical variables
- provides stable predictions and acts as the low-variance component of the final ensemble

#### Component 2: Histogram Gradient Boosting pipeline

Input handling:

- numeric features are median-imputed
- categorical features are ordinal-encoded

Estimator:

- `HistGradientBoostingRegressor`
- `learning_rate=0.05`
- `max_depth=4`
- `max_iter=900`
- `min_samples_leaf=15`
- `l2_regularization=0.5`

Role in the ensemble:

- captures nonlinear effects and feature interactions
- models relationships that the ridge component cannot represent well
- acts as the higher-capacity nonlinear component of the final ensemble

#### Blend rule

Let:

- `y_ridge` = prediction from the ridge pipeline
- `y_hist` = prediction from the histogram gradient boosting pipeline

The final prediction is:

- `0.65 * y_ridge + 0.35 * y_hist`

This weighted average is implemented through a custom blend regressor in `src/model_candidates.py`.

### Why This Blend Works

The two component models make different types of errors:

- ridge is strong on broad linear trends and categorical structure
- histogram gradient boosting is stronger on nonlinear interactions within the metadata and turn-level aggregates

Blending them reduces dependence on one model family and improves overall validation performance. In this project, the weighted blend performs better than either component alone on grouped-human cross-validation.

## Final Result

Final validation performance:

- RMSE: `141.9355`
- MAE: `109.6770`

This is the best project-wide result obtained under the grouped-human validation protocol.

## Implementation

The final-model training scaffold is implemented in:

- `src/model_candidates.py`
