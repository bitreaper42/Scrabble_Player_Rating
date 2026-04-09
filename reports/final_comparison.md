# Final Model Comparison

## Scope

This report compares:

- the best validated baseline model
- the current selected final model

The goal is to measure what changed after introducing turn-level aggregate features from `src/turn_features.py` into the final-model pipeline.

## Validation Setup

Both results below use the same validation principle:

- grouped validation by `human_nickname`
- 5 folds
- evaluation on held-out human rows only

This keeps the comparison fair and directly aligned with the competition setting.

## Baseline Model

Feature source:

- metadata-only features from `src/baseline_model.py`
- no turn-level features

Selected baseline model:

- `ridge_onehot`

Baseline performance:

- RMSE: `147.2317`
- MAE: `112.0614`

## Final Model

Feature source:

- metadata features from `src/baseline_model.py`
- turn-level aggregate features from `src/turn_features.py`

Selected final model:

- `blend_ridge_hist`

Final-model performance:

- RMSE: `141.9355`
- MAE: `109.6770`

## Model-Wise Comparison

| Stage | Model | Features Used | RMSE | MAE |
|---|---|---|---:|---:|
| Baseline | `ridge_onehot` | metadata only | `147.2317` | `112.0614` |
| Final | `blend_ridge_hist` | metadata + turn features | `141.9355` | `109.6770` |

## Effect Of Introducing Turn Features

Comparing the current final model against the best baseline:

- RMSE change: `-5.2962`
- MAE change: `-2.3844`

Interpretation:

- lower is better for both RMSE and MAE
- so the current final model is **better** than the best baseline on grouped validation

## What This Means

Adding the current turn-level feature set and then tuning/blending the final model **did** improve the best validation score.

What improved:

- the project now has a richer feature space
- the final-model scaffold can evaluate stronger nonlinear and blended models
- the blended final model now beats the old metadata-only baseline on both RMSE and MAE

## Important Observation

Within the final combined-feature comparison itself:

- `blend_ridge_hist` ranked 1st by RMSE
- it also achieved the best MAE among the combined-feature candidates

So the correct conclusion is:

- the current selected final model is `blend_ridge_hist`
- it is also the best project-wide result so far

## Recommended Next Step

Before treating the current final model as submission-final, the team should:

1. keep the grouped-human validation fixed
2. generate the final test-set predictions from `blend_ridge_hist`
3. keep `ridge_onehot` as the fallback control model
4. only replace the blend if a simpler model clearly beats it under the same folds

## Final Summary

- Best baseline model: `ridge_onehot`
- Baseline score: RMSE `147.2317`, MAE `112.0614`
- Current final model: `blend_ridge_hist`
- Final-model score: RMSE `141.9355`, MAE `109.6770`
- Net effect of adding current turn features and then tuning/blending the final model: validation performance improved
