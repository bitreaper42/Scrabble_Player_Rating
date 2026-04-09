# Final Model Comparison

## Evaluation Setting

All results in this report use the same validation setup:

- 5-fold grouped cross-validation
- grouping by `human_nickname`
- evaluation on held-out human rows only

This keeps the baseline and final-model comparison directly comparable.

## Baseline Model

Model:

- `ridge_onehot`

Feature set:

- game-level metadata from `games.csv`
- player-game metadata derived from `train.csv`
- no turn-level features from `turns.csv`

Validation performance:

- RMSE: `147.2317`
- MAE: `112.0614`

## Final Model

Model:

- `blend_ridge_hist`

Feature set:

- metadata features from the baseline pipeline
- turn-level aggregate features derived from `turns.csv`

Validation performance:

- RMSE: `141.9355`
- MAE: `109.6770`

## Comparison

| Model | Features Used | RMSE | MAE |
|---|---|---:|---:|
| `ridge_onehot` | metadata only | `147.2317` | `112.0614` |
| `blend_ridge_hist` | metadata + turn features | `141.9355` | `109.6770` |

## Impact Of Turn Features

Change from baseline to final model:

- RMSE improvement: `5.2962`
- MAE improvement: `2.3844`

Lower RMSE and MAE are better, so the final model outperforms the baseline on both metrics.

## Conclusion

The final submission model is `blend_ridge_hist`. Compared with the metadata-only baseline, adding turn-level aggregate features and selecting the blended model improves validation performance under the grouped-human evaluation protocol.
