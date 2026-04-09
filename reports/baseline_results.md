# Baseline Model Report

## Baseline Scope

The baseline model uses:

- player-game rows from `train.csv`
- game-level metadata from `games.csv`

It does not use turn-level features from `turns.csv`.

## Validation Protocol

Baseline evaluation uses the grouped-human validation scheme implemented in `src/validation.py`:

- 5 folds
- grouping by `human_nickname`
- all games of the same human placed in the same fold
- scoring on held-out human rows only

## Baseline Features

Numeric features:

- `human_score`
- `bot_score`
- `score_diff`
- `score_total`
- `bot_rating`
- `first_is_bot`
- `initial_time_seconds`
- `increment_seconds`
- `max_overtime_minutes`
- `game_duration_seconds`

Categorical features:

- `bot_nickname`
- `time_control_name`
- `game_end_reason`
- `lexicon`
- `rating_mode`

## Models Evaluated

### Global Mean

- RMSE: `238.8272`
- MAE: `208.1407`

### Ridge Regression With One-Hot Encoding

- RMSE: `147.2317`
- MAE: `112.0614`

Fold-wise RMSE:

- Fold 0: `120.2585`
- Fold 1: `161.4775`
- Fold 2: `149.8568`
- Fold 3: `148.1997`
- Fold 4: `156.3660`

Fold-wise MAE:

- Fold 0: `87.5960`
- Fold 1: `127.8221`
- Fold 2: `107.8489`
- Fold 3: `117.6819`
- Fold 4: `119.3582`

### HistGradientBoostingRegressor

- RMSE: `157.0773`
- MAE: `122.7128`

## Selected Baseline Model

Selected baseline:

- `ridge_onehot`

Baseline performance:

- RMSE: `147.2317`
- MAE: `112.0614`

## Interpretation

The metadata-only baseline already captures substantial signal from final scores, bot context, and game metadata. Under grouped-human validation, ridge regression with one-hot encoded categorical features is the strongest baseline among the evaluated metadata-only models.

## Implementation

The baseline pipeline is implemented in:

- `src/baseline_model.py`
