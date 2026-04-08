# Baseline Results

## Scope

This baseline uses only:

- human-vs-bot game rows derived from `train.csv` / `test.csv`
- game-level metadata from `games.csv`

It does **not** use any turn-level features from `turns.csv`.

## Validation Approach Used

The baseline reuses the grouped-human validation strategy from `src/validation.py`.

Validation setup:

- 5 folds
- grouped by `human_nickname`
- all games of the same human stay in the same fold
- evaluation is on held-out human rows only

This matches the competition setup better than random row splits because the test set contains mostly unseen humans and only the three bot nicknames overlap between train and test.

## Features Used

The baseline feature set is intentionally simple and metadata-light.

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

Categorical one-hot features:

- `bot_nickname`
- `time_control_name`
- `game_end_reason`
- `lexicon`
- `rating_mode`

Notes:

- `first_is_bot` is derived from `games.csv:first`
- no turn-level logs are used
- no nickname-level aggregate history is used

## Models Tried

### 1. Global Mean

Predict the same average human rating for every held-out row.

Average CV metrics:

- RMSE: `238.8272`
- MAE: `208.1407`

### 2. Ridge + One-Hot Metadata

`scikit-learn` pipeline:

- numeric imputation + scaling
- categorical imputation + one-hot encoding
- `Ridge(alpha=3.0)`

Average CV metrics:

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

### 3. HistGradientBoosting

`scikit-learn` pipeline:

- numeric median imputation
- categorical ordinal encoding
- `HistGradientBoostingRegressor`

Average CV metrics:

- RMSE: `157.0773`
- MAE: `122.7128`

## Best Baseline Result

Best baseline so far:

- model: `ridge_onehot`
- average RMSE: `147.2317`
- average MAE: `112.0614`

This gives the team a clean metadata-only benchmark before any turn-level feature engineering.

## Interpretation

- The grouped-human validation is noticeably harder than random row evaluation would be.
- Final scores, bot rating, and game metadata already contain useful signal for human rating prediction.
- Even without turn logs, metadata-only baselines already provide strong signal.
- There is still fold variation, so later work should test whether turn-level features improve both average score and fold stability.

## Implementation

Code file:

- `src/baseline_model.py`

The script currently:

- builds human-only training examples
- merges in game metadata
- evaluates baseline models with grouped-human folds
- can train a full sklearn baseline and generate sample test predictions

## Environment Note

The baseline now uses a local virtual environment at `.venv/` with standard libraries including:

- `numpy`
- `pandas`
- `scikit-learn`
