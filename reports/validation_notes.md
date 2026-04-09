# Validation Report

## Dataset Table Roles

- `train.csv`: one row per player per game in the training split, with columns `game_id`, `nickname`, `score`, and `rating`
- `test.csv`: same row structure as `train.csv`, but the target rating is unknown for human rows
- `games.csv`: one row per game containing metadata such as time control, lexicon, winner, duration, and creation timestamp
- `turns.csv`: one row per turn for one player in one game, containing rack, move, points, running score, and turn type

## Table Relationships

- `train.csv` and `test.csv` join to `games.csv` on `game_id`
- `train.csv` and `test.csv` join to `turns.csv` on `game_id` and `nickname`
- each game appears twice in `train.csv` or `test.csv`: once for the bot and once for the human opponent

## Prediction Unit

One supervised row corresponds to one player in one game. For this competition, the meaningful target rows are the human player rows:

- every game has exactly one bot and one human
- the bot nicknames are `BetterBot`, `HastyBot`, and `STEEBot`
- in `test.csv`, bot rows already have ratings populated
- the prediction target is the human rating

## Observed Structure

- `train.csv`: 100,820 rows across 50,410 games
- `test.csv`: 44,726 rows across 22,363 games
- `games.csv`: 72,773 rows
- `turns.csv`: 2,005,498 rows

Additional observations:

- train and test game IDs do not overlap
- the only nickname overlap between train and test is the three bot names
- human players are effectively unseen between train and test

## Leakage Analysis

The main leakage risk is identity leakage across repeated human players in the training set.

Potential failure modes:

- random row splits allow the same human nickname to appear in both train and validation
- splitting only by `game_id` still allows the same human to appear on both sides in different games
- evaluating on bot rows does not match the actual prediction target
- precomputing nickname-level aggregates across all rows before splitting would leak validation information into training

Game outcome fields such as final score, winner, and end reason are valid features for this task because prediction happens after the game log is observed. The leakage concern is cross-player overlap, not within-game chronology.

## Final Validation Strategy

The final validation protocol is:

- grouped cross-validation by `human_nickname`
- 5 folds
- all games of the same human stay in the same fold
- evaluation only on held-out human rows

This protocol best matches the test setting, where the model must generalize to different human opponents while playing against the same small set of bots.

## Implementation

The reusable validation logic is implemented in:

- `src/validation.py`
