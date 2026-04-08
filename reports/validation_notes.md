# Validation Notes

## Table Roles

- `train.csv`: one row per player per game in the training split. Each game appears exactly twice, once for the bot and once for the human opponent. Columns: `game_id`, `nickname`, `score`, `rating`.
- `test.csv`: same row structure as `train.csv`, but only the human rows have unknown targets. Bot rows in test already have ratings populated.
- `games.csv`: one row per game with metadata such as time control, lexicon, winner, duration, and creation timestamp.
- `turns.csv`: one row per turn for one player in one game, including rack, move, points, running score, and turn type.

## How The Tables Join

- `train.csv` and `test.csv` join to `games.csv` on `game_id`.
- `train.csv` and `test.csv` join to `turns.csv` on `game_id` and `nickname`.
- `games.csv` covers all games from both train and test.
- `turns.csv` covers all player-game pairs from both train and test.

## What One Training Row Means

One row in `train.csv` is one player in one game.

For this dataset:

- every training game has exactly 2 rows
- every game contains exactly 1 bot and 1 human
- the three bot nicknames are `BetterBot`, `HastyBot`, and `STEEBot`

So the natural supervised unit for the competition is:

- one human player in one game against one of the three bots

This is the unit that matters because in `test.csv`:

- bot rows have known ratings
- human rows have `rating = NA`

Therefore the actual prediction target is the rating of the human player rows only.

## Observed Dataset Structure

- `train.csv`: 100,820 rows, 50,410 games
- `test.csv`: 44,726 rows, 22,363 games
- `games.csv`: 72,773 rows, matching total train + test games
- `turns.csv`: 2,005,498 rows

Additional observations:

- train and test game IDs do not overlap
- only 3 nicknames overlap between train and test, and they are exactly the three bots
- all test games involve one of the three bots
- most human nicknames are unseen between train and test
- all games are from 2022, concentrated in July to September, so a strict time split is not the best approximation of the test setting

## Leakage Risks

### 1. Random row split

This is the main risk. A random split would place games from the same human nickname into both train and validation. That makes validation too optimistic because the model would see the same human player in training and validation.

### 2. Game-level split without grouping humans

Splitting by `game_id` alone is not enough. It prevents the same game from appearing in both train and validation, but the same human can still appear across both sides in different games.

### 3. Evaluating on bot rows

This is a target mismatch. The competition target is the human rating in test, not the bot rating. Validation should therefore be computed on held-out human rows only.

### 4. Human identity leakage through aggregated features

If future feature engineering aggregates across all games for a nickname before splitting, that would leak player identity information from validation back into training. Any nickname-level aggregates must be computed inside each fold using training data only.

### 5. Using post-game information without care

Some columns, such as final score, winner, and end reason, are available at prediction time for this competition because the task is to infer rating after the game log is observed. They are not train-test leakage by themselves. The leakage issue is not "future within the game"; it is "future across the same player identity appearing in both train and validation."

## Recommended Validation Strategy

### Primary recommendation

Use grouped cross-validation by **human nickname**, and score only on held-out human rows.

Why this is the best choice:

- it matches the test setting, where nearly all humans are unseen
- it prevents the same human nickname from appearing in both train and validation
- it still allows the three bots to appear naturally across folds, which matches the test set better than holding bots out entirely

### Practical fold definition

- identify the human row in each training game as the row whose nickname is not one of the three bot names
- assign each human nickname to exactly one fold
- place all games for that human into the same fold
- optionally balance folds by row count and bot-opponent mix
- evaluate only on the held-out human rows

### Secondary strategy

A grouped holdout split by human nickname can be used for quick iteration, but final comparison should use grouped cross-validation.

### Not recommended

- random K-fold on rows
- split by `game_id` only
- time-based split as the main validation method

## Reusable Split Code

The reusable validation helper is implemented in:

- `src/validation.py`

It provides:

- bot-name handling
- extraction of human-vs-bot game records
- deterministic grouped fold assignment by human nickname
- fold-balance inspection helpers

## Team Guidance

- Baseline and final models should treat human rows as the supervised target rows.
- Bot rows can still be useful as contextual information, for example to identify the opponent bot and its observed game statistics.
- If anyone trains on both human and bot rows for convenience, they should still report validation metrics on held-out human rows only.
