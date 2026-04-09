# Turn Feature Report

## Source And Join Logic

`turns.csv` is a turn-by-turn gameplay log with one row per player action inside a game. The turn-level feature table is aggregated to one row per:

- `game_id`
- `nickname`

These keys align directly with the player-game rows in `train.csv` and `test.csv`.

## Engineered Feature Groups

The turn-feature pipeline produces the following groups of per-player-per-game features:

### Turn Volume And Tempo

- `turns_played`
- `avg_turn_number`
- `max_turn_number`

### Turn-Type Mix

- `play_turns`
- `exchange_turns`
- `pass_turns`
- `challenge_turns`
- `timeout_turns`
- `end_turns`
- `six_zero_rule_turns`
- `missing_type_turns`
- `non_play_turns`
- `non_play_turn_rate`

### Scoring Profile

- `scoring_turn_rate`
- `zero_point_turn_rate`
- `high_scoring_turn_rate`
- `avg_points`
- `median_points`
- `max_points`
- `std_points`

### Score Progression

- `avg_score_gain`
- `max_score_gain`
- `std_score_gain`
- `final_score`
- `points_per_turn`

### Rack And Move Shape

- `avg_rack_len`
- `avg_blank_tiles_on_rack`
- `avg_move_len`
- `max_move_len`
- `board_move_rate`

### Bingo-Like Indicators

- `bingo_like_turns`
- `bingo_like_rate`

## Feature Construction Notes

- turns are sorted by `game_id`, `nickname`, and `turn_number`
- `score_gain` is derived from the cumulative `score` sequence within each player-game
- blank-tile pressure is approximated by counting `?` in the rack
- bingo-like turns are approximated using cleaned move strings with length at least 7
- missing values in `turn_type`, `rack`, `move`, and `location` are normalized before aggregation

## Role In The Final Pipeline

These features are merged with metadata features and used by the final-model pipeline. They add gameplay-specific information that is not present in game-level metadata alone.

## Implementation

The turn-feature pipeline is implemented in:

- `src/turn_features.py`
