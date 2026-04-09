# Turn Feature Notes

## Scope

This file documents the Member 3 turn-level feature work only. The implementation lives in `src/turn_features.py` and does not modify any raw data files or the reference notebook.

## Source Table and Merge Keys

- `turns.csv` is a turn-by-turn event log with one row per player action within a game.
- The reusable feature table is aggregated to one row per `game_id` + `nickname`.
- Those keys align cleanly with both `train.csv` and `test.csv`, which also contain one row per player-game pair.
- A quick coverage check showed every `train` and `test` player row has a matching `turns.csv` key.

## Feature Groups Created

The feature builder creates a compact per-player-per-game table with these groups:

- Turn volume and tempo:
  `turns_played`, `avg_turn_number`, `max_turn_number`
- Turn-type mix:
  `play_turns`, `exchange_turns`, `pass_turns`, `challenge_turns`, `timeout_turns`, `end_turns`, `six_zero_rule_turns`, `missing_type_turns`, `non_play_turns`, `non_play_turn_rate`
- Scoring profile:
  `scoring_turn_rate`, `zero_point_turn_rate`, `high_scoring_turn_rate`, `avg_points`, `median_points`, `max_points`, `std_points`
- Score progression:
  `avg_score_gain`, `max_score_gain`, `std_score_gain`, `final_score`, `points_per_turn`
- Rack and move shape:
  `avg_rack_len`, `avg_blank_tiles_on_rack`, `avg_move_len`, `max_move_len`, `board_move_rate`
- Bingo-style pressure indicators:
  `bingo_like_turns`, `bingo_like_rate`

## Implementation Notes

- The code sorts turns by `game_id`, `nickname`, and `turn_number` before computing progression features.
- `score_gain` is derived from the cumulative `score` column within each player-game sequence.
- Blank-tile pressure is approximated by counting `?` characters in `rack`.
- Bingo-like turns are approximated as cleaned move strings with length at least 7. This is a practical heuristic, not a rules-perfect bingo detector.
- Missing categorical values in `turn_type`, `rack`, `move`, and `location` are normalized before aggregation so the output is merge-safe.
- `merge_turn_features(...)` is included so other teammates can attach the turn aggregates to `train` or `test` without duplicating merge logic.

## Most Promising Features

The most promising turn-derived groups for later modeling are likely:

- Scoring consistency and ceiling:
  `avg_points`, `std_points`, `max_points`, `high_scoring_turn_rate`
- Risk / stability behavior:
  `exchange_turns`, `pass_turns`, `timeout_turns`, `non_play_turn_rate`
- Move quality proxies:
  `bingo_like_turns`, `bingo_like_rate`, `avg_move_len`
- Overall game execution:
  `final_score`, `points_per_turn`, `avg_score_gain`

These should be especially useful when combined with game metadata and validation logic owned by other teammates.
