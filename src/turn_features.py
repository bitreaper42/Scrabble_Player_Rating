from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd


TURNS_CSV = ROOT / "data/raw/scrabble-player-rating/turns.csv"
ID_COLUMNS = ["game_id", "nickname"]


def load_turns(turns_csv_path: str | Path = TURNS_CSV) -> pd.DataFrame:
    turns = pd.read_csv(turns_csv_path)
    turns["game_id"] = turns["game_id"].astype(str)
    turns["nickname"] = turns["nickname"].astype(str)
    return turns.sort_values(["game_id", "nickname", "turn_number"]).reset_index(drop=True)


def _prepare_turn_level_columns(turns: pd.DataFrame) -> pd.DataFrame:
    frame = turns.copy()
    frame["turn_type"] = frame["turn_type"].fillna("Missing")
    frame["rack"] = frame["rack"].fillna("")
    frame["move"] = frame["move"].fillna("")
    frame["location"] = frame["location"].fillna("")

    frame["prev_score"] = frame.groupby(ID_COLUMNS)["score"].shift(fill_value=0)
    frame["score_gain"] = frame["score"] - frame["prev_score"]
    frame["score_gain"] = frame["score_gain"].clip(lower=0)

    frame["rack_len"] = frame["rack"].str.len()
    frame["blank_tiles_on_rack"] = frame["rack"].str.count(r"\?")
    frame["move_clean"] = frame["move"].str.replace(".", "", regex=False)
    frame["move_len"] = frame["move_clean"].str.len()

    frame["is_play"] = (frame["turn_type"] == "Play").astype(int)
    frame["is_exchange"] = (frame["turn_type"] == "Exchange").astype(int)
    frame["is_pass"] = (frame["turn_type"] == "Pass").astype(int)
    frame["is_challenge"] = (frame["turn_type"] == "Challenge").astype(int)
    frame["is_timeout"] = (frame["turn_type"] == "Timeout").astype(int)
    frame["is_end"] = (frame["turn_type"] == "End").astype(int)
    frame["is_six_zero_rule"] = (frame["turn_type"] == "Six-Zero Rule").astype(int)
    frame["is_missing_type"] = (frame["turn_type"] == "Missing").astype(int)
    frame["is_scoring_turn"] = (frame["points"] > 0).astype(int)
    frame["is_zero_point_turn"] = (frame["points"] == 0).astype(int)
    frame["is_high_scoring_turn"] = (frame["points"] >= 30).astype(int)
    frame["used_all_tiles"] = (frame["move_len"] >= 7).astype(int)
    frame["has_board_location"] = frame["location"].ne("").astype(int)

    return frame


def build_turn_features(turns: pd.DataFrame | None = None) -> pd.DataFrame:
    if turns is None:
        turns = load_turns()

    prepared = _prepare_turn_level_columns(turns)

    feature_frame = (
        prepared.groupby(ID_COLUMNS)
        .agg(
            turns_played=("turn_number", "size"),
            avg_turn_number=("turn_number", "mean"),
            max_turn_number=("turn_number", "max"),
            play_turns=("is_play", "sum"),
            exchange_turns=("is_exchange", "sum"),
            pass_turns=("is_pass", "sum"),
            challenge_turns=("is_challenge", "sum"),
            timeout_turns=("is_timeout", "sum"),
            end_turns=("is_end", "sum"),
            six_zero_rule_turns=("is_six_zero_rule", "sum"),
            missing_type_turns=("is_missing_type", "sum"),
            scoring_turn_rate=("is_scoring_turn", "mean"),
            zero_point_turn_rate=("is_zero_point_turn", "mean"),
            high_scoring_turn_rate=("is_high_scoring_turn", "mean"),
            avg_points=("points", "mean"),
            median_points=("points", "median"),
            max_points=("points", "max"),
            std_points=("points", "std"),
            avg_score_gain=("score_gain", "mean"),
            max_score_gain=("score_gain", "max"),
            std_score_gain=("score_gain", "std"),
            avg_rack_len=("rack_len", "mean"),
            avg_blank_tiles_on_rack=("blank_tiles_on_rack", "mean"),
            avg_move_len=("move_len", "mean"),
            max_move_len=("move_len", "max"),
            bingo_like_turns=("used_all_tiles", "sum"),
            board_move_rate=("has_board_location", "mean"),
            final_score=("score", "max"),
        )
        .reset_index()
    )

    feature_frame["points_per_turn"] = feature_frame["final_score"] / feature_frame["turns_played"]
    feature_frame["bingo_like_rate"] = feature_frame["bingo_like_turns"] / feature_frame["turns_played"]
    feature_frame["non_play_turns"] = (
        feature_frame["exchange_turns"]
        + feature_frame["pass_turns"]
        + feature_frame["challenge_turns"]
        + feature_frame["timeout_turns"]
        + feature_frame["six_zero_rule_turns"]
        + feature_frame["missing_type_turns"]
    )
    feature_frame["non_play_turn_rate"] = feature_frame["non_play_turns"] / feature_frame["turns_played"]

    numeric_columns = feature_frame.select_dtypes(include="number").columns
    feature_frame[numeric_columns] = feature_frame[numeric_columns].fillna(0.0)
    return feature_frame.sort_values(ID_COLUMNS).reset_index(drop=True)


def merge_turn_features(
    frame: pd.DataFrame,
    turn_features: pd.DataFrame | None = None,
    nickname_column: str = "nickname",
) -> pd.DataFrame:
    if turn_features is None:
        turn_features = build_turn_features()

    if nickname_column not in frame.columns:
        raise KeyError(f"Expected nickname column '{nickname_column}' in frame")

    keyed_frame = frame.copy()
    keyed_frame["game_id"] = keyed_frame["game_id"].astype(str)
    keyed_frame[nickname_column] = keyed_frame[nickname_column].astype(str)

    features = turn_features.rename(columns={"nickname": nickname_column})
    merged = keyed_frame.merge(features, on=["game_id", nickname_column], how="left", validate="one_to_one")

    feature_columns = [column for column in features.columns if column not in {"game_id", nickname_column}]
    merged[feature_columns] = merged[feature_columns].fillna(0.0)
    return merged


def build_feature_views() -> dict[str, pd.DataFrame]:
    turns = load_turns()
    features = build_turn_features(turns)
    return {
        "turns": turns,
        "turn_features": features,
    }


def main() -> None:
    feature_frame = build_turn_features()
    print("turn_feature_rows", len(feature_frame))
    print("turn_feature_columns", len(feature_frame.columns))
    print(feature_frame.head().to_string(index=False))


if __name__ == "__main__":
    main()
