from __future__ import annotations

import csv
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


BOT_NAMES = frozenset({"BetterBot", "HastyBot", "STEEBot"})


def load_csv_rows(path: str | Path) -> List[dict]:
    path = Path(path)
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def is_bot(nickname: str) -> bool:
    return nickname in BOT_NAMES


def split_train_rows_by_role(train_csv_path: str | Path) -> tuple[list[dict], list[dict]]:
    rows = load_csv_rows(train_csv_path)
    human_rows = [row for row in rows if not is_bot(row["nickname"])]
    bot_rows = [row for row in rows if is_bot(row["nickname"])]
    return human_rows, bot_rows


def build_game_player_map(rows: Sequence[dict]) -> Dict[str, List[dict]]:
    by_game: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        by_game[row["game_id"]].append(row)
    return by_game


def extract_human_game_records(train_csv_path: str | Path) -> List[dict]:
    rows = load_csv_rows(train_csv_path)
    by_game = build_game_player_map(rows)
    records: List[dict] = []

    for game_id, game_rows in by_game.items():
        if len(game_rows) != 2:
            raise ValueError(f"Expected exactly 2 rows for game_id={game_id}, found {len(game_rows)}")

        human_rows = [row for row in game_rows if not is_bot(row["nickname"])]
        bot_rows = [row for row in game_rows if is_bot(row["nickname"])]

        if len(human_rows) != 1 or len(bot_rows) != 1:
            raise ValueError(
                "Expected one human row and one bot row per game, "
                f"found {len(human_rows)} human rows and {len(bot_rows)} bot rows for game_id={game_id}"
            )

        human_row = human_rows[0]
        bot_row = bot_rows[0]
        records.append(
            {
                "game_id": game_id,
                "human_nickname": human_row["nickname"],
                "human_rating": float(human_row["rating"]),
                "human_score": float(human_row["score"]),
                "bot_nickname": bot_row["nickname"],
                "bot_rating": float(bot_row["rating"]),
                "bot_score": float(bot_row["score"]),
            }
        )

    return records


def summarize_human_groups(train_csv_path: str | Path) -> Dict[str, dict]:
    records = extract_human_game_records(train_csv_path)
    summary: Dict[str, dict] = {}

    by_human: Dict[str, List[dict]] = defaultdict(list)
    for record in records:
        by_human[record["human_nickname"]].append(record)

    for nickname, human_records in by_human.items():
        bot_counts = Counter(record["bot_nickname"] for record in human_records)
        ratings = [record["human_rating"] for record in human_records]
        summary[nickname] = {
            "num_games": len(human_records),
            "bot_counts": dict(bot_counts),
            "rating_min": min(ratings),
            "rating_max": max(ratings),
        }

    return summary


def make_grouped_human_folds(
    train_csv_path: str | Path,
    n_splits: int = 5,
    seed: int = 42,
) -> Dict[str, int]:
    """
    Assign each human nickname to exactly one fold.

    This mirrors the test setting more closely than random row splits:
    almost all human nicknames in test are unseen, while the three bots are seen.
    """

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")

    group_summary = summarize_human_groups(train_csv_path)
    humans = list(group_summary.items())

    rng = random.Random(seed)
    rng.shuffle(humans)
    humans.sort(key=lambda item: item[1]["num_games"], reverse=True)

    fold_sizes = [0] * n_splits
    fold_bot_counts = [Counter() for _ in range(n_splits)]
    assignments: Dict[str, int] = {}

    for nickname, info in humans:
        best_fold = None
        best_score = None

        for fold_idx in range(n_splits):
            bot_penalty = 0
            for bot_name in BOT_NAMES:
                projected_bot_count = fold_bot_counts[fold_idx][bot_name] + info["bot_counts"].get(bot_name, 0)
                current_bot_totals = [counter[bot_name] for counter in fold_bot_counts]
                projected_mean = (sum(current_bot_totals) + info["bot_counts"].get(bot_name, 0)) / n_splits
                bot_penalty += abs(projected_bot_count - projected_mean)

            score = (fold_sizes[fold_idx], bot_penalty, fold_idx)
            if best_score is None or score < best_score:
                best_score = score
                best_fold = fold_idx

        assignments[nickname] = best_fold  # type: ignore[assignment]
        fold_sizes[best_fold] += info["num_games"]  # type: ignore[index]
        for bot_name, count in info["bot_counts"].items():
            fold_bot_counts[best_fold][bot_name] += count  # type: ignore[index]

    return assignments


def build_human_validation_rows(
    train_csv_path: str | Path,
    n_splits: int = 5,
    seed: int = 42,
) -> List[dict]:
    fold_by_human = make_grouped_human_folds(train_csv_path, n_splits=n_splits, seed=seed)
    records = extract_human_game_records(train_csv_path)

    validation_rows = []
    for record in records:
        validation_rows.append(
            {
                "game_id": record["game_id"],
                "human_nickname": record["human_nickname"],
                "bot_nickname": record["bot_nickname"],
                "target_rating": record["human_rating"],
                "fold": fold_by_human[record["human_nickname"]],
            }
        )
    return validation_rows


def iter_fold_indices(
    validation_rows: Sequence[dict],
    fold: int,
) -> tuple[List[int], List[int]]:
    train_idx: List[int] = []
    valid_idx: List[int] = []

    for idx, row in enumerate(validation_rows):
        if row["fold"] == fold:
            valid_idx.append(idx)
        else:
            train_idx.append(idx)

    return train_idx, valid_idx


def describe_fold_balance(validation_rows: Sequence[dict]) -> List[dict]:
    fold_counts: Dict[int, Counter] = defaultdict(Counter)

    for row in validation_rows:
        fold = int(row["fold"])
        fold_counts[fold]["rows"] += 1
        fold_counts[fold][row["bot_nickname"]] += 1

    description = []
    for fold in sorted(fold_counts):
        counter = fold_counts[fold]
        description.append(
            {
                "fold": fold,
                "rows": counter["rows"],
                "BetterBot": counter["BetterBot"],
                "HastyBot": counter["HastyBot"],
                "STEEBot": counter["STEEBot"],
            }
        )
    return description


if __name__ == "__main__":
    train_csv = Path("data/raw/scrabble-player-rating/train.csv")
    validation_rows = build_human_validation_rows(train_csv, n_splits=5, seed=42)
    for item in describe_fold_balance(validation_rows):
        print(item)
