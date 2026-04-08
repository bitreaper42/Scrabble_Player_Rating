from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.linear_model import Ridge

from src.validation import BOT_NAMES, extract_human_game_records, make_grouped_human_folds


TRAIN_CSV = ROOT / "data/raw/scrabble-player-rating/train.csv"
TEST_CSV = ROOT / "data/raw/scrabble-player-rating/test.csv"
GAMES_CSV = ROOT / "data/raw/scrabble-player-rating/games.csv"


NUMERIC_FEATURES = [
    "human_score",
    "bot_score",
    "score_diff",
    "score_total",
    "bot_rating",
    "first_is_bot",
    "initial_time_seconds",
    "increment_seconds",
    "max_overtime_minutes",
    "game_duration_seconds",
]

CATEGORICAL_FEATURES = [
    "bot_nickname",
    "time_control_name",
    "game_end_reason",
    "lexicon",
    "rating_mode",
]


def load_games() -> pd.DataFrame:
    games = pd.read_csv(GAMES_CSV)
    games["game_id"] = games["game_id"].astype(str)
    return games


def load_train_frame() -> pd.DataFrame:
    records = extract_human_game_records(TRAIN_CSV)
    frame = pd.DataFrame(records)
    frame["game_id"] = frame["game_id"].astype(str)
    return add_game_features(frame, load_games())


def load_test_frame() -> pd.DataFrame:
    test = pd.read_csv(TEST_CSV)
    test["game_id"] = test["game_id"].astype(str)
    human_rows = test[~test["nickname"].isin(BOT_NAMES)].rename(
        columns={"nickname": "human_nickname", "score": "human_score"}
    )
    bot_rows = test[test["nickname"].isin(BOT_NAMES)].rename(
        columns={"nickname": "bot_nickname", "score": "bot_score", "rating": "bot_rating"}
    )

    merged = human_rows[["game_id", "human_nickname", "human_score"]].merge(
        bot_rows[["game_id", "bot_nickname", "bot_score", "bot_rating"]],
        on="game_id",
        how="inner",
    )
    merged["human_score"] = merged["human_score"].astype(float)
    merged["bot_score"] = merged["bot_score"].astype(float)
    merged["bot_rating"] = merged["bot_rating"].astype(float)
    return add_game_features(merged, load_games())


def add_game_features(frame: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    merged = frame.merge(games, on="game_id", how="left", validate="one_to_one")
    merged["score_diff"] = merged["human_score"] - merged["bot_score"]
    merged["score_total"] = merged["human_score"] + merged["bot_score"]
    merged["first_is_bot"] = (merged["first"] == merged["bot_nickname"]).astype(float)
    return merged


def build_folded_train_frame(n_splits: int = 5, seed: int = 42) -> pd.DataFrame:
    frame = load_train_frame()
    fold_by_human = make_grouped_human_folds(TRAIN_CSV, n_splits=n_splits, seed=seed)
    frame["fold"] = frame["human_nickname"].map(fold_by_human)
    return frame


def make_model_pipelines() -> dict[str, Pipeline]:
    one_hot_preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), NUMERIC_FEATURES),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    ordinal_preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), NUMERIC_FEATURES),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "ordinal",
                            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                        ),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    return {
        "global_mean": Pipeline([("model", DummyRegressor(strategy="mean"))]),
        "ridge_onehot": Pipeline(
            [
                ("preprocess", one_hot_preprocessor),
                ("model", Ridge(alpha=3.0)),
            ]
        ),
        "hist_gbm": Pipeline(
            [
                ("preprocess", ordinal_preprocessor),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        learning_rate=0.06,
                        max_depth=6,
                        max_iter=300,
                        min_samples_leaf=30,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def evaluate_models(frame: pd.DataFrame, n_splits: int = 5) -> list[dict]:
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    models = make_model_pipelines()
    results: list[dict] = []

    for model_name, model in models.items():
        fold_metrics = []
        for fold in range(n_splits):
            train_fold = frame[frame["fold"] != fold]
            valid_fold = frame[frame["fold"] == fold]

            x_train = train_fold[features]
            y_train = train_fold["human_rating"]
            x_valid = valid_fold[features]
            y_valid = valid_fold["human_rating"]

            model.fit(x_train, y_train)
            preds = model.predict(x_valid)

            fold_metrics.append(
                {
                    "fold": fold,
                    "rmse": float(root_mean_squared_error(y_valid, preds)),
                    "mae": float(mean_absolute_error(y_valid, preds)),
                }
            )

        results.append(
            {
                "model": model_name,
                "folds": fold_metrics,
                "avg_rmse": sum(item["rmse"] for item in fold_metrics) / len(fold_metrics),
                "avg_mae": sum(item["mae"] for item in fold_metrics) / len(fold_metrics),
            }
        )

    results.sort(key=lambda item: item["avg_rmse"])
    return results


def fit_best_baseline(frame: pd.DataFrame) -> Pipeline:
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    model = make_model_pipelines()["ridge_onehot"]
    model.fit(frame[features], frame["human_rating"])
    return model


def sample_test_predictions(model: Pipeline, test_frame: pd.DataFrame, n: int = 5) -> list[dict]:
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    preds = model.predict(test_frame[features].head(n))
    output = []
    for row, pred in zip(test_frame.head(n).to_dict("records"), preds):
        output.append(
            {
                "game_id": row["game_id"],
                "nickname": row["human_nickname"],
                "predicted_rating": float(pred),
            }
        )
    return output


def main() -> None:
    train_frame = build_folded_train_frame(n_splits=5, seed=42)
    test_frame = load_test_frame()

    print("train_rows", len(train_frame))
    print("test_rows", len(test_frame))

    results = evaluate_models(train_frame, n_splits=5)
    for result in results:
        print(
            result["model"],
            "avg_rmse",
            round(result["avg_rmse"], 4),
            "avg_mae",
            round(result["avg_mae"], 4),
        )
        for fold_metric in result["folds"]:
            print(
                "  fold",
                fold_metric["fold"],
                "rmse",
                round(fold_metric["rmse"], 4),
                "mae",
                round(fold_metric["mae"], 4),
            )

    best_model = fit_best_baseline(train_frame)
    print("sample_test_predictions", sample_test_predictions(best_model, test_frame))


if __name__ == "__main__":
    main()
