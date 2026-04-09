from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.baseline_model import (
    # CATEGORICAL_FEATURES,
    # NUMERIC_FEATURES,
    build_folded_train_frame,
    load_test_frame,
)
from src.turn_features import build_turn_features

try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None


class WeightedBlendRegressor:
    def __init__(self, estimators: list[tuple[str, object]], weights: list[float]) -> None:
        self.estimators = estimators
        self.weights = weights

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WeightedBlendRegressor":
        self.fitted_estimators_: list[tuple[str, object]] = []
        for name, estimator in self.estimators:
            fitted = estimator.fit(X, y)
            self.fitted_estimators_.append((name, fitted))
        return self

    def predict(self, X: pd.DataFrame):
        blended = None
        for weight, (_, estimator) in zip(self.weights, self.fitted_estimators_):
            preds = estimator.predict(X)
            blended = weight * preds if blended is None else blended + weight * preds
        return blended


def merge_optional_features(
    base_frame: pd.DataFrame,
    extra_frame: pd.DataFrame | None = None,
    key_columns: list[str] | None = None,
) -> pd.DataFrame:

    if extra_frame is None:
        return base_frame.copy()

    key_columns = key_columns or ["game_id", "human_nickname"]
    merged = base_frame.merge(extra_frame, on=key_columns, how="left")
    return merged


def load_turn_feature_table(nickname_column: str = "human_nickname") -> pd.DataFrame:
    feature_frame = build_turn_features().rename(columns={"nickname": nickname_column})
    feature_frame["game_id"] = feature_frame["game_id"].astype(str)
    feature_frame[nickname_column] = feature_frame[nickname_column].astype(str)
    return feature_frame


def infer_feature_lists(
    frame: pd.DataFrame,
    target_column: str = "human_rating",
    exclude_columns: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    exclude = set(exclude_columns or [])
    exclude.update({"fold", target_column})

    numeric_features: list[str] = []
    categorical_features: list[str] = []

    for column in frame.columns:
        if column in exclude:
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            numeric_features.append(column)
        else:
            categorical_features.append(column)

    return numeric_features, categorical_features


def build_linear_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    alpha: float = 3.0,
) -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    return Pipeline(
        [
            ("preprocess", preprocess),
            ("model", Ridge(alpha=alpha)),
        ]
    )


def build_hist_gbm_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    learning_rate: float = 0.05,
    max_depth: int = 6,
    max_iter: int = 400,
    min_samples_leaf: int = 30,
    l2_regularization: float = 0.0,
) -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric_features,
            ),
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
                categorical_features,
            ),
        ]
    )

    return Pipeline(
        [
            ("preprocess", preprocess),
            (
                "model",
                HistGradientBoostingRegressor(
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    max_iter=max_iter,
                    min_samples_leaf=min_samples_leaf,
                    l2_regularization=l2_regularization,
                    random_state=42,
                ),
            ),
        ]
    )


def get_candidate_builders() -> dict[str, Callable[[list[str], list[str]], Pipeline]]:
    builders: dict[str, Callable[[list[str], list[str]], Pipeline]] = {
        "ridge_onehot": lambda num, cat: build_linear_pipeline(num, cat, alpha=10.0),
        "hist_gbm": lambda num, cat: build_hist_gbm_pipeline(
            num,
            cat,
            learning_rate=0.05,
            max_depth=4,
            max_iter=900,
            min_samples_leaf=15,
            l2_regularization=0.5,
        ),
    }

    if LGBMRegressor is not None:
        builders["lightgbm"] = lambda num, cat: Pipeline(
            [
                (
                    "preprocess",
                    ColumnTransformer(
                        transformers=[
                            (
                                "num",
                                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                                num,
                            ),
                            (
                                "cat",
                                Pipeline(
                                    [
                                        ("imputer", SimpleImputer(strategy="most_frequent")),
                                        (
                                            "ordinal",
                                            OrdinalEncoder(
                                                handle_unknown="use_encoded_value",
                                                unknown_value=-1,
                                            ),
                                        ),
                                    ]
                                ),
                                cat,
                            ),
                        ]
                    ),
                ),
                (
                    "model",
                    LGBMRegressor(
                        n_estimators=700,
                        learning_rate=0.03,
                        num_leaves=31,
                        max_depth=-1,
                        min_child_samples=20,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        reg_alpha=0.0,
                        reg_lambda=0.5,
                        random_state=42,
                        verbose=-1,
                    ),
                ),
            ]
        )

    if XGBRegressor is not None:
        builders["xgboost"] = lambda num, cat: Pipeline(
            [
                (
                    "preprocess",
                    ColumnTransformer(
                        transformers=[
                            (
                                "num",
                                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                                num,
                            ),
                            (
                                "cat",
                                Pipeline(
                                    [
                                        ("imputer", SimpleImputer(strategy="most_frequent")),
                                        (
                                            "ordinal",
                                            OrdinalEncoder(
                                                handle_unknown="use_encoded_value",
                                                unknown_value=-1,
                                            ),
                                        ),
                                    ]
                                ),
                                cat,
                            ),
                        ]
                    ),
                ),
                (
                    "model",
                    XGBRegressor(
                        n_estimators=600,
                        learning_rate=0.03,
                        max_depth=5,
                        min_child_weight=4,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        reg_alpha=0.0,
                        reg_lambda=1.0,
                        objective="reg:squarederror",
                        random_state=42,
                        n_jobs=4,
                        verbosity=0,
                    ),
                ),
            ]
        )

    builders["blend_ridge_hist"] = lambda num, cat: WeightedBlendRegressor(
        estimators=[
            ("ridge", build_linear_pipeline(num, cat, alpha=10.0)),
            (
                "hist_gbm",
                build_hist_gbm_pipeline(
                    num,
                    cat,
                    learning_rate=0.05,
                    max_depth=4,
                    max_iter=900,
                    min_samples_leaf=15,
                    l2_regularization=0.5,
                ),
            ),
        ],
        weights=[0.65, 0.35],
    )

    return builders

def evaluate_candidates(
    train_frame: pd.DataFrame,
    feature_frame: pd.DataFrame | None = None,
    key_columns: list[str] | None = None,
    target_column: str = "human_rating",
) -> pd.DataFrame:
    merged = merge_optional_features(train_frame, feature_frame, key_columns=key_columns)

    numeric_features, categorical_features = infer_feature_lists(
        merged,
        target_column=target_column,
        exclude_columns=["game_id", "human_nickname"],
    )

    candidate_builders = get_candidate_builders()
    rows: list[dict] = []

    for model_name, builder in candidate_builders.items():
        fold_scores = []
        for fold in sorted(merged["fold"].unique()):
            train_fold = merged[merged["fold"] != fold]
            valid_fold = merged[merged["fold"] == fold]

            x_train = train_fold[numeric_features + categorical_features]
            y_train = train_fold[target_column]
            x_valid = valid_fold[numeric_features + categorical_features]
            y_valid = valid_fold[target_column]

            model = builder(numeric_features, categorical_features)
            model.fit(x_train, y_train)
            preds = model.predict(x_valid)

            fold_scores.append(
                {
                    "model": model_name,
                    "fold": int(fold),
                    "rmse": float(root_mean_squared_error(y_valid, preds)),
                    "mae": float(mean_absolute_error(y_valid, preds)),
                }
            )

        fold_df = pd.DataFrame(fold_scores)
        rows.append(
            {
                "model": model_name,
                "avg_rmse": float(fold_df["rmse"].mean()),
                "avg_mae": float(fold_df["mae"].mean()),
                "std_rmse": float(fold_df["rmse"].std(ddof=0)),
                "std_mae": float(fold_df["mae"].std(ddof=0)),
                "num_numeric_features": len(numeric_features),
                "num_categorical_features": len(categorical_features),
            }
        )

    return pd.DataFrame(rows).sort_values("avg_rmse").reset_index(drop=True)


def fit_best_candidate_model(
    train_frame: pd.DataFrame,
    feature_frame: pd.DataFrame | None = None,
    key_columns: list[str] | None = None,
    target_column: str = "human_rating",
) -> tuple[str, Pipeline, list[str], list[str], pd.DataFrame]:
    results = evaluate_candidates(
        train_frame,
        feature_frame=feature_frame,
        key_columns=key_columns,
        target_column=target_column,
    )
    best_model_name = str(results.iloc[0]["model"])

    merged = merge_optional_features(train_frame, feature_frame, key_columns=key_columns)
    numeric_features, categorical_features = infer_feature_lists(
        merged,
        target_column=target_column,
        exclude_columns=["game_id", "human_nickname", "fold"],
    )

    candidate_builders = get_candidate_builders()
    model = candidate_builders[best_model_name](numeric_features, categorical_features)
    model.fit(merged[numeric_features + categorical_features], merged[target_column])
    return best_model_name, model, numeric_features, categorical_features, results


def predict_test_frame(
    model: Pipeline,
    test_frame: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    feature_frame: pd.DataFrame | None = None,
    key_columns: list[str] | None = None,
) -> pd.DataFrame:
    merged = merge_optional_features(test_frame, feature_frame, key_columns=key_columns)
    merged = merged.copy()
    merged["predicted_rating"] = model.predict(merged[numeric_features + categorical_features])
    return merged[["game_id", "human_nickname", "predicted_rating"]].rename(
        columns={"human_nickname": "nickname"}
    )


def main() -> None:
    train_frame = build_folded_train_frame(n_splits=5, seed=42)
    test_frame = load_test_frame()
    turn_feature_frame = load_turn_feature_table()

    results = evaluate_candidates(
        train_frame,
        feature_frame=turn_feature_frame,
        key_columns=["game_id", "human_nickname"],
    )
    print(results.to_string(index=False))

    (
        best_model_name,
        model,
        numeric_features,
        categorical_features,
        _,
    ) = fit_best_candidate_model(
        train_frame,
        feature_frame=turn_feature_frame,
        key_columns=["game_id", "human_nickname"],
    )
    print("selected_model", best_model_name)
    preds = predict_test_frame(
        model,
        test_frame,
        numeric_features,
        categorical_features,
        feature_frame=turn_feature_frame,
        key_columns=["game_id", "human_nickname"],
    )
    print(preds.head().to_string(index=False))


if __name__ == "__main__":
    main()
