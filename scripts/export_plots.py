from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from src.baseline_model import build_folded_train_frame, evaluate_models
from src.model_candidates import (
    evaluate_candidates,
    get_candidate_builders,
    infer_feature_lists,
    load_turn_feature_table,
    merge_optional_features,
)


PLOTS_DIR = ROOT / "plots"


def save(fig: plt.Figure, name: str) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_DIR / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def evaluate_candidates_verbose(train_frame: pd.DataFrame, feature_frame: pd.DataFrame) -> pd.DataFrame:
    merged = merge_optional_features(
        train_frame,
        feature_frame,
        key_columns=["game_id", "human_nickname"],
    )
    numeric_features, categorical_features = infer_feature_lists(
        merged,
        target_column="human_rating",
        exclude_columns=["game_id", "human_nickname"],
    )
    rows: list[dict] = []
    builders = get_candidate_builders()

    for model_name, builder in builders.items():
        for fold in sorted(merged["fold"].unique()):
            train_fold = merged[merged["fold"] != fold]
            valid_fold = merged[merged["fold"] == fold]

            x_train = train_fold[numeric_features + categorical_features]
            y_train = train_fold["human_rating"]
            x_valid = valid_fold[numeric_features + categorical_features]
            y_valid = valid_fold["human_rating"]

            model = builder(numeric_features, categorical_features)
            model.fit(x_train, y_train)
            preds = model.predict(x_valid)

            rows.append(
                {
                    "model": model_name,
                    "fold": int(fold),
                    "rmse": float(root_mean_squared_error(y_valid, preds)),
                    "mae": float(mean_absolute_error(y_valid, preds)),
                }
            )

    return pd.DataFrame(rows)


def export_baseline_plots(train_frame: pd.DataFrame) -> pd.DataFrame:
    baseline_results = evaluate_models(train_frame, n_splits=5)
    baseline_summary = pd.DataFrame(
        [
            {"model": item["model"], "avg_rmse": item["avg_rmse"], "avg_mae": item["avg_mae"]}
            for item in baseline_results
        ]
    ).sort_values("avg_rmse").reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    baseline_summary.plot.bar(x="model", y="avg_rmse", ax=axes[0], color=colors[: len(baseline_summary)])
    axes[0].set_title("Baseline Models: Average RMSE")
    axes[0].set_ylabel("RMSE")
    axes[0].tick_params(axis="x", rotation=15)

    baseline_summary.plot.bar(x="model", y="avg_mae", ax=axes[1], color=colors[: len(baseline_summary)])
    axes[1].set_title("Baseline Models: Average MAE")
    axes[1].set_ylabel("MAE")
    axes[1].tick_params(axis="x", rotation=15)

    plt.tight_layout()
    save(fig, "baseline_model_comparison.png")
    return baseline_summary


def export_final_plots(train_frame: pd.DataFrame, turn_feature_frame: pd.DataFrame, baseline_summary: pd.DataFrame) -> None:
    candidate_summary = evaluate_candidates(
        train_frame,
        feature_frame=turn_feature_frame,
        key_columns=["game_id", "human_nickname"],
    )
    candidate_folds = evaluate_candidates_verbose(train_frame, turn_feature_frame)

    plot_df = candidate_summary.sort_values("avg_rmse", ascending=True).copy()
    selected_model_name = str(plot_df.iloc[0]["model"])

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    rmse_colors = ["#1b9e77" if m == selected_model_name else "#7f8c8d" for m in plot_df["model"]]

    axes[0].barh(plot_df["model"], plot_df["avg_rmse"], color=rmse_colors)
    axes[0].set_title("Final Candidate Models: Average RMSE")
    axes[0].set_xlabel("RMSE")
    for y_pos, value in enumerate(plot_df["avg_rmse"]):
        axes[0].text(value + 0.2, y_pos, f"{value:.2f}", va="center")

    axes[1].barh(plot_df["model"], plot_df["avg_mae"], color=rmse_colors)
    axes[1].set_title("Final Candidate Models: Average MAE")
    axes[1].set_xlabel("MAE")
    for y_pos, value in enumerate(plot_df["avg_mae"]):
        axes[1].text(value + 0.2, y_pos, f"{value:.2f}", va="center")

    plt.tight_layout()
    save(fig, "final_candidate_comparison.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    for _, row in plot_df.iterrows():
        color = "#d95f02" if row["model"] == selected_model_name else "#4c78a8"
        size = 180 if row["model"] == selected_model_name else 100
        ax.scatter(row["avg_rmse"], row["avg_mae"], s=size, color=color)
        ax.text(row["avg_rmse"] + 0.15, row["avg_mae"] + 0.05, row["model"], fontsize=10)
    ax.set_title("Final Models: RMSE vs MAE")
    ax.set_xlabel("Average RMSE")
    ax.set_ylabel("Average MAE")
    save(fig, "final_rmse_vs_mae_scatter.png")

    rmse_heatmap = candidate_folds.pivot(index="model", columns="fold", values="rmse").loc[plot_df["model"]]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    im = ax.imshow(rmse_heatmap.values, cmap="YlGnBu", aspect="auto")
    ax.set_title("Fold-wise RMSE Heatmap")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Model")
    ax.set_xticks(range(len(rmse_heatmap.columns)))
    ax.set_xticklabels(rmse_heatmap.columns)
    ax.set_yticks(range(len(rmse_heatmap.index)))
    ax.set_yticklabels(rmse_heatmap.index)
    for i in range(rmse_heatmap.shape[0]):
        for j in range(rmse_heatmap.shape[1]):
            ax.text(j, i, f"{rmse_heatmap.iloc[i, j]:.1f}", ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, label="RMSE")
    plt.tight_layout()
    save(fig, "final_foldwise_rmse_heatmap.png")

    baseline_best = baseline_summary.iloc[0]
    final_best = plot_df.iloc[0]
    comparison = pd.DataFrame(
        [
            {"stage": "Baseline", "rmse": baseline_best["avg_rmse"], "mae": baseline_best["avg_mae"]},
            {"stage": "Final", "rmse": final_best["avg_rmse"], "mae": final_best["avg_mae"]},
        ]
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(comparison["stage"], comparison["rmse"], color=["#6c757d", "#1b9e77"])
    axes[0].set_title("Baseline vs Final RMSE")
    axes[0].set_ylabel("RMSE")
    for idx, value in enumerate(comparison["rmse"]):
        axes[0].text(idx, value + 0.2, f"{value:.2f}", ha="center")

    axes[1].bar(comparison["stage"], comparison["mae"], color=["#6c757d", "#1b9e77"])
    axes[1].set_title("Baseline vs Final MAE")
    axes[1].set_ylabel("MAE")
    for idx, value in enumerate(comparison["mae"]):
        axes[1].text(idx, value + 0.2, f"{value:.2f}", ha="center")

    plt.tight_layout()
    save(fig, "baseline_vs_final_comparison.png")


def main() -> None:
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    plt.style.use("seaborn-v0_8-whitegrid")

    train_frame = build_folded_train_frame(n_splits=5, seed=42)
    turn_feature_frame = load_turn_feature_table()

    baseline_summary = export_baseline_plots(train_frame)
    export_final_plots(train_frame, turn_feature_frame, baseline_summary)

    print(f"Exported plots to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
