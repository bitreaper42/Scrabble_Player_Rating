import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error

sys.path.append('src')
from model_candidates import (
    build_folded_train_frame,
    get_candidate_builders,
    infer_feature_lists,
    merge_optional_features,
    build_linear_pipeline,
    build_hist_gbm_pipeline,
    load_turn_feature_table
)

def main():
    train_frame = build_folded_train_frame(n_splits=5, seed=42)
    turn_feature_frame = load_turn_feature_table()
    
    merged = merge_optional_features(train_frame, turn_feature_frame, key_columns=["game_id", "human_nickname"])
    numeric_features, categorical_features = infer_feature_lists(
        merged,
        target_column="human_rating",
        exclude_columns=["game_id", "human_nickname", "fold"],
    )

    builders = get_candidate_builders()
    
    ridge_builder = lambda num, cat: build_linear_pipeline(num, cat, alpha=10.0)
    hist_builder = lambda num, cat: build_hist_gbm_pipeline(
        num,
        cat,
        learning_rate=0.05,
        max_depth=4,
        max_iter=900,
        min_samples_leaf=15,
        l2_regularization=0.5,
    )

    print("Training models across 5 folds...")
    
    # Store predictions per fold to calculate metrics exactly like evaluate_candidates
    fold_results = []
    
    for fold in sorted(merged["fold"].unique()):
        train_fold = merged[merged["fold"] != fold]
        valid_fold = merged[merged["fold"] == fold]
        
        x_train = train_fold[numeric_features + categorical_features]
        y_train = train_fold["human_rating"]
        x_valid = valid_fold[numeric_features + categorical_features]
        y_valid = valid_fold["human_rating"]
        
        ridge_model = ridge_builder(numeric_features, categorical_features)
        ridge_model.fit(x_train, y_train)
        ridge_preds_fold = ridge_model.predict(x_valid)
        
        hist_model = hist_builder(numeric_features, categorical_features)
        hist_model.fit(x_train, y_train)
        hist_preds_fold = hist_model.predict(x_valid)
        
        fold_results.append({
            'y_valid': y_valid.values,
            'ridge_preds': ridge_preds_fold,
            'hist_preds': hist_preds_fold
        })

    weights = np.linspace(0, 1, 101)
    avg_rmses = []
    
    for w in weights:
        # Calculate RMSE for each fold independently
        fold_rmses = []
        for result in fold_results:
            fold_pred = w * result['ridge_preds'] + (1 - w) * result['hist_preds']
            fold_rmse = root_mean_squared_error(result['y_valid'], fold_pred)
            fold_rmses.append(fold_rmse)
        
        # Average the fold RMSEs
        avg_rmse = np.mean(fold_rmses)
        avg_rmses.append(avg_rmse)
        
    plt.figure(figsize=(10, 6))
    plt.plot(weights, avg_rmses, label='Avg Fold RMSE', color='blue', linewidth=2)
    plt.axvline(x=0.65, color='red', linestyle='--', label=f'Chosen Weight (Ridge=0.65)')
    
    plt.scatter([0], [avg_rmses[0]], color='orange', s=100, zorder=5, label=f'100% HistGBM (RMSE: {avg_rmses[0]:.2f})')
    plt.scatter([1], [avg_rmses[-1]], color='purple', s=100, zorder=5, label=f'100% Ridge (RMSE: {avg_rmses[-1]:.2f})')
    
    # Calculate exact RMSE for 0.65 weight
    chosen_w = 0.65
    chosen_fold_rmses = []
    for result in fold_results:
        fold_pred = chosen_w * result['ridge_preds'] + (1 - chosen_w) * result['hist_preds']
        chosen_fold_rmses.append(root_mean_squared_error(result['y_valid'], fold_pred))
    chosen_rmse = np.mean(chosen_fold_rmses)
    
    plt.scatter([chosen_w], [chosen_rmse], color='red', s=100, zorder=5)

    plt.title('Model Blend Optimization: Average 5-Fold RMSE', fontsize=14, pad=15)
    plt.xlabel('Weight assigned to Ridge Model (1.0 = 100% Ridge, 0.0 = 100% HistGBM)', fontsize=12)
    plt.ylabel('Average 5-Fold RMSE (Lower is Better)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    textstr = f'Chosen Blend RMSE: {chosen_rmse:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
            
    plt.tight_layout()
    
    out_path = Path("plots/blend_weight_optimization.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {out_path}")
    print(f"Exact calculated RMSE for 0.65 Ridge / 0.35 HistGBM: {chosen_rmse:.4f}")
    
if __name__ == '__main__':
    main()
