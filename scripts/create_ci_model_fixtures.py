"""Seeds lightweight synthetic 28-feature artifacts for CI when gitignored model weights are absent."""

from __future__ import annotations

import json
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_contract import CATEGORICAL_ID_COLUMNS, FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION, TARGET_METRICS


def seed_ci_models(model_dir: str = "models/wf_fold2", force: bool = False) -> None:
    os.makedirs(model_dir, exist_ok=True)
    sentinel = os.path.join(model_dir, "global_xpts_xgb.json")
    if os.path.exists(sentinel) and not force:
        print(f"Model artifacts already exist in {model_dir}; skipping synthetic CI seed.")
        return

    rng = np.random.default_rng(42)
    n = 60
    data = {}
    player_cats = ["201939", "2544", "1628983"]
    team_cats = ["14", "2", "1610612744"]
    for col in FEATURE_COLUMNS:
        if col == "player_id":
            data[col] = pd.Categorical([player_cats[i % len(player_cats)] for i in range(n)], categories=player_cats)
        elif col in ("team_id", "opponent_team_id"):
            data[col] = pd.Categorical([team_cats[i % len(team_cats)] for i in range(n)], categories=team_cats)
        elif col == "player_tier":
            data[col] = np.array([i % 4 for i in range(n)], dtype=float)
        else:
            data[col] = rng.normal(10.0, 2.0, size=n).astype(float)

    X = pd.DataFrame(data, columns=FEATURE_COLUMNS)
    y = rng.normal(0.0, 1.5, size=n).astype(float)

    for target_key, stat_col in TARGET_METRICS.items():
        xgb_model = xgb.XGBRegressor(
            n_estimators=4,
            max_depth=2,
            learning_rate=0.1,
            enable_categorical=True,
            tree_method="hist",
            random_state=42,
        )
        xgb_model.fit(X, y)
        xgb_model.save_model(os.path.join(model_dir, f"global_x{target_key}_xgb.json"))
        xgb_model.save_model(os.path.join(model_dir, f"global_x{target_key}.json"))

        dtrain = lgb.Dataset(X, label=y, categorical_feature=list(CATEGORICAL_ID_COLUMNS))
        lgb_model = lgb.train(
            {"objective": "regression_l1", "num_leaves": 7, "min_data_in_leaf": 5, "verbose": -1},
            dtrain,
            num_boost_round=4,
        )
        lgb_model.save_model(os.path.join(model_dir, f"global_x{target_key}_lgb.txt"))

        meta = {
            "target": target_key,
            "stat_column": stat_col,
            "target_type": "residual",
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "feature_columns": FEATURE_COLUMNS,
            "train_seasons": ["2023-24", "2024-25"],
            "intercept": 0.0,
            "weights": [0.5, 0.5],
            "features": ["xgb_residual", "lgb_residual"],
            "alpha": 0.35,
            "n_splits": 5,
            "training_samples": n,
        }
        with open(os.path.join(model_dir, f"global_x{target_key}_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    print(f"Seeded synthetic 28-feature CI model fixtures under {model_dir}.")


if __name__ == "__main__":
    seed_ci_models()
