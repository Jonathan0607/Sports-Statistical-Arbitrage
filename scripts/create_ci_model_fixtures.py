"""Self-contained CI model fixture seeder for gitignored binary model weights.

Intentionally avoids importing `src.*` runtime dependencies (such as `arch` or `copulas`)
so `docker-smoke-test` can generate valid 28-feature artifacts using minimal packages
(`numpy`, `pandas`, `xgboost`, `lightgbm`) before running `docker build`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from typing import Dict, List

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

FEATURE_SCHEMA_VERSION = "1.0.0"

TARGET_METRICS: Dict[str, str] = {
    "pts": "points",
    "reb": "rebounds",
    "ast": "assists",
    "fg3m": "three_pointers_made",
    "stl": "steals",
    "blk": "blocks",
    "tov": "turnovers",
}

CATEGORICAL_ID_COLUMNS = ("player_id", "team_id", "opponent_team_id")

FEATURE_COLUMNS: List[str] = [
    "roll_min",
    "roll_stat",
    "roll_usg",
    "days_rest",
    "is_back_to_back",
    "opp_def_rating",
    "opp_pace_factor",
    "opp_pos_pts_allowed",
    "vacated_usage",
    "roll_potential_assists",
    "roll_touches",
    "roll_passes_made",
    "roll_time_of_possession_sec",
    "roll_defender_distance",
    "is_home_game",
    "pace_differential",
    "roll_stat_var",
    "roll_residual",
    "roll_ts_pct",
    "roll_reb_rate",
    "roll_ast_rate",
    "roll_shorthanded_points",
    "roll_plus_minus",
    "roll_usg_efficiency",
    "player_id",
    "team_id",
    "opponent_team_id",
    "player_tier",
]


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def seed_ci_models(
    model_dir: str = "models/wf_fold2",
    force: bool = False,
    seed_ci_registry: bool = False,
    registry_dir: str = "models/registry",
) -> None:
    os.makedirs(model_dir, exist_ok=True)
    sentinel = os.path.join(model_dir, "global_xpts_xgb.json")
    seeded = False
    if os.path.exists(sentinel) and not force:
        print(f"Model artifacts already exist in {model_dir}; skipping synthetic CI seed.")
    else:
        rng = np.random.default_rng(42)
        n = 60
        data = {}
        player_cats = ["201939", "2544", "1628983"]
        team_cats = ["14", "2", "1610612744"]
        for col in FEATURE_COLUMNS:
            if col == "player_id":
                data[col] = pd.Categorical(
                    [player_cats[i % len(player_cats)] for i in range(n)],
                    categories=player_cats,
                )
            elif col in ("team_id", "opponent_team_id"):
                data[col] = pd.Categorical(
                    [team_cats[i % len(team_cats)] for i in range(n)],
                    categories=team_cats,
                )
            elif col == "player_tier":
                data[col] = np.array([i % 4 for i in range(n)], dtype=float)
            else:
                data[col] = rng.normal(10.0, 2.0, size=n).astype(float)

        X = pd.DataFrame(data, columns=FEATURE_COLUMNS)
        y = rng.normal(0.0, 1.5, size=n).astype(float)

        for target_key, stat_col in TARGET_METRICS.items():
            dmat_xgb = xgb.DMatrix(X, label=y, enable_categorical=True)
            xgb_booster = xgb.train(
                {
                    "objective": "reg:absoluteerror",
                    "max_depth": 2,
                    "learning_rate": 0.1,
                    "tree_method": "hist",
                    "seed": 42,
                },
                dmat_xgb,
                num_boost_round=4,
            )
            xgb_booster.save_model(os.path.join(model_dir, f"global_x{target_key}_xgb.json"))
            xgb_booster.save_model(os.path.join(model_dir, f"global_x{target_key}.json"))

            dtrain = lgb.Dataset(X, label=y, categorical_feature=list(CATEGORICAL_ID_COLUMNS))
            lgb_model = lgb.train(
                {
                    "objective": "regression_l1",
                    "num_leaves": 7,
                    "min_data_in_leaf": 5,
                    "verbose": -1,
                },
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
        seeded = True
        print(f"Seeded synthetic 28-feature CI model fixtures under {model_dir}.")

    if seed_ci_registry and seeded:
        os.makedirs(os.path.join(registry_dir, "releases"), exist_ok=True)
        checksums = {}
        for target_key in TARGET_METRICS:
            for fname in (
                f"global_x{target_key}_xgb.json",
                f"global_x{target_key}_lgb.txt",
                f"global_x{target_key}_meta.json",
                f"global_x{target_key}.json",
            ):
                fpath = os.path.join(model_dir, fname)
                if os.path.isfile(fpath):
                    checksums[fname] = _sha256(fpath)
        champ_path = os.path.join(registry_dir, "champion.json")
        if os.path.exists(champ_path):
            with open(champ_path, "r") as f:
                champ = json.load(f)
            champ["model_dir"] = model_dir
            champ["artifact_checksums"] = checksums
            champ["notes"] = "CI synthetic model fixtures aligned with champion schema"
            with open(champ_path, "w") as f:
                json.dump(champ, f, indent=2)
                f.write("\n")
            print(f"Synced CI fixture SHA-256 checksums into {champ_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed lightweight 28-feature model fixtures for CI")
    parser.add_argument("--model-dir", default="models/wf_fold2")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed-ci-registry", action="store_true")
    parser.add_argument("--registry-dir", default="models/registry")
    args = parser.parse_args()
    seed_ci_models(
        model_dir=args.model_dir,
        force=args.force,
        seed_ci_registry=args.seed_ci_registry,
        registry_dir=args.registry_dir,
    )
