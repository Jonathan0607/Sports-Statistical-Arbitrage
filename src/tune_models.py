import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import sqlalchemy
import xgboost as xgb
import lightgbm as lgb
import optuna
from dotenv import load_dotenv
from sklearn.model_selection import TimeSeriesSplit

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import ContextualFeatureEngine
from src.models import get_positional_prior, apply_bayesian_shrinkage, get_default_priors

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("TuneModels")

TARGET_METRICS = {
    "pts": "points",
    "reb": "rebounds",
    "ast": "assists",
    "fg3m": "three_pointers_made",
    "stl": "steals",
    "blk": "blocks",
    "tov": "turnovers",
}

FEATURE_COLUMNS = [
    "roll_min", "roll_stat", "roll_usg",
    "days_rest", "is_back_to_back", "opp_def_rating", "opp_pace_factor",
    "opp_pos_pts_allowed", "vacated_usage",
    "roll_potential_assists", "roll_touches", "roll_passes_made", 
    "roll_time_of_possession_sec", "roll_defender_distance",
    "is_home_game", "pace_differential", "roll_stat_var", "roll_residual",
    "roll_ts_pct", "roll_reb_rate", "roll_ast_rate", "roll_shorthanded_points",
    "roll_plus_minus", "roll_usg_efficiency"
]


def load_feature_matrix_for_tuning(target_key: str = "pts", db_uri: str = None, return_full: bool = False, train_seasons: list = None) -> tuple:
    """
    Replicates the data-fetching and preprocessing pipeline from train_global_model.py
    to build the exact feature matrix using continuous vacated_usage and tracking features.
    
    If return_full is False (legacy compatibility): returns (X, y_residual).
    If return_full is True: returns (X, y_residual, roll_stat, y_actual).
    """
    load_dotenv()
    db_uri = db_uri or os.getenv("POSTGRES_URI", "postgresql+psycopg2://postgres:password@localhost:5432/sports_db")
    if db_uri.startswith("postgres://"):
        db_uri = db_uri.replace("postgres://", "postgresql+psycopg2://", 1)

    logger.info(f"Connecting to database: {db_uri}")
    engine = sqlalchemy.create_engine(db_uri)
    fe = ContextualFeatureEngine(db_uri=db_uri)

    stat_col = TARGET_METRICS.get(target_key, "points")

    try:
        with engine.connect() as conn:
            if train_seasons:
                seasons_tuple = tuple(train_seasons)
                if len(seasons_tuple) == 1:
                    query = f"SELECT DISTINCT g.player_id FROM player_game_logs g JOIN games gm ON g.game_id = gm.game_id WHERE gm.season = '{seasons_tuple[0]}'"
                else:
                    query = f"SELECT DISTINCT g.player_id FROM player_game_logs g JOIN games gm ON g.game_id = gm.game_id WHERE gm.season IN {seasons_tuple}"
                res = conn.execute(sqlalchemy.text(query)).fetchall()
            else:
                res = conn.execute(sqlalchemy.text("SELECT DISTINCT player_id FROM player_game_logs")).fetchall()
        player_ids = [row[0] for row in res]
    except Exception as e:
        logger.warning(f"Could not load player IDs from database: {e}. Generating fallback data.")
        player_ids = []

    logger.info(f"Found {len(player_ids)} players with game logs.")

    # Pre-load positions
    try:
        with engine.connect() as conn:
            pos_res = conn.execute(sqlalchemy.text("SELECT master_player_id, position FROM players")).fetchall()
        player_positions = {row[0]: row[1] for row in pos_res}
    except Exception:
        player_positions = {}

    cached_priors = get_default_priors()

    all_dfs = []

    for pid in player_ids:
        try:
            df_p = fe.build_player_feature_matrix(pid, pd.DataFrame())
            if not df_p.empty and len(df_p) >= 5:
                if train_seasons:
                    df_p = df_p[df_p["season"].isin(train_seasons)].copy()
                    if df_p.empty or len(df_p) < 5:
                        continue
                
                # Add usage_rate and EWMA rolling averages matching train_global_model.py
                df_p["usage_rate"] = (
                    (df_p["points"] + 1.5 * df_p["assists"] + df_p["turnovers"])
                    / (df_p["minutes_played"] * 2.0 + 1e-5)
                ).clip(0.05, 0.45)
                df_p["roll_min"] = df_p.groupby("season")["minutes_played"].transform(
                    lambda s: s.shift(1).ewm(alpha=0.3, adjust=False).mean()
                )
                df_p["roll_usg"] = df_p.groupby("season")["usage_rate"].transform(
                    lambda s: s.shift(1).ewm(alpha=0.3, adjust=False).mean()
                )

                # Tracking metrics
                for col, default in [("potential_assists", 0.0), ("touches", 0.0), ("passes_made", 0.0), ("time_of_possession_sec", 0.0), ("avg_defender_distance_ft", 4.5), ("shorthanded_points", 0.0), ("plus_minus", 0.0), ("field_goals_attempted", 0.0), ("free_throws_attempted", 0.0)]:
                    if col not in df_p.columns:
                        df_p[col] = default
                    else:
                        df_p[col] = df_p[col].fillna(default)
                        
                df_p['ts_pct'] = df_p['points'] / (2 * (df_p['field_goals_attempted'] + 0.44 * df_p['free_throws_attempted']) + 1e-5)
                df_p['reb_rate'] = (df_p['rebounds'] / (df_p['minutes_played'] + 1e-5)) * 36.0
                df_p['ast_rate'] = (df_p['assists'] / (df_p['minutes_played'] + 1e-5)) * 36.0
                df_p['usg_efficiency'] = df_p['points'] / ((df_p['usage_rate'] * df_p['minutes_played']) + 1e-5)

                df_p["roll_potential_assists"] = df_p.groupby("season")["potential_assists"].transform(
                    lambda s: s.shift(1).ewm(alpha=0.3, adjust=False).mean()
                ).fillna(0.0)
                df_p["roll_touches"] = df_p.groupby("season")["touches"].transform(
                    lambda s: s.shift(1).ewm(alpha=0.3, adjust=False).mean()
                ).fillna(0.0)
                df_p["roll_passes_made"] = df_p.groupby("season")["passes_made"].transform(
                    lambda s: s.shift(1).ewm(alpha=0.3, adjust=False).mean()
                ).fillna(0.0)
                df_p["roll_time_of_possession_sec"] = df_p.groupby("season")["time_of_possession_sec"].transform(
                    lambda s: s.shift(1).ewm(alpha=0.3, adjust=False).mean()
                ).fillna(0.0)
                df_p["roll_defender_distance"] = df_p.groupby("season")["avg_defender_distance_ft"].transform(
                    lambda s: s.shift(1).ewm(alpha=0.3, adjust=False).mean()
                ).fillna(4.5)
                df_p['roll_shorthanded_points'] = df_p.groupby('season')['shorthanded_points'].transform(
                    lambda s: s.shift(1).ewm(alpha=0.3, adjust=False).mean()
                ).fillna(0.0)
                df_p['roll_plus_minus'] = df_p.groupby('season')['plus_minus'].transform(
                    lambda s: s.shift(1).ewm(alpha=0.3, adjust=False).mean()
                ).fillna(0.0)
                df_p['roll_ts_pct'] = df_p.groupby('season')['ts_pct'].transform(
                    lambda s: s.shift(1).ewm(alpha=0.3, adjust=False).mean()
                ).fillna(0.0)
                df_p['roll_reb_rate'] = df_p.groupby('season')['reb_rate'].transform(
                    lambda s: s.shift(1).ewm(alpha=0.3, adjust=False).mean()
                ).fillna(0.0)
                df_p['roll_ast_rate'] = df_p.groupby('season')['ast_rate'].transform(
                    lambda s: s.shift(1).ewm(alpha=0.3, adjust=False).mean()
                ).fillna(0.0)
                df_p['roll_usg_efficiency'] = df_p.groupby('season')['usg_efficiency'].transform(
                    lambda s: s.shift(1).ewm(alpha=0.3, adjust=False).mean()
                ).fillna(0.0)

                if stat_col not in df_p.columns:
                    df_p[stat_col] = 0.0

                position = player_positions.get(pid, "SG")
                prior_per_36 = get_positional_prior(stat_col, position, cached_priors=cached_priors)

                df_p["roll_stat_raw"] = df_p.groupby("season")[stat_col].transform(
                    lambda s: s.shift(1).ewm(alpha=0.3, adjust=False).mean()
                )
                df_p["cum_min"] = df_p.groupby("season")["minutes_played"].transform(
                    lambda s: s.shift(1).cumsum()
                ).fillna(0.0)
                df_p["roll_stat_per_36"] = (df_p["roll_stat_raw"] / (df_p["roll_min"] + 1e-5)) * 36

                # Apply Bayesian shrinkage to match inference-time features
                df_p["bayesian_roll_stat_per_36"] = df_p.apply(
                    lambda row: apply_bayesian_shrinkage(row["roll_stat_per_36"], row["cum_min"], prior_per_36, K=500.0),
                    axis=1,
                )
                df_p["roll_stat"] = (df_p["bayesian_roll_stat_per_36"] / 36.0) * df_p["roll_min"]
                df_p["target_residual"] = df_p[stat_col] - df_p["roll_stat"]
                
                # Contextual Rolling Target Variance
                df_p["roll_stat_var"] = df_p.groupby("season")[stat_col].transform(
                    lambda s: s.shift(1).ewm(alpha=0.3, adjust=False).var()
                ).fillna(0.0)
                
                # Contextual Rolling Residual Momentum
                df_p["roll_residual"] = df_p.groupby("season")["target_residual"].transform(
                    lambda s: s.shift(1).ewm(alpha=0.3, adjust=False).mean()
                ).fillna(0.0)

                all_dfs.append(df_p)
        except Exception as e:
            logger.error(f"Error compiling features for player {pid}: {e}")

    if not all_dfs:
        logger.warning("No data retrieved from DB. Creating synthetic feature matrix for standalone execution.")
        np.random.seed(42)
        n_samples = 150
        dates = pd.date_range("2025-10-20", periods=n_samples, freq="12h")
        syn_df = pd.DataFrame({
            "start_time": dates,
            "roll_min": np.random.uniform(20.0, 38.0, n_samples),
            "roll_stat": np.random.uniform(10.0, 32.0, n_samples),
            "roll_usg": np.random.uniform(0.15, 0.35, n_samples),
            "days_rest": np.random.choice([1.0, 2.0, 3.0], n_samples),
            "is_back_to_back": np.random.choice([0.0, 1.0], n_samples),
            "opp_def_rating": np.random.uniform(105.0, 120.0, n_samples),
            "opp_pace_factor": np.random.uniform(95.0, 105.0, n_samples),
            "opp_pos_pts_allowed": np.random.uniform(18.0, 26.0, n_samples),
            "vacated_usage": np.random.choice([0.0, 0.22, 0.30], n_samples),
            "roll_potential_assists": np.random.uniform(2.0, 8.0, n_samples),
            "roll_touches": np.random.uniform(25.0, 80.0, n_samples),
            "roll_passes_made": np.random.uniform(10.0, 40.0, n_samples),
            "roll_time_of_possession_sec": np.random.uniform(60.0, 200.0, n_samples),
            "roll_defender_distance": np.random.uniform(3.5, 5.5, n_samples),
            "is_home_game": np.random.choice([0.0, 1.0], n_samples),
            "pace_differential": np.random.uniform(-5.0, 5.0, n_samples),
            "roll_stat_var": np.random.uniform(1.0, 15.0, n_samples),
            "roll_residual": np.random.uniform(-3.0, 3.0, n_samples),
            "roll_ts_pct": np.random.uniform(0.45, 0.68, n_samples),
            "roll_reb_rate": np.random.uniform(4.0, 16.0, n_samples),
            "roll_ast_rate": np.random.uniform(2.0, 12.0, n_samples),
            "roll_shorthanded_points": np.random.uniform(0.0, 5.0, n_samples),
            "roll_plus_minus": np.random.uniform(-10.0, 10.0, n_samples),
            "roll_usg_efficiency": np.random.uniform(1.0, 3.5, n_samples),
            "player_id": np.random.choice([2544, 201939, 203999], n_samples).astype(float),
            "team_id": np.random.choice([14, 10, 8], n_samples).astype(float),
            "opponent_team_id": np.random.choice([2, 5, 21], n_samples).astype(float),
            "player_tier": np.random.choice([0.0, 1.0, 2.0, 3.0], n_samples),
            stat_col: np.random.poisson(22.0, n_samples).astype(float),
        })
        syn_df["target_residual"] = syn_df[stat_col] - syn_df["roll_stat"]
        train_df = syn_df
    else:
        df = pd.concat(all_dfs, ignore_index=True)
        required_cols = FEATURE_COLUMNS + ["target_residual", stat_col, "start_time"]
        train_df = df.dropna(subset=required_cols).copy()

    # Sort strictly chronologically to enforce Point-in-Time order
    if "start_time" in train_df.columns:
        train_df["start_time"] = pd.to_datetime(train_df["start_time"], utc=True)
        train_df = train_df.sort_values("start_time").reset_index(drop=True)

    for col in FEATURE_COLUMNS + ["target_residual", stat_col]:
        train_df[col] = pd.to_numeric(train_df[col], errors="coerce").astype(float)
    train_df = train_df.dropna(subset=FEATURE_COLUMNS + ["target_residual", stat_col]).reset_index(drop=True)

    X = train_df[FEATURE_COLUMNS].values
    y_residual = train_df["target_residual"].values
    roll_stat = train_df["roll_stat"].values
    y_actual = train_df[stat_col].values

    if return_full:
        return X, y_residual, roll_stat, y_actual
    return X, y_residual


def get_stat_specific_search_space(stat_key: str, trial: optuna.Trial) -> dict:
    """
    Returns suggested hyperparameter dictionary.
    For high-variance/low-signal counting stats (reb, ast), enforces aggressive regularization:
      - max_depth: 1 to 3 (very shallow stumps/trees)
      - reg_alpha: 5.0 to 100.0 (aggressive L1 sparsity)
      - reg_lambda: 10.0 to 200.0 (aggressive L2 shrinkage to 0)
      - learning_rate: 0.005 to 0.05
      - subsample: 0.5 to 0.9
      - colsample_bytree: 0.4 to 0.8
    For scoring/volume stats (pts):
      - max_depth: 3 to 6
      - reg_alpha: 0.1 to 5.0
      - reg_lambda: 0.5 to 10.0
      - learning_rate: 0.01 to 0.10
      - subsample: 0.6 to 1.0
      - colsample_bytree: 0.6 to 1.0
    """
    if stat_key in ("reb", "ast", "pts"):
        max_depth = trial.suggest_int("max_depth", 1, 3)
        learning_rate = trial.suggest_float("learning_rate", 0.005, 0.05, log=True)
        subsample = trial.suggest_float("subsample", 0.5, 0.9)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.4, 0.8)
        reg_alpha = trial.suggest_float("reg_alpha", 5.0, 100.0, log=True)
        reg_lambda = trial.suggest_float("reg_lambda", 15.0, 250.0, log=True)
    else:
        max_depth = trial.suggest_int("max_depth", 3, 6)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.10, log=True)
        subsample = trial.suggest_float("subsample", 0.6, 1.0)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0)
        reg_alpha = trial.suggest_float("reg_alpha", 0.1, 5.0, log=True)
        reg_lambda = trial.suggest_float("reg_lambda", 0.5, 10.0, log=True)

    return {
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "reg_alpha": reg_alpha,
        "reg_lambda": reg_lambda,
    }


def tune_xgboost_hyperparameters(target_key: str = "pts", n_trials: int = 25, db_uri: str = None, train_seasons: list = None) -> optuna.Study:
    """
    Runs Bayesian Hyperparameter Optimization using Optuna and TimeSeriesSplit
    forward-chaining cross-validation directly minimizing end-to-end prediction MAE:
    mae = mean(|actual - (roll_stat + predicted_residual)|).
    """
    logger.info(f"Initiating Bayesian Regularization Tuning for target '{target_key}'...")
    X, y_residual, roll_stat, y_actual = load_feature_matrix_for_tuning(
        target_key=target_key, db_uri=db_uri, return_full=True, train_seasons=train_seasons
    )

    logger.info(f"Loaded feature matrix: {X.shape[0]} samples, {X.shape[1]} features.")

    if len(X) < 15:
        raise ValueError(f"Insufficient samples ({len(X)}) for 5-fold TimeSeriesSplit.")

    def objective(trial: optuna.Trial) -> float:
        params = get_stat_specific_search_space(target_key, trial)

        tscv = TimeSeriesSplit(n_splits=5)
        fold_maes = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_res = y_residual[train_idx]
            y_val_act = y_actual[val_idx]
            roll_stat_val = roll_stat[val_idx]

            model = xgb.XGBRegressor(
                objective="reg:absoluteerror",
                n_estimators=120,
                max_depth=params["max_depth"],
                learning_rate=params["learning_rate"],
                subsample=params["subsample"],
                colsample_bytree=params["colsample_bytree"],
                reg_alpha=params["reg_alpha"],
                reg_lambda=params["reg_lambda"],
                random_state=42,
                verbosity=0,
            )

            model.fit(X_train_fold, y_train_res)
            pred_res = model.predict(X_val_fold)
            
            # End-to-end prediction = roll_stat + residual
            combined_pred = np.maximum(0.001, roll_stat_val + pred_res)
            mae = float(np.mean(np.abs(y_val_act - combined_pred)))
            fold_maes.append(mae)

        return float(np.mean(fold_maes))

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(direction="minimize", study_name=f"xgb_{target_key}_regularization")
    study.optimize(objective, n_trials=n_trials)

    print("\n" + "=" * 65)
    print(f"🎯 OPTUNA REGULARIZATION OPTIMIZATION: {target_key.upper()} 🎯")
    print("=" * 65)
    print(f"Total Trials Executed:       {len(study.trials)}")
    print(f"Best Cross-Validation End-to-End MAE: {study.best_value:.4f}")
    print("-" * 65)
    print("Optimal Hyperparameters:")
    for param_name, param_val in study.best_params.items():
        if isinstance(param_val, float):
            print(f"  • {param_name:<20}: {param_val:.5f}")
        else:
            print(f"  • {param_name:<20}: {param_val}")
    print("=" * 65 + "\n")

    return study


def run_stat_regularization_sweep(
    stats: list = None,
    n_trials: int = 25,
    output_path: str = "config/stat_hyperparameters.json",
    db_uri: str = None,
    train_seasons: list = None,
) -> dict:
    """
    Executes stat-specific hyperparameter sweeps and saves the results to a config file.
    """
    if stats is None:
        stats = ["reb", "ast"]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load existing configs if present to preserve other stats
    all_configs = {}
    if os.path.exists(output_path):
        try:
            with open(output_path, "r") as f:
                all_configs = json.load(f)
        except Exception:
            all_configs = {}

    for stat in stats:
        logger.info(f"Starting sweep for stat: {stat} with {n_trials} trials...")
        study = tune_xgboost_hyperparameters(target_key=stat, n_trials=n_trials, db_uri=db_uri, train_seasons=train_seasons)

        best_params = study.best_params
        best_mae = study.best_value

        xgb_cfg = {
            "n_estimators": 150,
            "max_depth": int(best_params["max_depth"]),
            "learning_rate": float(best_params["learning_rate"]),
            "colsample_bytree": float(best_params["colsample_bytree"]),
            "subsample": float(best_params["subsample"]),
            "reg_lambda": float(best_params["reg_lambda"]),
            "reg_alpha": float(best_params["reg_alpha"]),
            "objective": "reg:absoluteerror",
            "random_state": 42,
        }

        # Matched LightGBM parameters
        lgb_cfg = {
            "n_estimators": 150,
            "max_depth": int(best_params["max_depth"]),
            "learning_rate": float(best_params["learning_rate"]),
            "colsample_bytree": float(best_params["colsample_bytree"]),
            "subsample": float(best_params["subsample"]),
            "reg_lambda": float(best_params["reg_lambda"]),
            "reg_alpha": float(best_params["reg_alpha"]),
            "objective": "mean_absolute_error",
            "random_state": 42,
            "verbose": -1,
        }

        all_configs[stat] = {
            "target": stat,
            "stat_column": TARGET_METRICS.get(stat, stat),
            "best_cv_mae": float(best_mae),
            "xgb": xgb_cfg,
            "lgb": lgb_cfg,
        }

    with open(output_path, "w") as f:
        json.dump(all_configs, f, indent=2)

    logger.info(f"Successfully saved regularized hyperparameter configs to {output_path}")
    return all_configs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run stat-specific regularization Optuna sweep")
    parser.add_argument("--stats", nargs="+", default=["pts"], help="Stats to tune (default: pts)")
    parser.add_argument("--n-trials", type=int, default=25, help="Number of Optuna trials")
    parser.add_argument("--output", type=str, default="config/stat_hyperparameters.json", help="Path to output json")
    parser.add_argument("--train-seasons", nargs="+", default=None, help="Specific seasons to train on for nested optimization (e.g., '2023-24' '2024-25')")

    args = parser.parse_args()
    run_stat_regularization_sweep(stats=args.stats, n_trials=args.n_trials, output_path=args.output, train_seasons=args.train_seasons)
