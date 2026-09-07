import os
import sys
import logging
import numpy as np
import pandas as pd
import sqlalchemy
import xgboost as xgb
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


def load_feature_matrix_for_tuning(target_key: str = "pts", db_uri: str = None) -> tuple:
    """
    Replicates the data-fetching and preprocessing pipeline from train_global_model.py
    to build the exact feature matrix using continuous vacated_usage.
    Returns (X, y) sorted chronologically.
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
                # Add usage_rate and basic rolling averages
                df_p["usage_rate"] = (
                    (df_p["points"] + 1.5 * df_p["assists"] + df_p["turnovers"])
                    / (df_p["minutes_played"] * 2.0 + 1e-5)
                ).clip(0.05, 0.45)
                df_p["roll_min"] = df_p["minutes_played"].shift(1).rolling(3).mean()
                df_p["roll_usg"] = df_p["usage_rate"].shift(1).rolling(3).mean()

                # Dynamic Feature Column Swapping: roll_stat uses target stat column
                if stat_col not in df_p.columns:
                    df_p[stat_col] = 0.0

                position = player_positions.get(pid, "SG")
                prior_per_36 = get_positional_prior(stat_col, position, cached_priors=cached_priors)

                df_p["roll_stat_raw"] = df_p[stat_col].shift(1).rolling(3).mean()
                df_p["cum_min"] = df_p["minutes_played"].shift(1).cumsum().fillna(0.0)
                df_p["roll_stat_per_36"] = (df_p["roll_stat_raw"] / (df_p["roll_min"] + 1e-5)) * 36

                # Apply Bayesian shrinkage to match inference-time features
                df_p["bayesian_roll_stat_per_36"] = df_p.apply(
                    lambda row: apply_bayesian_shrinkage(row["roll_stat_per_36"], row["cum_min"], prior_per_36, K=500.0),
                    axis=1,
                )
                df_p["roll_stat"] = (df_p["bayesian_roll_stat_per_36"] / 36.0) * df_p["roll_min"]

                all_dfs.append(df_p)
        except Exception as e:
            logger.error(f"Error compiling features for player {pid}: {e}")

    if not all_dfs:
        logger.warning("No data retrieved from DB. Creating synthetic feature matrix for standalone execution.")
        # Create synthetic deterministic matrix for demonstration / offline tuning
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
            stat_col: np.random.poisson(22.0, n_samples).astype(float),
        })
        train_df = syn_df
    else:
        df = pd.concat(all_dfs, ignore_index=True)
        train_df = df.dropna(subset=[
            "roll_min", "roll_stat", "roll_usg", stat_col,
            "days_rest", "is_back_to_back", "opp_def_rating", "opp_pace_factor",
            "opp_pos_pts_allowed", "vacated_usage", "start_time"
        ]).copy()

    # Sort strictly chronologically to enforce Point-in-Time order
    if "start_time" in train_df.columns:
        train_df["start_time"] = pd.to_datetime(train_df["start_time"])
        train_df = train_df.sort_values("start_time").reset_index(drop=True)

    feature_cols = [
        "roll_min", "roll_stat", "roll_usg",
        "days_rest", "is_back_to_back", "opp_def_rating", "opp_pace_factor",
        "opp_pos_pts_allowed", "vacated_usage"
    ]

    X = train_df[feature_cols].values
    y = train_df[stat_col].values

    return X, y


def tune_xgboost_hyperparameters(target_key: str = "pts", n_trials: int = 25, db_uri: str = None) -> optuna.Study:
    """
    Runs Bayesian Hyperparameter Optimization using Optuna and TimeSeriesSplit
    forward-chaining cross validation on Poisson XGBoost Regressors.
    """
    logger.info(f"Initiating Bayesian Hyperparameter Tuning for target '{target_key}'...")
    X, y = load_feature_matrix_for_tuning(target_key=target_key, db_uri=db_uri)

    logger.info(f"Loaded feature matrix: {X.shape[0]} samples, {X.shape[1]} features.")

    if len(X) < 15:
        raise ValueError(f"Insufficient samples ({len(X)}) for 5-fold TimeSeriesSplit.")

    def objective(trial: optuna.Trial) -> float:
        # Suggest hyperparameters
        max_depth = trial.suggest_int("max_depth", 3, 7)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.1, log=True)
        subsample = trial.suggest_float("subsample", 0.6, 1.0)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0)
        reg_alpha = trial.suggest_float("reg_alpha", 0.0, 5.0)
        reg_lambda = trial.suggest_float("reg_lambda", 0.0, 5.0)

        # Enforce forward-chaining cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        fold_rmses = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            model = xgb.XGBRegressor(
                objective="count:poisson",
                n_estimators=120,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                random_state=42,
                verbosity=0,
            )

            model.fit(X_train_fold, y_train_fold)
            preds = model.predict(X_val_fold)
            rmse = float(np.sqrt(np.mean((y_val_fold - preds) ** 2)))
            fold_rmses.append(rmse)

        return float(np.mean(fold_rmses))

    # Disable default Optuna verbosity to keep console output clean
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(direction="minimize", study_name=f"xgb_{target_key}_tuning")
    study.optimize(objective, n_trials=n_trials)

    print("\n" + "=" * 65)
    print(f"🎯 OPTUNA BAYESIAN HYPERPARAMETER OPTIMIZATION: {target_key.upper()} 🎯")
    print("=" * 65)
    print(f"Total Trials Executed:       {len(study.trials)}")
    print(f"Best Cross-Validation RMSE: {study.best_value:.4f}")
    print("-" * 65)
    print("Optimal Hyperparameters:")
    for param_name, param_val in study.best_params.items():
        if isinstance(param_val, float):
            print(f"  • {param_name:<20}: {param_val:.5f}")
        else:
            print(f"  • {param_name:<20}: {param_val}")
    print("=" * 65 + "\n")

    return study


if __name__ == "__main__":
    tune_xgboost_hyperparameters(target_key="pts", n_trials=20)
