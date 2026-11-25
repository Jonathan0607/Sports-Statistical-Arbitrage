# src/train_global_model.py
import json
import logging
import os
import sys

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import scipy.stats as stats
import sqlalchemy
import xgboost as xgb
from dotenv import load_dotenv
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_contract import (
    CATEGORICAL_ID_COLUMNS,
    FEATURE_COLUMNS,
    FEATURE_SCHEMA_VERSION,
    PLAYER_TIERS,
    TARGET_METRICS,
    TIER_ORDER,
    WNBA_FEATURE_COLUMNS,
    assign_player_tier,
    compute_causal_player_features,
    compute_target_rolling_features,
    get_active_feature_columns,
    get_tier_alpha,
    prepare_xgb_lgb_frames,
)
from src.features import ContextualFeatureEngine
from src.models import apply_bayesian_shrinkage, get_default_priors, get_positional_prior

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TrainGlobalModels")


def _iter_grouped_time_splits(df: pd.DataFrame, n_splits: int = 5):
    """Yields expanding-window (train_idx, val_idx) splits grouped by timestamp to avoid intra-day leakage."""
    if "start_time" not in df.columns or df["start_time"].nunique() <= n_splits:
        tscv = TimeSeriesSplit(n_splits=n_splits)
        yield from tscv.split(df)
        return

    unique_times = np.sort(df["start_time"].unique())
    tscv = TimeSeriesSplit(n_splits=n_splits)
    time_series = df["start_time"].to_numpy()
    for train_t_idx, val_t_idx in tscv.split(unique_times):
        train_times = set(unique_times[train_t_idx])
        val_times = set(unique_times[val_t_idx])
        train_idx = np.flatnonzero(np.isin(time_series, list(train_times)))
        val_idx = np.flatnonzero(np.isin(time_series, list(val_times)))
        if len(train_idx) > 0 and len(val_idx) > 0:
            yield train_idx, val_idx


def train_global_ensemble(
    target_keys: list = None,
    config_path: str = "config/stat_hyperparameters.json",
    train_seasons: list = None,
    model_dir: str = "models",
    league: str = "NBA",
):
    """
    Trains a Meta-Ensemble (Stacking) predictive layer across target statistics:
    1. Base Model 1: XGBoost Regressor
    2. Base Model 2: LightGBM Regressor
    3. Meta-Learner: Ridge Regression (alpha=1.0) trained on date-grouped OOF predictions
    4. Distributional Calibration: Optuna finds optimal NB dispersion (alpha) strictly on OOF residuals.
    """
    load_dotenv()
    db_uri = os.getenv("POSTGRES_URI", "postgresql://postgres:password@localhost:5432/sports_db")
    if db_uri.startswith("postgres://"):
        db_uri = db_uri.replace("postgres://", "postgresql+psycopg2://", 1)

    logger.info(f"Connecting to database: {db_uri}")
    engine = sqlalchemy.create_engine(db_uri)
    fe = ContextualFeatureEngine(db_uri=db_uri, league=league)

    logger.info("Extracting unique player IDs from PostgreSQL...")
    with engine.connect() as conn:
        if train_seasons:
            stmt = sqlalchemy.text(
                "SELECT DISTINCT g.player_id FROM player_game_logs g "
                "JOIN games gm ON g.game_id = gm.game_id "
                "WHERE gm.sport = :league AND gm.season IN :train_seasons"
            ).bindparams(sqlalchemy.bindparam("train_seasons", expanding=True))
            res = conn.execute(stmt, {"league": league, "train_seasons": list(train_seasons)}).fetchall()
            cutoff_stmt = sqlalchemy.text(
                "SELECT MAX(start_time) FROM games WHERE sport = :league AND season IN :train_seasons"
            ).bindparams(sqlalchemy.bindparam("train_seasons", expanding=True))
            max_dt = conn.execute(cutoff_stmt, {"league": league, "train_seasons": list(train_seasons)}).scalar()
            train_cutoff = (pd.to_datetime(max_dt, utc=True) + pd.Timedelta(seconds=1)).isoformat() if max_dt is not None else None
        else:
            stmt = sqlalchemy.text(
                "SELECT DISTINCT g.player_id FROM player_game_logs g "
                "JOIN games gm ON g.game_id = gm.game_id "
                "WHERE gm.sport = :league"
            )
            res = conn.execute(stmt, {"league": league}).fetchall()
            train_cutoff = None
    player_ids = [row[0] for row in res]
    logger.info(f"Found {len(player_ids)} players with game logs (train_seasons={train_seasons}, cutoff={train_cutoff}).")

    logger.info("Loading player positions...")
    with engine.connect() as conn:
        pos_res = conn.execute(sqlalchemy.text("SELECT master_player_id, position FROM players")).fetchall()
    player_positions = {row[0]: row[1] for row in pos_res}

    cached_priors = get_default_priors(league=league)

    logger.info("Compiling global feature matrix player-by-player...")
    all_player_dfs = {}

    for pid in player_ids:
        try:
            df_p = fe.build_player_feature_matrix(pid, pd.DataFrame(), target_date=train_cutoff)
            if train_seasons and not df_p.empty and "season" in df_p.columns:
                # Strictly wall off future test seasons BEFORE computing any career tiers or rolling features.
                df_p = df_p[df_p["season"].isin(train_seasons)].copy()
            if df_p.empty or len(df_p) < 5:
                continue

            df_p = compute_causal_player_features(df_p, league=league)
            for stat_col in TARGET_METRICS.values():
                if stat_col not in df_p.columns:
                    df_p[stat_col] = 0.0
            all_player_dfs[pid] = df_p
        except Exception as e:
            logger.error(f"Error compiling features for player {pid}: {e}")

    os.makedirs(model_dir, exist_ok=True)

    xgb_params = {
        "objective": "reg:absoluteerror",
        "eval_metric": "mae",
        "tree_method": "hist",
        "enable_categorical": True,
        "n_estimators": 150,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
    }

    lgb_params = {
        "objective": "regression_l1",
        "metric": "mae",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "random_state": 42,
        "verbose": -1,
    }

    custom_config = {}
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            custom_config = json.load(f)
        logger.info(f"Loaded custom model configuration from {config_path}")

    for target_key, stat_col in TARGET_METRICS.items():
        if target_keys is not None and target_key not in target_keys:
            continue

        logger.info(f"Preparing training data for target: {target_key} (col: {stat_col})")
        active_features = get_active_feature_columns(league=league, stat_col=stat_col)

        target_xgb_params = xgb_params.copy()
        target_lgb_params = lgb_params.copy()
        if not any(c in active_features for c in CATEGORICAL_ID_COLUMNS):
            target_xgb_params.pop("enable_categorical", None)

        if target_key in custom_config:
            if "xgb_params" in custom_config[target_key]:
                target_xgb_params.update(custom_config[target_key]["xgb_params"])
            if "lgb_params" in custom_config[target_key]:
                target_lgb_params.update(custom_config[target_key]["lgb_params"])
        else:
            if "xgb_params" in custom_config:
                target_xgb_params.update(custom_config["xgb_params"])
            if "lgb_params" in custom_config:
                target_lgb_params.update(custom_config["lgb_params"])

        all_dfs = []
        for pid, df_p in all_player_dfs.items():
            if stat_col not in df_p.columns:
                continue
            position = player_positions.get(df_p["player_id"].iloc[0], "SG")
            df = compute_target_rolling_features(
                df_p,
                stat_col=stat_col,
                position=position,
                cached_priors=cached_priors,
                league=league,
            )
            all_dfs.append(df)

        if not all_dfs:
            continue

        df = pd.concat(all_dfs, ignore_index=True)
        if "start_time" in df.columns:
            df["start_time"] = pd.to_datetime(df["start_time"], utc=True)
            df = df.sort_values(["start_time", "player_id"]).reset_index(drop=True)

        for col in active_features + ["target_residual", stat_col]:
            if col in CATEGORICAL_ID_COLUMNS:
                if col == "player_id":
                    df[col] = df[col].astype(str)
                else:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

        train_df = df.dropna(subset=active_features + ["target_residual", stat_col]).copy().reset_index(drop=True)
        if train_df.empty:
            logger.warning(f"No valid data remaining for {target_key} after NA drop.")
            continue

        for cat_col in CATEGORICAL_ID_COLUMNS:
            if cat_col in train_df.columns:
                if cat_col == "player_id":
                    train_df[cat_col] = train_df[cat_col].astype(str).astype("category")
                else:
                    train_df[cat_col] = train_df[cat_col].astype(int).astype("category")

        logger.info(f"Target {target_key} clean training logs: {len(train_df)}")
        if len(train_df) < 20:
            continue

        X_train = train_df[active_features]
        y_train = train_df["target_residual"].values

        n_splits = 5
        oof_xgb = []
        oof_lgb = []
        oof_y = []
        oof_indices = []

        logger.info(f"Generating OOF residual predictions via {n_splits}-split date-grouped TimeSeriesSplit for {target_key}...")
        for train_idx, val_idx in _iter_grouped_time_splits(train_df, n_splits=n_splits):
            X_f_train, X_f_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_f_train, y_f_val = y_train[train_idx], y_train[val_idx]

            es_split = max(1, int(len(X_f_train) * 0.85))
            X_es_train, X_es_val = X_f_train.iloc[:es_split], X_f_train.iloc[es_split:]
            y_es_train, y_es_val = y_f_train[:es_split], y_f_train[es_split:]

            fold_xgb = xgb.XGBRegressor(**{**target_xgb_params, "n_estimators": 500, "early_stopping_rounds": 20})
            fold_xgb.fit(X_es_train, y_es_train, eval_set=[(X_es_val, y_es_val)], verbose=False)
            pred_xgb_val = fold_xgb.predict(X_f_val)

            dtrain_f = lgb.Dataset(X_es_train, label=y_es_train)
            dval_f = lgb.Dataset(X_es_val, label=y_es_val, reference=dtrain_f)
            fold_lgb = lgb.train(
                target_lgb_params,
                dtrain_f,
                num_boost_round=500,
                valid_sets=[dval_f],
                callbacks=[lgb.early_stopping(20, verbose=False)],
            )
            pred_lgb_val = fold_lgb.predict(X_f_val)

            oof_xgb.extend(pred_xgb_val)
            oof_lgb.extend(pred_lgb_val)
            oof_y.extend(y_f_val)
            oof_indices.extend(val_idx.tolist())

        stacked_oof = np.column_stack([oof_xgb, oof_lgb])
        oof_y_arr = np.array(oof_y)

        meta_learner = Ridge(alpha=1.0)
        meta_learner.fit(stacked_oof, oof_y_arr)

        logger.info(f"Fitting final full-dataset XGBoost and LightGBM models for {target_key}...")
        final_xgb = xgb.XGBRegressor(**{**target_xgb_params, "n_estimators": 150})
        final_xgb.fit(X_train, y_train, verbose=False)

        dtrain_full = lgb.Dataset(X_train, label=y_train)
        final_lgb = lgb.train(target_lgb_params, dtrain_full, num_boost_round=500)

        # Calibrate Negative Binomial Dispersion (alpha) strictly on OOF predictions (no in-sample bias)
        oof_meta_residuals = meta_learner.predict(stacked_oof)
        oof_df = train_df.iloc[oof_indices]

        if league == "WNBA":
            if stat_col == "points":
                mu_vals = oof_df["roll_stat"].values + (oof_meta_residuals * oof_df["roll_min"].values)
            else:
                mu_vals = oof_df["roll_stat"].values + oof_meta_residuals
        else:
            mu_vals = oof_df["bayesian_roll_stat"].values + oof_meta_residuals

        mu_vals = np.clip(mu_vals, 0.001, None)
        true_vals = oof_df[stat_col].values

        def crps_objective(trial):
            alpha = trial.suggest_float("alpha", 0.01, 3.0)
            n_nb = 1.0 / alpha
            p_nb = 1.0 / (1.0 + alpha * mu_vals)
            if np.any(p_nb <= 0.0) or np.any(p_nb >= 1.0):
                return 1e9
            return float(-np.mean(stats.nbinom.logpmf(true_vals, n_nb, p_nb)))

        logger.info(f"Running Optuna calibration for {target_key} dispersion parameter (alpha)...")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize")
        study.optimize(crps_objective, n_trials=20)
        best_alpha = study.best_params["alpha"]
        logger.info(f"Target {target_key} Calibrated Dispersion (alpha): {best_alpha:.4f}")

        os.makedirs(model_dir, exist_ok=True)
        xgb_path = os.path.join(model_dir, f"global_x{target_key}_xgb.json")
        lgb_path = os.path.join(model_dir, f"global_x{target_key}_lgb.txt")
        meta_path = os.path.join(model_dir, f"global_x{target_key}_meta.json")
        compat_path = os.path.join(model_dir, f"global_x{target_key}.json")

        final_xgb.save_model(xgb_path)
        final_xgb.save_model(compat_path)
        final_lgb.save_model(lgb_path)

        meta_data = {
            "target": target_key,
            "stat_column": stat_col,
            "target_type": "residual",
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "feature_columns": active_features,
            "train_seasons": list(train_seasons) if train_seasons else None,
            "intercept": float(meta_learner.intercept_),
            "weights": [float(w) for w in meta_learner.coef_],
            "features": ["xgb_residual", "lgb_residual"],
            "alpha": best_alpha,
            "n_splits": n_splits,
            "training_samples": len(train_df),
        }
        with open(meta_path, "w") as f:
            json.dump(meta_data, f, indent=2)

        logger.info(f"Successfully saved residual ensemble artifacts for {target_key}")


if __name__ == "__main__":
    train_global_ensemble()
