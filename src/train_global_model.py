# src/train_global_model.py
import os
import sys
import json
import logging
import pandas as pd
import numpy as np
import sqlalchemy
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from dotenv import load_dotenv
import optuna
import scipy.stats as stats

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import ContextualFeatureEngine
from src.models import get_positional_prior, apply_bayesian_shrinkage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TrainGlobalModels")

TARGET_METRICS = {
    'pts': 'points',
    'reb': 'rebounds',
    'ast': 'assists',
    'fg3m': 'three_pointers_made',
    'stl': 'steals',
    'blk': 'blocks',
    'tov': 'turnovers'
}

FEATURE_COLUMNS = [
    'roll_min', 'roll_stat', 'roll_usg',
    'days_rest', 'is_back_to_back', 'opp_def_rating', 'opp_pace_factor',
    'opp_pos_pts_allowed', 'vacated_usage',
    'roll_potential_assists', 'roll_touches', 'roll_passes_made',
    'roll_time_of_possession_sec', 'roll_defender_distance',
    'is_home_game', 'pace_differential', 'roll_stat_var', 'roll_residual',
    'roll_ts_pct', 'roll_reb_rate', 'roll_ast_rate', 'roll_shorthanded_points',
    'roll_plus_minus', 'roll_usg_efficiency',
    'player_id', 'team_id', 'opponent_team_id'
]


def train_global_ensemble(
    target_keys: list = None,
    config_path: str = "config/stat_hyperparameters.json",
    train_seasons: list = None,
    model_dir: str = "models"
):
    """
    Trains a Meta-Ensemble (Stacking) predictive layer across all target statistics:
    1. Base Model 1: XGBoost Regressor
    2. Base Model 2: LightGBM Regressor
    3. Meta-Learner: Ridge Regression (alpha=1.0) trained on out-of-fold (OOF)
       predictions from a 5-split TimeSeriesSplit cross-validation loop.
    4. Distributional Calibration: Optuna finds optimal Negative Binomial dispersion (alpha).
    """
    load_dotenv()
    db_uri = os.getenv("POSTGRES_URI", "postgresql://postgres:password@localhost:5432/sports_db")
    if db_uri.startswith("postgres://"):
        db_uri = db_uri.replace("postgres://", "postgresql+psycopg2://", 1)
        
    logger.info(f"Connecting to database: {db_uri}")
    engine = sqlalchemy.create_engine(db_uri)
    fe = ContextualFeatureEngine(db_uri=db_uri)
    
    logger.info("Extracting unique player IDs from PostgreSQL...")
    with engine.connect() as conn:
        res = conn.execute(sqlalchemy.text("SELECT DISTINCT player_id FROM player_game_logs")).fetchall()
    player_ids = [row[0] for row in res]
    logger.info(f"Found {len(player_ids)} players with game logs.")
    
    logger.info("Loading player positions...")
    with engine.connect() as conn:
        pos_res = conn.execute(sqlalchemy.text("SELECT master_player_id, position FROM players")).fetchall()
    player_positions = {row[0]: row[1] for row in pos_res}
    
    from src.models import get_default_priors
    cached_priors = get_default_priors()
    
    logger.info("Compiling global feature matrix player-by-player...")
    all_player_dfs = {}
    
    for idx, pid in enumerate(player_ids):
        try:
            df_p = fe.build_player_feature_matrix(pid, pd.DataFrame())
            if not df_p.empty and len(df_p) >= 5:
                df_p['usage_rate'] = (
                    (df_p['points'] + 1.5 * df_p['assists'] + df_p['turnovers'])
                    / (df_p['minutes_played'] * 2.0 + 1e-5)
                )
                df_p['usage_rate'] = df_p['usage_rate'].clip(0.05, 0.45)
                df_p['roll_min'] = df_p['minutes_played'].shift(1).ewm(alpha=0.3, adjust=False).mean()
                df_p['roll_usg'] = df_p['usage_rate'].shift(1).ewm(alpha=0.3, adjust=False).mean()

                for col, default in [
                    ('potential_assists', 0.0), ('touches', 0.0), ('passes_made', 0.0), 
                    ('time_of_possession_sec', 0.0), ('avg_defender_distance_ft', 4.5),
                    ('shorthanded_points', 0.0), ('pace_differential', 0.0)
                ]:
                    if col not in df_p.columns:
                        df_p[col] = default
                    else:
                        df_p[col] = df_p[col].fillna(default)
                        
                df_p['roll_potential_assists'] = df_p['potential_assists'].shift(1).ewm(alpha=0.3, adjust=False).mean().fillna(0.0)
                df_p['roll_touches'] = df_p['touches'].shift(1).ewm(alpha=0.3, adjust=False).mean().fillna(0.0)
                df_p['roll_passes_made'] = df_p['passes_made'].shift(1).ewm(alpha=0.3, adjust=False).mean().fillna(0.0)
                df_p['roll_time_of_possession_sec'] = df_p['time_of_possession_sec'].shift(1).ewm(alpha=0.3, adjust=False).mean().fillna(0.0)
                df_p['roll_defender_distance'] = df_p['avg_defender_distance_ft'].shift(1).ewm(alpha=0.3, adjust=False).mean().fillna(4.5)
                df_p['roll_shorthanded_points'] = df_p['shorthanded_points'].shift(1).ewm(alpha=0.3, adjust=False).mean().fillna(0.0)
                
                df_p['ts_pct'] = df_p['points'] / (2 * (df_p['field_goals_attempted'] + 0.44 * df_p['free_throws_attempted']) + 1e-5)
                df_p['reb_rate'] = (df_p['rebounds'] / (df_p['minutes_played'] + 1e-5)) * 36.0
                df_p['ast_rate'] = (df_p['assists'] / (df_p['minutes_played'] + 1e-5)) * 36.0
                df_p['usg_efficiency'] = df_p['points'] / ((df_p['usage_rate'] * df_p['minutes_played']) + 1e-5)

                df_p['roll_ts_pct'] = df_p['ts_pct'].shift(1).ewm(alpha=0.3, adjust=False).mean()
                df_p['roll_reb_rate'] = df_p['reb_rate'].shift(1).ewm(alpha=0.3, adjust=False).mean()
                df_p['roll_ast_rate'] = df_p['ast_rate'].shift(1).ewm(alpha=0.3, adjust=False).mean()
                df_p['roll_plus_minus'] = df_p['plus_minus'].shift(1).ewm(alpha=0.3, adjust=False).mean()
                df_p['roll_usg_efficiency'] = df_p['usg_efficiency'].shift(1).ewm(alpha=0.3, adjust=False).mean()
                
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
        "random_state": 42
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
        "verbose": -1
    }

    for target_key, stat_col in TARGET_METRICS.items():
        logger.info(f"Preparing training data for target: {target_key} (col: {stat_col})")
        all_dfs = []
        for pid, df_p in all_player_dfs.items():
            df = df_p.copy()
            if stat_col not in df.columns:
                continue

            df['roll_stat'] = df.groupby('player_id')[stat_col].shift(1).ewm(alpha=0.3, adjust=False).mean()
            df['roll_stat_var'] = df.groupby('player_id')[stat_col].shift(1).ewm(alpha=0.3, adjust=False).var().fillna(0.0)
            
            # Bayesian shrinkage for residuals
            position = player_positions.get(df['player_id'].iloc[0], "SG")
            prior_per_36 = cached_priors.get(stat_col, {}).get(position, 15.0)
            
            df['roll_stat_per_36'] = (df['roll_stat'] / (df['roll_min'] + 1e-5)) * 36.0
            cum_min = df.groupby('player_id')['minutes_played'].cumsum()
            
            df['bayesian_roll_per_36'] = [
                apply_bayesian_shrinkage(rs_36, cm, prior_per_36, K=500.0)
                for rs_36, cm in zip(df['roll_stat_per_36'], cum_min)
            ]
            df['bayesian_roll_stat'] = (df['bayesian_roll_per_36'] / 36.0) * df['roll_min']
            
            df['target_residual'] = df[stat_col] - df['bayesian_roll_stat']
            df['roll_residual'] = df.groupby('player_id')['target_residual'].shift(1).ewm(alpha=0.3, adjust=False).mean().fillna(0.0)
            
            all_dfs.append(df)
            
        if not all_dfs:
            continue
            
        df = pd.concat(all_dfs, ignore_index=True)
        if 'start_time' in df.columns:
            df['start_time'] = pd.to_datetime(df['start_time'], utc=True)
            df = df.sort_values('start_time').reset_index(drop=True)

        for col in FEATURE_COLUMNS + ['target_residual', stat_col]:
            if col in ['player_id', 'team_id', 'opponent_team_id']:
                df[col] = df[col].astype('category')
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
        
        train_df = df.dropna(subset=FEATURE_COLUMNS + ['target_residual', stat_col]).copy()
        if train_df.empty:
            logger.warning(f"No valid data remaining for {target_key} after NA drop.")
            continue
            
        for cat_col in ['player_id', 'team_id', 'opponent_team_id']:
            train_df[cat_col] = train_df[cat_col].astype('category')
            
        logger.info(f"Target {target_key} clean training logs: {len(train_df)}")
        if len(train_df) < 20:
            continue
            
        X_train = train_df[FEATURE_COLUMNS]
        y_train = train_df['target_residual'].values

        n_splits = 5
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        oof_xgb = []
        oof_lgb = []
        oof_y = []

        logger.info(f"Generating OOF residual predictions via {n_splits}-split TimeSeriesSplit for {target_key}...")
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
            X_f_train, X_f_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_f_train, y_f_val = y_train[train_idx], y_train[val_idx]

            fold_xgb = xgb.XGBRegressor(**xgb_params)
            fold_xgb.fit(X_f_train, y_f_train)
            pred_xgb_val = fold_xgb.predict(X_f_val)

            dtrain_f = lgb.Dataset(X_f_train, label=y_f_train)
            fold_lgb = lgb.train(lgb_params, dtrain_f, num_boost_round=150)
            pred_lgb_val = fold_lgb.predict(X_f_val)

            oof_xgb.extend(pred_xgb_val)
            oof_lgb.extend(pred_lgb_val)
            oof_y.extend(y_f_val)

        stacked_oof = np.column_stack([oof_xgb, oof_lgb])
        oof_y_arr = np.array(oof_y)

        meta_learner = Ridge(alpha=1.0)
        meta_learner.fit(stacked_oof, oof_y_arr)

        logger.info(f"Fitting final full-dataset XGBoost and LightGBM models for {target_key}...")
        final_xgb = xgb.XGBRegressor(**xgb_params)
        final_xgb.fit(X_train, y_train)

        dtrain_full = lgb.Dataset(X_train, label=y_train)
        final_lgb = lgb.train(lgb_params, dtrain_full, num_boost_round=150)
        
        # ------------------------------------------------------------------
        # Calibrate Negative Binomial Dispersion (alpha) via Optuna
        # ------------------------------------------------------------------
        full_preds_xgb = final_xgb.predict(X_train)
        full_preds_lgb = final_lgb.predict(X_train)
        full_stacked = np.column_stack([full_preds_xgb, full_preds_lgb])
        full_meta_residuals = meta_learner.predict(full_stacked)
        
        mu_vals = train_df['roll_stat'].values + full_meta_residuals
        mu_vals = np.clip(mu_vals, 0.001, None)
        true_vals = train_df[stat_col].values
        
        def crps_objective(trial):
            alpha = trial.suggest_float('alpha', 0.01, 3.0)
            n_nb = 1.0 / alpha
            p_nb = 1.0 / (1.0 + alpha * mu_vals)
            
            # Penalize mathematically impossible states or extreme overdispersion
            if np.any(p_nb <= 0.0) or np.any(p_nb >= 1.0):
                return 1e9
                
            nll = -np.mean(stats.nbinom.logpmf(true_vals, n_nb, p_nb))
            return nll

        logger.info(f"Running Optuna calibration for {target_key} dispersion parameter (alpha)...")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize")
        study.optimize(crps_objective, n_trials=20)
        best_alpha = study.best_params['alpha']
        logger.info(f"Target {target_key} Calibrated Dispersion (alpha): {best_alpha:.4f}")

        xgb_path = f"models/global_x{target_key}_xgb.json"
        lgb_path = f"models/global_x{target_key}_lgb.txt"
        meta_path = f"models/global_x{target_key}_meta.json"
        compat_path = f"models/global_x{target_key}.json"

        final_xgb.save_model(xgb_path)
        final_xgb.save_model(compat_path)
        final_lgb.save_model(lgb_path)

        meta_data = {
            "target": target_key,
            "stat_column": stat_col,
            "target_type": "residual",
            "intercept": float(meta_learner.intercept_),
            "weights": [float(w) for w in meta_learner.coef_],
            "features": ["xgb_residual", "lgb_residual"],
            "alpha": best_alpha,
            "n_splits": n_splits,
            "training_samples": len(train_df)
        }
        with open(meta_path, "w") as f:
            json.dump(meta_data, f, indent=2)

        logger.info(f"Successfully saved residual ensemble artifacts for {target_key}")

if __name__ == "__main__":
    train_global_ensemble()
