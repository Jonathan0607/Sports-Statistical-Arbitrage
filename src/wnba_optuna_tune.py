import os
import json
import optuna
import pandas as pd
import numpy as np
import xgboost as xgb
import sqlalchemy
from dotenv import load_dotenv
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

from src.features import ContextualFeatureEngine
from src.train_global_model import WNBA_FEATURE_COLUMNS, assign_player_tier, get_tier_alpha, TIER_ORDER, TARGET_METRICS, apply_bayesian_shrinkage

def build_wnba_data():
    load_dotenv()
    db_uri = os.getenv("POSTGRES_URI", "postgresql+psycopg2://postgres:password@localhost:5432/sports_db")
    if db_uri.startswith("postgres://"):
        db_uri = db_uri.replace("postgres://", "postgresql+psycopg2://", 1)
        
    engine = sqlalchemy.create_engine(db_uri)
    fe = ContextualFeatureEngine(db_uri=db_uri, league='WNBA')

    query = """
    SELECT DISTINCT g.player_id 
    FROM player_game_logs g
    JOIN games gm ON g.game_id = gm.game_id
    WHERE gm.sport = 'WNBA'
    """
    with engine.connect() as conn:
        res = conn.execute(sqlalchemy.text(query)).fetchall()
    player_ids = [str(r[0]) for r in res]
    
    print(f"Building features for {len(player_ids)} WNBA players...")
    all_dfs = []
    
    for pid in player_ids:
        df_p = fe.build_player_feature_matrix(pid, pd.DataFrame())
        if df_p.empty or len(df_p) < 5:
            continue
            
        df_p['usage_rate'] = ((df_p['points'] + 1.5 * df_p['assists'] + df_p['turnovers']) / (df_p['minutes_played'] * 2.0 + 1e-5))
        df_p['usage_rate'] = df_p['usage_rate'].clip(0.05, 0.45)
        
        avg_min = float(df_p['minutes_played'].mean())
        avg_usg = float(df_p['usage_rate'].mean())
        tier_name = assign_player_tier(avg_min, avg_usg)
        tier_alpha = get_tier_alpha(tier_name)
        df_p['player_tier'] = TIER_ORDER.index(tier_name)
        
        df_p['roll_min'] = df_p['minutes_played'].shift(1).ewm(alpha=tier_alpha, adjust=False).mean()
        df_p['roll_usg'] = df_p['usage_rate'].shift(1).ewm(alpha=tier_alpha, adjust=False).mean()

        for col, default in [
            ('time_of_possession_sec', 0.0), ('avg_defender_distance_ft', 4.5),
            ('shorthanded_points', 0.0), ('pace_differential', 0.0)
        ]:
            if col not in df_p.columns:
                df_p[col] = default
            else:
                df_p[col] = df_p[col].fillna(default)
                
        df_p['roll_defender_distance'] = df_p['avg_defender_distance_ft'].shift(1).ewm(alpha=tier_alpha, adjust=False).mean().fillna(4.5)
        df_p['roll_shorthanded_points'] = df_p['shorthanded_points'].shift(1).ewm(alpha=tier_alpha, adjust=False).mean().fillna(0.0)
        
        df_p['ts_pct'] = df_p['points'] / (2 * (df_p['field_goals_attempted'] + 0.44 * df_p['free_throws_attempted']) + 1e-5)
        df_p['reb_rate'] = (df_p['rebounds'] / (df_p['minutes_played'] + 1e-5)) * 36.0
        df_p['ast_rate'] = (df_p['assists'] / (df_p['minutes_played'] + 1e-5)) * 36.0
        df_p['usg_efficiency'] = df_p['points'] / ((df_p['usage_rate'] * df_p['minutes_played']) + 1e-5)

        df_p['roll_ts_pct'] = df_p['ts_pct'].shift(1).ewm(alpha=tier_alpha, adjust=False).mean()
        df_p['roll_reb_rate'] = df_p['reb_rate'].shift(1).ewm(alpha=tier_alpha, adjust=False).mean()
        df_p['roll_ast_rate'] = df_p['ast_rate'].shift(1).ewm(alpha=tier_alpha, adjust=False).mean()
        df_p['roll_plus_minus'] = df_p['plus_minus'].shift(1).ewm(alpha=tier_alpha, adjust=False).mean()
        df_p['roll_usg_efficiency'] = df_p['usg_efficiency'].shift(1).ewm(alpha=tier_alpha, adjust=False).mean()
        df_p['roll_fga'] = df_p['field_goals_attempted'].shift(1).ewm(alpha=tier_alpha, adjust=False).mean() if 'field_goals_attempted' in df_p.columns else 0.0
        df_p['roll_fta'] = df_p['free_throws_attempted'].shift(1).ewm(alpha=tier_alpha, adjust=False).mean() if 'free_throws_attempted' in df_p.columns else 0.0
        df_p['roll_expected_points'] = 2.0 * df_p['roll_ts_pct'] * (df_p['roll_fga'] + 0.44 * df_p['roll_fta'])
        
        for stat_col in TARGET_METRICS.values():
            if stat_col not in df_p.columns:
                df_p[stat_col] = 0.0
                
        all_dfs.append(df_p)

    return all_dfs

def run_optuna_tuning():
    all_dfs = build_wnba_data()
    
    # We will tune on points, and use the same params for all metrics 
    # to avoid overly complex config.
    target_key = 'pts'
    stat_col = TARGET_METRICS[target_key]
    
    metric_dfs = []
    for df_p in all_dfs:
        df = df_p.copy()
        if stat_col not in df.columns:
            continue
            
        t_alpha = get_tier_alpha(TIER_ORDER[int(df['player_tier'].iloc[0])] if 'player_tier' in df.columns and not df['player_tier'].isna().all() else 'fringe')
        df['roll_stat'] = df.groupby('player_id')[stat_col].shift(1).ewm(alpha=t_alpha, adjust=False).mean()
        df['roll_stat_var'] = df.groupby('player_id')[stat_col].shift(1).ewm(alpha=t_alpha, adjust=False).var().fillna(0.0)
        
        prior_per_36 = 15.0
        df['roll_stat_per_36'] = (df['roll_stat'] / (df['roll_min'] + 1e-5)) * 36.0
        cum_min = df.groupby('player_id')['minutes_played'].cumsum()
        df['bayesian_roll_per_36'] = [
            apply_bayesian_shrinkage(rs_36, cm, prior_per_36, K=500.0)
            for rs_36, cm in zip(df['roll_stat_per_36'], cum_min)
        ]
        df['bayesian_roll_stat'] = (df['bayesian_roll_per_36'] / 36.0) * df['roll_min']
        if stat_col == 'points':
            df['target_residual'] = (df[stat_col] / (df['minutes_played'] + 1e-5)) - (df['roll_stat'] / (df['roll_min'] + 1e-5))
        else:
            df['target_residual'] = df[stat_col] - df['roll_stat']
            
        df['roll_residual'] = df.groupby('player_id')['target_residual'].shift(1).ewm(alpha=t_alpha, adjust=False).mean().fillna(0.0)
        
        metric_dfs.append(df)
        
    metric_df = pd.concat(metric_dfs, ignore_index=True)
    if 'start_time' in metric_df.columns:
        metric_df['start_time'] = pd.to_datetime(metric_df['start_time'], utc=True)
        metric_df = metric_df.sort_values('start_time').reset_index(drop=True)
        
    for col in WNBA_FEATURE_COLUMNS + ['target_residual']:
        metric_df[col] = pd.to_numeric(metric_df[col], errors='coerce').astype(float)
        
    train_df = metric_df.dropna(subset=WNBA_FEATURE_COLUMNS + ['target_residual']).copy()
    X = train_df[WNBA_FEATURE_COLUMNS]
    y = train_df['target_residual'].values
    
    print(f"Starting WNBA Optuna Tuning on {len(X)} samples with {X.shape[1]} pruned features.")

    def objective(trial):
        # Heavy regularization for small dataset
        param = {
            "objective": "reg:absoluteerror",
            "eval_metric": "mae",
            "tree_method": "hist",
            "max_depth": trial.suggest_int("max_depth", 2, 4),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 50.0, 300.0, log=True) if stat_col == 'points' else trial.suggest_float("reg_lambda", 10.0, 100.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 10.0, 100.0, log=True) if stat_col == 'points' else trial.suggest_float("reg_alpha", 1.0, 20.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 0.8),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.8),
            "min_child_weight": trial.suggest_int("min_child_weight", 5, 30),
            "n_estimators": 150,
            "random_state": 42
        }

        tscv = TimeSeriesSplit(n_splits=5)
        maes = []
        
        for train_index, valid_index in tscv.split(X):
            X_tr, X_val = X.iloc[train_index], X.iloc[valid_index]
            y_tr, y_val = y[train_index], y[valid_index]
            
            model = xgb.XGBRegressor(**param)
            model.fit(X_tr, y_tr, verbose=False)
            
            preds = model.predict(X_val)
            
            if stat_col == 'points':
                val_df = train_df.iloc[valid_index]
                predicted_abs = val_df['roll_stat'].values + (preds * val_df['roll_min'].values)
                maes.append(mean_absolute_error(val_df[stat_col].values, predicted_abs))
            else:
                maes.append(mean_absolute_error(y_val, preds))
            
        return np.mean(maes)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)
    
    best_params = study.best_params
    best_params["n_estimators"] = 150
    
    print(f"Best params: {best_params}")
    print(f"Best CV MAE: {study.best_value}")
    
    # Save candidate config
    config = {
        "pts": {
            "xgb_params": best_params,
            "lgb_params": {
                "objective": "regression_l1",
                "metric": "mae",
                "boosting_type": "gbdt",
                "learning_rate": best_params.get("learning_rate", 0.05),
                "num_leaves": 15,
                "feature_fraction": best_params.get("colsample_bytree", 0.8),
                "bagging_fraction": best_params.get("subsample", 0.8),
                "bagging_freq": 1,
                "lambda_l1": best_params.get("reg_alpha", 0.0),
                "lambda_l2": best_params.get("reg_lambda", 0.0),
                "random_state": 42,
                "verbose": -1
            }
        }
    }
    
    os.makedirs("config", exist_ok=True)
    with open("config/wnba_model_config_candidate.json", "w") as f:
        json.dump(config, f, indent=4)
        
    print("Saved candidate config to config/wnba_model_config_candidate.json")

if __name__ == "__main__":
    run_optuna_tuning()
