import os
import pandas as pd
import numpy as np
import sqlalchemy
import xgboost as xgb
import lightgbm as lgb
from joblib import Parallel, delayed
from src.features import ContextualFeatureEngine
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FastEvaluator")

db_uri = "postgresql+psycopg2://postgres:password@localhost:5432/sports_db"
engine = sqlalchemy.create_engine(db_uri)


def load_models(fold_dir):
    models = {}
    for stat in ['pts', 'reb', 'ast']:
        models[stat] = {}
        try:
            bst = xgb.XGBRegressor()
            bst.load_model(f"{fold_dir}/global_x{stat}_xgb.json")
            models[stat]['xgb'] = bst
        except: pass
        try:
            models[stat]['lgb'] = lgb.Booster(model_file=f"{fold_dir}/global_x{stat}_lgb.txt")
        except: pass
        try:
            with open(f"{fold_dir}/global_x{stat}_meta.json", "r") as f:
                models[stat]['meta'] = json.load(f)
        except: pass
    return models

def evaluate_fold(test_seasons, fold_dir):
    logger.info(f"Evaluating {test_seasons} with models from {fold_dir}")
    models = load_models(fold_dir)
    
    # Get all players who played in test seasons
    seasons_str = "('" + "','".join(test_seasons) + "')"
    query = f"SELECT DISTINCT g.player_id FROM player_game_logs g JOIN games gm ON g.game_id = gm.game_id WHERE gm.season IN {seasons_str}"
    with engine.connect() as conn:
        players = pd.read_sql(sqlalchemy.text(query), conn)['player_id'].tolist()
        
    logger.info(f"Found {len(players)} players in test seasons. Building matrices...")
    
    TARGET_METRICS = {"points": "pts", "rebounds": "reb", "assists": "ast"}
    
    # We will build the exact same feature matrix as training!
    from src.models import get_default_priors, apply_bayesian_shrinkage
    cached_priors = get_default_priors()
    with engine.connect() as conn:
        pos_res = conn.execute(sqlalchemy.text("SELECT master_player_id, position FROM players")).fetchall()
    player_positions = {str(row[0]): row[1] for row in pos_res}

    def process_player(pid):
        try:
            global _fe
            if '_fe' not in globals():
                _fe = ContextualFeatureEngine(db_uri=db_uri)
            df_p = _fe.build_player_feature_matrix(pid, pd.DataFrame())
            if df_p.empty or len(df_p) < 5: return None
            
            df_p['usage_rate'] = ((df_p['points'] + 1.5 * df_p['assists'] + df_p['turnovers']) / (df_p['minutes_played'] * 2.0 + 1e-5)).clip(0.05, 0.45)
            df_p['roll_min'] = df_p['minutes_played'].shift(1).ewm(alpha=0.3, adjust=False).mean()
            df_p['roll_usg'] = df_p['usage_rate'].shift(1).ewm(alpha=0.3, adjust=False).mean()

            for col, default in [
                ('potential_assists', 0.0), ('touches', 0.0), ('passes_made', 0.0), 
                ('time_of_possession_sec', 0.0), ('avg_defender_distance_ft', 4.5),
                ('shorthanded_points', 0.0), ('pace_differential', 0.0)
            ]:
                if col not in df_p.columns: df_p[col] = default
                else: df_p[col] = df_p[col].fillna(default)
                    
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
            
            # Filter for test seasons
            df_test = df_p[df_p['season'].isin(test_seasons)].copy()
            if df_test.empty: return None
            
            return df_p # Return full to keep rolling context, filter later
        except: return None
        
    all_dfs = Parallel(n_jobs=8)(delayed(process_player)(pid) for pid in players)
    all_dfs = [df for df in all_dfs if df is not None]
    
    logger.info(f"Compiled {len(all_dfs)} player matrices. Evaluating...")
    results = []
    
    for stat_col, target_key in TARGET_METRICS.items():
        if target_key not in models: continue
        
        stat_dfs = []
        for df_p in all_dfs:
            df = df_p.copy()
            if stat_col not in df.columns: continue
            
            df['roll_stat'] = df[stat_col].shift(1).ewm(alpha=0.3, adjust=False).mean()
            df['roll_stat_var'] = df[stat_col].shift(1).ewm(alpha=0.3, adjust=False).var().fillna(0.0)
            
            position = player_positions.get(str(df['player_id'].iloc[0]), "SG")
            prior_per_36 = cached_priors.get(stat_col, {}).get(position, 15.0)
            
            df['roll_stat_per_36'] = (df['roll_stat'] / (df['roll_min'] + 1e-5)) * 36.0
            cum_min = df['minutes_played'].cumsum()
            
            df['bayesian_roll_per_36'] = [apply_bayesian_shrinkage(rs_36, cm, prior_per_36, K=500.0) for rs_36, cm in zip(df['roll_stat_per_36'], cum_min)]
            df['bayesian_roll_stat'] = (df['bayesian_roll_per_36'] / 36.0) * df['roll_min']
            
            df['target_residual'] = df[stat_col] - df['bayesian_roll_stat']
            df['roll_residual'] = df['target_residual'].shift(1).ewm(alpha=0.3, adjust=False).mean().fillna(0.0)
            
            stat_dfs.append(df)
            
        full_df = pd.concat(stat_dfs, ignore_index=True)
        test_df = full_df[full_df['season'].isin(test_seasons)].copy()
        test_df = test_df.dropna(subset=['target_residual', stat_col])
        
        if test_df.empty: continue
        
        pred_cols = [
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
        X = test_df[pred_cols].copy()
        # Permutation Test: Scramble the categorical IDs
        X["player_id"] = np.random.permutation(X["player_id"].values)
        X["team_id"] = np.random.permutation(X["team_id"].values)
        X["opponent_team_id"] = np.random.permutation(X["opponent_team_id"].values)
        for col in ['player_id', 'team_id', 'opponent_team_id']:
            X[col] = X[col].astype('category')
            
        xgb_model = models[target_key].get('xgb')
        lgb_model = models[target_key].get('lgb')
        meta_data = models[target_key].get('meta', {})
        
        xgb_res = xgb_model.predict(X) if xgb_model else np.zeros(len(X))
        lgb_res = lgb_model.predict(X) if lgb_model else xgb_res
        
        w = meta_data.get('weights', [0.5, 0.5])
        intercept = meta_data.get('intercept', 0.0)
        
        ewma_baseline = test_df['bayesian_roll_stat'].values
        
        if len(w) == 3:
            meta_pred = ewma_baseline + intercept + (w[0] * xgb_res) + (w[1] * lgb_res) + (w[2] * ewma_baseline)
        else:
            meta_pred = ewma_baseline + intercept + (w[0] * xgb_res) + (w[1] * lgb_res)
            
        meta_pred = np.maximum(0.001, meta_pred)
        
        alpha = meta_data.get('alpha', 0.1)
        var_i = meta_pred + alpha * (meta_pred ** 2)
        n_nb = np.full(len(meta_pred), 1.0 / alpha).reshape(-1, 1)
        p_nb = (meta_pred / var_i).reshape(-1, 1)
        
        sims = np.random.negative_binomial(n=n_nb, p=p_nb, size=(len(meta_pred), 10000))
        sims = np.sort(sims, axis=1)
        actuals = test_df[stat_col].values.reshape(-1, 1)
        
        mae_sim = np.mean(np.abs(sims - actuals), axis=1)
        n_iters = 10000
        diffs = sims[:, 1:] - sims[:, :-1]
        weights = np.arange(1, n_iters) * np.arange(n_iters - 1, 0, -1)
        expected_diff = np.sum(diffs * weights, axis=1) / (n_iters * n_iters)
        crps = mae_sim - expected_diff
        
        res_df = pd.DataFrame({
            'stat_category': target_key,
            'actual_value': test_df[stat_col].values,
            'meta_prediction': meta_pred,
            'ewma_baseline': ewma_baseline,
            'crps': crps
        })
        results.append(res_df)
        
    final_res = pd.concat(results, ignore_index=True)
    
    print(f"\n==========================================================================================")
    print(f"📊  FAST VECTORIZED EVALUATION  📊")
    print(f"==========================================================================================")
    print(f"Time Range / Seasons: {test_seasons}")
    print(f"Total Box Scores Evaluated: {len(final_res)}")
    print(f"------------------------------------------------------------------------------------------")
    print(f"{'Category':<16} | {'Count':<6} | {'Meta MAE':<9} | {'EWMA MAE':<9} | {'Simulated CRPS'}")
    print(f"------------------------------------------------------------------------------------------")

    for stat, group in final_res.groupby('stat_category'):
        meta_mae = np.mean(np.abs(group['actual_value'] - group['meta_prediction']))
        ewma_mae = np.mean(np.abs(group['actual_value'] - group['ewma_baseline']))
        sim_crps = np.mean(group['crps'])
        print(f"{stat:<16} | {len(group):<6} | {meta_mae:<9.3f} | {ewma_mae:<9.3f} | {sim_crps:<15.3f}")

    overall_meta_mae = np.mean(np.abs(final_res["actual_value"] - final_res["meta_prediction"]))
    overall_ewma_mae = np.mean(np.abs(final_res["actual_value"] - final_res["ewma_baseline"]))
    overall_crps = np.mean(final_res["crps"])
    print(f"------------------------------------------------------------------------------------------")
    print(f"{'OVERALL':<16} | {len(final_res):<6} | {overall_meta_mae:<9.3f} | {overall_ewma_mae:<9.3f} | {overall_crps:<15.3f}")
    print(f"==========================================================================================\n")

if __name__ == "__main__":
    evaluate_fold(["2025-26"], "models/wf_fold2")
