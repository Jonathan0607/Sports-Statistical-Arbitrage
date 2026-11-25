import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import sqlalchemy
import os
from dotenv import load_dotenv

from src.features import ContextualFeatureEngine
from src.train_global_model import FEATURE_COLUMNS, assign_player_tier, get_tier_alpha, TIER_ORDER

def run_audit():
    load_dotenv()
    db_uri = os.getenv("POSTGRES_URI", "postgresql+psycopg2://postgres:password@localhost:5432/sports_db")
    if db_uri.startswith("postgres://"):
        db_uri = db_uri.replace("postgres://", "postgresql+psycopg2://", 1)
        
    engine = sqlalchemy.create_engine(db_uri)
    fe = ContextualFeatureEngine(db_uri=db_uri)

    # 1. Get WNBA player ids
    query = """
    SELECT DISTINCT g.player_id 
    FROM player_game_logs g
    JOIN games gm ON g.game_id = gm.game_id
    WHERE gm.sport = 'WNBA'
    """
    with engine.connect() as conn:
        res = conn.execute(sqlalchemy.text(query)).fetchall()
    player_ids = [str(r[0]) for r in res]
    
    print(f"Found {len(player_ids)} WNBA players")
    all_dfs = []
    
    # 2. Build dataset
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
            ('potential_assists', 0.0), ('touches', 0.0), ('passes_made', 0.0), 
            ('time_of_possession_sec', 0.0), ('avg_defender_distance_ft', 4.5),
            ('shorthanded_points', 0.0), ('pace_differential', 0.0)
        ]:
            if col not in df_p.columns:
                df_p[col] = default
            else:
                df_p[col] = df_p[col].fillna(default)
                
        df_p['roll_potential_assists'] = df_p['potential_assists'].shift(1).ewm(alpha=tier_alpha, adjust=False).mean().fillna(0.0)
        df_p['roll_touches'] = df_p['touches'].shift(1).ewm(alpha=tier_alpha, adjust=False).mean().fillna(0.0)
        df_p['roll_passes_made'] = df_p['passes_made'].shift(1).ewm(alpha=tier_alpha, adjust=False).mean().fillna(0.0)
        df_p['roll_time_of_possession_sec'] = df_p['time_of_possession_sec'].shift(1).ewm(alpha=tier_alpha, adjust=False).mean().fillna(0.0)
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
        
        for stat_col in ['pts', 'reb', 'ast']:
            if stat_col not in df_p.columns:
                df_p[stat_col] = 0.0
                
        all_dfs.append(df_p)

    from src.train_global_model import TARGET_METRICS, apply_bayesian_shrinkage
    from src.models import get_default_priors

    cached_priors = get_default_priors()
    
    logger = None # Just to avoid logger undefined if we don't import it

    print(f"Final WNBA feature matrix player map generated.")

    metrics = ['pts', 'reb', 'ast']
    folds = ['wnba_wf_fold1', 'wnba_wf_fold2']
    
    overall_shap = {col: 0.0 for col in FEATURE_COLUMNS}
    
    for fold in folds:
        for target_key, stat_col in TARGET_METRICS.items():
            if target_key not in metrics:
                continue
                
            xgb_path = f"models/{fold}/global_x{target_key}_xgb.json"
            if not os.path.exists(xgb_path):
                print(f"Skipping {xgb_path}, not found.")
                continue
                
            metric_dfs = []
            for df_p in all_dfs:
                df = df_p.copy()
                if stat_col not in df.columns:
                    continue
                t_alpha = get_tier_alpha(TIER_ORDER[int(df['player_tier'].iloc[0])] if 'player_tier' in df.columns and not df['player_tier'].isna().all() else 'fringe')
                df['roll_stat'] = df.groupby('player_id')[stat_col].shift(1).ewm(alpha=t_alpha, adjust=False).mean()
                df['roll_stat_var'] = df.groupby('player_id')[stat_col].shift(1).ewm(alpha=t_alpha, adjust=False).var().fillna(0.0)
                
                prior_per_36 = 15.0 # default fallback
                
                df['roll_stat_per_36'] = (df['roll_stat'] / (df['roll_min'] + 1e-5)) * 36.0
                cum_min = df.groupby('player_id')['minutes_played'].cumsum()
                df['bayesian_roll_per_36'] = [
                    apply_bayesian_shrinkage(rs_36, cm, prior_per_36, K=500.0)
                    for rs_36, cm in zip(df['roll_stat_per_36'], cum_min)
                ]
                df['bayesian_roll_stat'] = (df['bayesian_roll_per_36'] / 36.0) * df['roll_min']
                df['target_residual'] = df[stat_col] - df['bayesian_roll_stat']
                df['roll_residual'] = df.groupby('player_id')['target_residual'].shift(1).ewm(alpha=t_alpha, adjust=False).mean().fillna(0.0)
                
                metric_dfs.append(df)
                
            if not metric_dfs:
                continue
                
            metric_df = pd.concat(metric_dfs, ignore_index=True)
            for col in FEATURE_COLUMNS:
                metric_df[col] = pd.to_numeric(metric_df[col], errors='coerce').astype(float)
                
            metric_train = metric_df.dropna(subset=FEATURE_COLUMNS).copy()
            X = metric_train[FEATURE_COLUMNS]
            print(f"Evaluating {target_key} on {len(X)} samples for {fold}")
            
            model = xgb.XGBRegressor()
            model.load_model(xgb_path)
            
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X)
            
            mean_abs_shap = np.abs(shap_vals).mean(axis=0)
            for i, col in enumerate(FEATURE_COLUMNS):
                overall_shap[col] += mean_abs_shap[i]
                
    for col in overall_shap:
        overall_shap[col] /= (len(folds) * len(metrics))
        
    sorted_shap = sorted(overall_shap.items(), key=lambda x: x[1], reverse=True)
    
    md_content = "# WNBA Feature Importance Audit (SHAP)\n\n"
    md_content += "Aggregated mean absolute SHAP values across all Walk-Forward folds and metrics (Points, Rebounds, Assists).\n\n"
    md_content += "| Rank | Feature | Mean Abs SHAP | Contribution % |\n"
    md_content += "|---|---|---|---|\n"
    
    total_shap = sum([x[1] for x in sorted_shap])
    for i, (feat, val) in enumerate(sorted_shap):
        pct = (val / total_shap) * 100 if total_shap > 0 else 0
        md_content += f"| {i+1} | `{feat}` | {val:.4f} | {pct:.2f}% |\n"
        
    out_path = "/Users/kingebenezer/.gemini/antigravity-ide/brain/45e5ba73-ae09-494c-a071-83c6f76c3f1e/wnba_feature_audit.md"
    with open(out_path, "w") as f:
        f.write(md_content)
        
    print(f"Saved {out_path}")

if __name__ == "__main__":
    run_audit()
