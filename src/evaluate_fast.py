import json
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import sqlalchemy
import xgboost as xgb
from joblib import Parallel, delayed

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_contract import (
    FEATURE_COLUMNS,
    FEATURE_SCHEMA_VERSION,
    PLAYER_TIERS,
    TIER_ORDER,
    assign_player_tier,
    compute_causal_player_features,
    compute_target_rolling_features,
    extract_xgb_categories_from_json,
    get_active_feature_columns,
    get_tier_alpha,
    predict_meta_ensemble,
)
from src.features import ContextualFeatureEngine
from src.models import apply_bayesian_shrinkage, get_default_priors

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FastEvaluator")

db_uri = os.getenv("POSTGRES_URI", "postgresql+psycopg2://postgres:password@localhost:5432/sports_db")
if db_uri.startswith("postgres://"):
    db_uri = db_uri.replace("postgres://", "postgresql+psycopg2://", 1)


def load_models(fold_dir: str) -> dict:
    models = {}
    for stat in ["pts", "reb", "ast"]:
        models[stat] = {}
        xgb_path = os.path.join(fold_dir, f"global_x{stat}_xgb.json")
        if not os.path.exists(xgb_path):
            xgb_path = os.path.join(fold_dir, f"global_x{stat}.json")
        lgb_path = os.path.join(fold_dir, f"global_x{stat}_lgb.txt")
        meta_path = os.path.join(fold_dir, f"global_x{stat}_meta.json")

        if os.path.exists(xgb_path):
            try:
                bst = xgb.XGBRegressor()
                bst.load_model(xgb_path)
                models[stat]["xgb"] = bst
                models[stat]["xgb_categories"] = extract_xgb_categories_from_json(xgb_path)
            except Exception as exc:
                logger.warning(f"Failed loading xgb for {stat}: {exc}")
        if os.path.exists(lgb_path):
            try:
                models[stat]["lgb"] = lgb.Booster(model_file=lgb_path)
            except Exception as exc:
                logger.warning(f"Failed loading lgb for {stat}: {exc}")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as f:
                    models[stat]["meta"] = json.load(f)
            except Exception as exc:
                logger.warning(f"Failed loading meta for {stat}: {exc}")
    return models


def _compute_crps_batch(
    meta_pred: np.ndarray,
    actuals: np.ndarray,
    alpha: float,
    n_iters: int = 10000,
    rng_seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(rng_seed)
    var_i = meta_pred + alpha * (meta_pred ** 2)
    n_nb = np.full(len(meta_pred), 1.0 / alpha).reshape(-1, 1)
    p_nb = (meta_pred / var_i).reshape(-1, 1)

    sims = rng.negative_binomial(n=n_nb, p=p_nb, size=(len(meta_pred), n_iters))
    sims = np.sort(sims, axis=1)
    actuals_col = np.asarray(actuals, dtype=float).reshape(-1, 1)

    mae_sim = np.mean(np.abs(sims - actuals_col), axis=1)
    diffs = sims[:, 1:] - sims[:, :-1]
    weights = np.arange(1, n_iters) * np.arange(n_iters - 1, 0, -1)
    expected_diff = np.sum(diffs * weights, axis=1) / (n_iters * n_iters)
    return mae_sim - expected_diff


def evaluate_dataframe_vectorized(
    player_dfs: List[pd.DataFrame],
    models: dict,
    test_seasons: List[str],
    player_positions: Optional[Dict[str, str]] = None,
    cached_priors: Optional[dict] = None,
    league: str = "NBA",
    compute_crps: bool = True,
    crps_n_sims: int = 10000,
    rng_seed: int = 42,
) -> Tuple[pd.DataFrame, List[dict]]:
    """Vectorized evaluation over pre-compiled player histories using the unified feature contract."""
    league_key = (league or "NBA").upper()
    positions = player_positions or {}
    priors = cached_priors or get_default_priors(league=league_key)
    target_metrics = {"points": "pts", "rebounds": "reb", "assists": "ast"}

    causal_dfs = []
    for df_p in player_dfs:
        if df_p is None or df_p.empty or len(df_p) < 2:
            continue
        featured = compute_causal_player_features(df_p, league=league_key)
        if "season" in featured.columns and test_seasons:
            if featured[featured["season"].isin(test_seasons)].empty:
                continue
        causal_dfs.append(featured)

    if not causal_dfs:
        return pd.DataFrame(), []

    results = []
    for stat_col, target_key in target_metrics.items():
        if target_key not in models or not models[target_key]:
            continue

        active_cols = get_active_feature_columns(league=league_key, stat_col=stat_col)
        stat_dfs = []
        for df_p in causal_dfs:
            if stat_col not in df_p.columns:
                continue
            pid = str(df_p["player_id"].iloc[0]) if "player_id" in df_p.columns else "0"
            pos = positions.get(pid, "SG")
            df_stat = compute_target_rolling_features(
                df_p,
                stat_col=stat_col,
                position=pos,
                cached_priors=priors,
                league=league_key,
            )
            stat_dfs.append(df_stat)

        if not stat_dfs:
            continue

        full_df = pd.concat(stat_dfs, ignore_index=True)
        if test_seasons and "season" in full_df.columns:
            test_df = full_df[full_df["season"].isin(test_seasons)].copy()
        else:
            test_df = full_df.copy()

        required = active_cols + ["target_residual", stat_col, "bayesian_roll_stat"]
        test_df = test_df.dropna(subset=required).copy()
        if test_df.empty:
            continue

        sort_cols = [c for c in ("player_id", "start_time", "game_id") if c in test_df.columns]
        if sort_cols:
            test_df = test_df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
        else:
            test_df = test_df.reset_index(drop=True)

        xgb_model = models[target_key].get("xgb")
        lgb_model = models[target_key].get("lgb")
        meta_data = models[target_key].get("meta", {})
        xgb_cats = models[target_key].get("xgb_categories")

        ewma_baseline = (
            test_df["roll_stat"].to_numpy(dtype=float)
            if league_key == "WNBA"
            else test_df["bayesian_roll_stat"].to_numpy(dtype=float)
        )
        preds = predict_meta_ensemble(
            test_df,
            active_cols,
            xgb_model=xgb_model,
            lgb_model=lgb_model,
            meta_data=meta_data,
            ewma_baseline=ewma_baseline,
            league=league_key,
            stat_col=stat_col,
            roll_min=test_df["roll_min"].to_numpy(dtype=float),
            xgb_categories=xgb_cats,
        )
        meta_pred = preds["meta_prediction"]
        actuals = test_df[stat_col].to_numpy(dtype=float)

        if compute_crps:
            alpha = float(meta_data.get("alpha", 0.1))
            crps = _compute_crps_batch(meta_pred, actuals, alpha, n_iters=crps_n_sims, rng_seed=rng_seed)
        else:
            crps = np.zeros(len(meta_pred), dtype=float)

        res_df = pd.DataFrame(
            {
                "player_id": test_df["player_id"].astype(str).values if "player_id" in test_df.columns else "",
                "game_id": test_df["game_id"].values if "game_id" in test_df.columns else np.arange(len(test_df)),
                "start_time": test_df["start_time"].values if "start_time" in test_df.columns else pd.NaT,
                "stat_category": target_key,
                "actual_value": actuals,
                "meta_prediction": meta_pred,
                "ewma_baseline": ewma_baseline,
                "xgb_residual": preds["xgb_residual"],
                "lgb_residual": preds["lgb_residual"],
                "crps": crps,
            }
        )
        results.append(res_df)

    if not results:
        return pd.DataFrame(), []

    final_res = pd.concat(results, ignore_index=True)
    summary = []
    for stat, group in final_res.groupby("stat_category"):
        meta_mae = float(np.mean(np.abs(group["actual_value"] - group["meta_prediction"])))
        ewma_mae = float(np.mean(np.abs(group["actual_value"] - group["ewma_baseline"])))
        sim_crps = float(np.mean(group["crps"]))
        summary.append(
            {
                "Category": stat,
                "Count": int(len(group)),
                "Meta MAE": meta_mae,
                "EWMA MAE": ewma_mae,
                "Sim CRPS": sim_crps,
            }
        )

    overall_meta_mae = float(np.mean(np.abs(final_res["actual_value"] - final_res["meta_prediction"])))
    overall_ewma_mae = float(np.mean(np.abs(final_res["actual_value"] - final_res["ewma_baseline"])))
    overall_crps = float(np.mean(final_res["crps"]))
    summary.append(
        {
            "Category": "OVERALL",
            "Count": int(len(final_res)),
            "Meta MAE": overall_meta_mae,
            "EWMA MAE": overall_ewma_mae,
            "Sim CRPS": overall_crps,
        }
    )
    return final_res, summary


def evaluate_fold(test_seasons: List[str], fold_dir: str, league: str = "NBA"):
    logger.info(f"Evaluating {test_seasons} with models from {fold_dir} (schema={FEATURE_SCHEMA_VERSION})")
    engine = sqlalchemy.create_engine(db_uri)
    models = load_models(fold_dir)

    with engine.connect() as conn:
        stmt = sqlalchemy.text(
            "SELECT DISTINCT g.player_id FROM player_game_logs g "
            "JOIN games gm ON g.game_id = gm.game_id WHERE gm.season IN :test_seasons"
        ).bindparams(sqlalchemy.bindparam("test_seasons", expanding=True))
        players = pd.read_sql(stmt, conn, params={"test_seasons": list(test_seasons)})["player_id"].tolist()
        cutoff_stmt = sqlalchemy.text(
            "SELECT MAX(start_time) FROM games WHERE sport = :league AND season IN :test_seasons"
        ).bindparams(sqlalchemy.bindparam("test_seasons", expanding=True))
        max_dt = conn.execute(cutoff_stmt, {"league": league, "test_seasons": list(test_seasons)}).scalar()
        eval_cutoff = (pd.to_datetime(max_dt, utc=True) + pd.Timedelta(seconds=1)).isoformat() if max_dt is not None else None
        pos_res = conn.execute(sqlalchemy.text("SELECT master_player_id, position FROM players")).fetchall()

    logger.info(f"Found {len(players)} players in test seasons. Building matrices (cutoff={eval_cutoff})...")
    cached_priors = get_default_priors(league=league)
    player_positions = {str(row[0]): row[1] for row in pos_res}

    def process_player(pid):
        try:
            global _fe
            if "_fe" not in globals():
                _fe = ContextualFeatureEngine(db_uri=db_uri, league=league)
            df_p = _fe.build_player_feature_matrix(pid, pd.DataFrame(), target_date=eval_cutoff)
            if df_p.empty or len(df_p) < 5:
                return None
            if df_p[df_p["season"].isin(test_seasons)].empty:
                return None
            return df_p
        except Exception:
            return None

    all_dfs = Parallel(n_jobs=8)(delayed(process_player)(pid) for pid in players)
    all_dfs = [df for df in all_dfs if df is not None]

    logger.info(f"Compiled {len(all_dfs)} player matrices. Evaluating...")
    final_res, summary = evaluate_dataframe_vectorized(
        all_dfs,
        models=models,
        test_seasons=test_seasons,
        player_positions=player_positions,
        cached_priors=cached_priors,
        league=league,
    )

    print("\n==========================================================================================")
    print("📊  FAST VECTORIZED EVALUATION (POINT-IN-TIME PARITY)  📊")
    print("==========================================================================================")
    print(f"Time Range / Seasons: {test_seasons}")
    print(f"Total Box Scores Evaluated: {len(final_res)}")
    print("------------------------------------------------------------------------------------------")
    print(f"{'Category':<16} | {'Count':<6} | {'Meta MAE':<9} | {'EWMA MAE':<9} | {'Simulated CRPS'}")
    print("------------------------------------------------------------------------------------------")

    for row in summary:
        if row["Category"] == "OVERALL":
            print("------------------------------------------------------------------------------------------")
        print(
            f"{row['Category']:<16} | {row['Count']:<6} | {row['Meta MAE']:<9.3f} | "
            f"{row['EWMA MAE']:<9.3f} | {row['Sim CRPS']:<15.3f}"
        )
    print("==========================================================================================\n")
    return final_res, summary


if __name__ == "__main__":
    evaluate_fold(["2025-26"], "models/wf_fold2")
