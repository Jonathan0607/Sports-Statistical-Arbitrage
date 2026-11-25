"""Experiment 14: Clean Leak-Free Ablation Study on Feature Contract v1.0.0.

Re-audits architectural decisions from Experiments 6-12 across strictly point-in-time
Fold 1 (2023-24 -> 2024-25) and Fold 2 (2023-24, 2024-25 -> 2025-26) splits using
in-memory cached raw SQL player histories for reproducible apples-to-apples comparison.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import scipy.stats as stats
import sqlalchemy
import xgboost as xgb
from dotenv import load_dotenv
from joblib import Parallel, delayed
from sklearn.linear_model import Ridge, RidgeCV

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluate_fast import _compute_crps_batch
from src.feature_contract import (
    CATEGORICAL_ID_COLUMNS,
    FEATURE_SCHEMA_VERSION,
    TARGET_METRICS,
    compute_causal_player_features,
    compute_target_rolling_features,
    extract_xgb_categories_from_json,
    get_active_feature_columns,
    predict_meta_ensemble,
)
from src.features import ContextualFeatureEngine
from src.model_registry import ModelRegistry, PromotionRejectedError
from src.models import get_default_priors
from src.train_global_model import _iter_grouped_time_splits

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("AblationExp14")

ABLATION_TMP_DIR = "models/ablation_tmp"
REPORT_PATH = "benchmarks/ablation_exp14_report.json"
EVAL_STATS = ("pts", "reb", "ast")


def _db_uri() -> str:
    uri = os.getenv("POSTGRES_URI", "postgresql+psycopg2://postgres:password@localhost:5432/sports_db")
    if uri.startswith("postgres://"):
        uri = uri.replace("postgres://", "postgresql+psycopg2://", 1)
    return uri


def _compile_fold_datasets(
    db_uri: str,
    seasons: List[str],
    league: str = "NBA",
) -> Tuple[List[pd.DataFrame], Dict[str, str], Optional[str]]:
    """Queries DB once for `seasons` up to the seasonal end cutoff."""
    engine = sqlalchemy.create_engine(db_uri)
    with engine.connect() as conn:
        stmt = sqlalchemy.text(
            "SELECT DISTINCT g.player_id FROM player_game_logs g "
            "JOIN games gm ON g.game_id = gm.game_id "
            "WHERE gm.sport = :league AND gm.season IN :seasons"
        ).bindparams(sqlalchemy.bindparam("seasons", expanding=True))
        player_ids = [r[0] for r in conn.execute(stmt, {"league": league, "seasons": list(seasons)}).fetchall()]

        cutoff_stmt = sqlalchemy.text(
            "SELECT MAX(start_time) FROM games WHERE sport = :league AND season IN :seasons"
        ).bindparams(sqlalchemy.bindparam("seasons", expanding=True))
        max_dt = conn.execute(cutoff_stmt, {"league": league, "seasons": list(seasons)}).scalar()
        cutoff = (pd.to_datetime(max_dt, utc=True) + pd.Timedelta(seconds=1)).isoformat() if max_dt is not None else None

        pos_res = conn.execute(sqlalchemy.text("SELECT master_player_id, position FROM players")).fetchall()
        player_positions = {str(r[0]): r[1] for r in pos_res}

    def _fetch(pid: str) -> Optional[pd.DataFrame]:
        global _ablation_fe
        if "_ablation_fe" not in globals():
            _ablation_fe = ContextualFeatureEngine(db_uri=db_uri, league=league)
        try:
            df_p = _ablation_fe.build_player_feature_matrix(pid, pd.DataFrame(), target_date=cutoff)
            if df_p is None or df_p.empty or len(df_p) < 5:
                return None
            return df_p
        except Exception:
            return None

    logger.info(f"Compiling {len(player_ids)} raw player matrices (seasons={seasons}, cutoff={cutoff})...")
    compiled = Parallel(n_jobs=8)(delayed(_fetch)(pid) for pid in player_ids)
    valid = [df for df in compiled if df is not None]
    logger.info(f"Compiled {len(valid)} non-empty player matrices for seasons={seasons}.")
    return valid, player_positions, cutoff


def _prepare_player_features(
    df_p: pd.DataFrame,
    *,
    flat_ewma_alpha: Optional[float] = None,
    league: str = "NBA",
) -> pd.DataFrame:
    out = compute_causal_player_features(df_p, league=league)
    if flat_ewma_alpha is not None and not out.empty:
        # Override causal cohort alpha with a uniform alpha while keeping causal tier label intact
        out = out.copy()
        out["tier_alpha"] = float(flat_ewma_alpha)
        from src.feature_contract import _causal_tier_ewma

        out["roll_min"] = _causal_tier_ewma(out, "minutes_played", default=None)
        out["roll_usg"] = _causal_tier_ewma(out, "usage_rate", default=None)
        out["roll_potential_assists"] = _causal_tier_ewma(out, "potential_assists", default=0.0)
        out["roll_touches"] = _causal_tier_ewma(out, "touches", default=0.0)
        out["roll_passes_made"] = _causal_tier_ewma(out, "passes_made", default=0.0)
        out["roll_time_of_possession_sec"] = _causal_tier_ewma(out, "time_of_possession_sec", default=0.0)
        out["roll_defender_distance"] = _causal_tier_ewma(out, "avg_defender_distance_ft", default=4.5)
        out["roll_shorthanded_points"] = _causal_tier_ewma(out, "shorthanded_points", default=0.0)
        out["roll_plus_minus"] = _causal_tier_ewma(out, "plus_minus", default=0.0)
        out["roll_ts_pct"] = _causal_tier_ewma(out, "ts_pct", default=0.0)
        out["roll_reb_rate"] = _causal_tier_ewma(out, "reb_rate", default=0.0)
        out["roll_ast_rate"] = _causal_tier_ewma(out, "ast_rate", default=0.0)
        out["roll_usg_efficiency"] = _causal_tier_ewma(out, "usg_efficiency", default=0.0)
        out["usage_x_minutes_interaction"] = out["roll_usg"].fillna(0.0) * out["roll_min"].fillna(0.0)
    return out


def _train_variant_fold(
    raw_train_dfs: List[pd.DataFrame],
    train_seasons: List[str],
    player_positions: Dict[str, str],
    out_dir: str,
    variant_cfg: Dict[str, Any],
    target_keys: Tuple[str, ...] = EVAL_STATS,
    config_path: str = "config/stat_hyperparameters.json",
    league: str = "NBA",
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    cached_priors = get_default_priors(league=league)

    with open(config_path, "r") as f:
        custom_config = json.load(f)

    drop_cat_ids = bool(variant_cfg.get("drop_cat_ids", False))
    use_early_stopping = bool(variant_cfg.get("use_early_stopping", False))
    use_ridge_cv = bool(variant_cfg.get("use_ridge_cv", False))
    use_l2_objective = bool(variant_cfg.get("use_l2_objective", False))
    flat_ewma_alpha = variant_cfg.get("flat_ewma_alpha")

    # Prepare per-player season-isolated base features
    prepared_players: List[pd.DataFrame] = []
    for df_p in raw_train_dfs:
        df_sub = df_p[df_p["season"].isin(train_seasons)].copy() if "season" in df_p.columns else df_p.copy()
        if df_sub.empty or len(df_sub) < 5:
            continue
        prepared_players.append(_prepare_player_features(df_sub, flat_ewma_alpha=flat_ewma_alpha, league=league))

    base_xgb = {
        "objective": "reg:squarederror" if use_l2_objective else "reg:absoluteerror",
        "eval_metric": "rmse" if use_l2_objective else "mae",
        "tree_method": "hist",
        "enable_categorical": True,
        "n_estimators": 150,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
    }
    base_lgb = {
        "objective": "regression" if use_l2_objective else "regression_l1",
        "metric": "rmse" if use_l2_objective else "mae",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "random_state": 42,
        "verbose": -1,
    }

    for target_key in target_keys:
        stat_col = TARGET_METRICS[target_key]
        active_features = get_active_feature_columns(league=league, stat_col=stat_col)
        if drop_cat_ids:
            active_features = [c for c in active_features if c not in CATEGORICAL_ID_COLUMNS]

        target_xgb_params = base_xgb.copy()
        target_lgb_params = base_lgb.copy()
        if target_key in custom_config:
            if "xgb_params" in custom_config[target_key]:
                target_xgb_params.update(custom_config[target_key]["xgb_params"])
            if "lgb_params" in custom_config[target_key]:
                target_lgb_params.update(custom_config[target_key]["lgb_params"])

        # Re-enforce objective choice over stat_hyperparameters.json when L2 ablation is tested
        if use_l2_objective:
            target_xgb_params["objective"] = "reg:squarederror"
            target_xgb_params["eval_metric"] = "rmse"
            target_lgb_params["objective"] = "regression"
            target_lgb_params["metric"] = "rmse"
        else:
            target_xgb_params["objective"] = "reg:absoluteerror"
            target_xgb_params["eval_metric"] = "mae"
            target_lgb_params["objective"] = "regression_l1"
            target_lgb_params["metric"] = "mae"

        if drop_cat_ids:
            target_xgb_params.pop("enable_categorical", None)

        all_dfs = []
        for df_p in prepared_players:
            if stat_col not in df_p.columns:
                continue
            pid = str(df_p["player_id"].iloc[0])
            pos = player_positions.get(pid, "SG")
            df_stat = compute_target_rolling_features(
                df_p,
                stat_col=stat_col,
                position=pos,
                cached_priors=cached_priors,
                league=league,
            )
            all_dfs.append(df_stat)

        df = pd.concat(all_dfs, ignore_index=True)
        if "start_time" in df.columns:
            df["start_time"] = pd.to_datetime(df["start_time"], utc=True)
            df = df.sort_values(["start_time", "player_id"]).reset_index(drop=True)

        full_contract_cols = get_active_feature_columns(league=league, stat_col=stat_col)
        for col in full_contract_cols + ["target_residual", stat_col, "bayesian_roll_stat"]:
            if col in CATEGORICAL_ID_COLUMNS:
                if col == "player_id":
                    df[col] = df[col].astype(str)
                else:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

        # Identical NA-drop set across all ablations so every variant uses the exact same samples
        train_df = (
            df.dropna(subset=full_contract_cols + ["target_residual", stat_col, "bayesian_roll_stat"])
            .copy()
            .reset_index(drop=True)
        )
        for cat_col in CATEGORICAL_ID_COLUMNS:
            if cat_col in train_df.columns and cat_col in active_features:
                if cat_col == "player_id":
                    train_df[cat_col] = train_df[cat_col].astype(str).astype("category")
                else:
                    train_df[cat_col] = train_df[cat_col].astype(int).astype("category")

        X_train = train_df[active_features]
        y_train = train_df["target_residual"].values

        n_splits = 5
        oof_xgb, oof_lgb, oof_y, oof_indices = [], [], [], []
        xgb_best_rounds, lgb_best_rounds = [], []

        for train_idx, val_idx in _iter_grouped_time_splits(train_df, n_splits=n_splits):
            X_f_train, X_f_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_f_train, y_f_val = y_train[train_idx], y_train[val_idx]

            es_split = max(1, int(len(X_f_train) * 0.85))
            X_es_train, X_es_val = X_f_train.iloc[:es_split], X_f_train.iloc[es_split:]
            y_es_train, y_es_val = y_f_train[:es_split], y_f_train[es_split:]

            fold_xgb = xgb.XGBRegressor(**{**target_xgb_params, "n_estimators": 500, "early_stopping_rounds": 20})
            fold_xgb.fit(X_es_train, y_es_train, eval_set=[(X_es_val, y_es_val)], verbose=False)
            pred_xgb_val = fold_xgb.predict(X_f_val)
            best_xgb_it = getattr(fold_xgb, "best_iteration", None)
            if best_xgb_it is not None:
                xgb_best_rounds.append(max(10, int(best_xgb_it) + 1))

            fold_lgb_params = {k: v for k, v in target_lgb_params.items() if k != "n_estimators"}
            dtrain_f = lgb.Dataset(X_es_train, label=y_es_train)
            dval_f = lgb.Dataset(X_es_val, label=y_es_val, reference=dtrain_f)
            fold_lgb = lgb.train(
                fold_lgb_params,
                dtrain_f,
                num_boost_round=500,
                valid_sets=[dval_f],
                callbacks=[lgb.early_stopping(20, verbose=False)],
            )
            pred_lgb_val = fold_lgb.predict(X_f_val)
            best_lgb_it = getattr(fold_lgb, "best_iteration", None)
            if best_lgb_it is not None and int(best_lgb_it) > 0:
                lgb_best_rounds.append(max(10, int(best_lgb_it)))

            oof_xgb.extend(pred_xgb_val)
            oof_lgb.extend(pred_lgb_val)
            oof_y.extend(y_f_val)
            oof_indices.extend(val_idx.tolist())

        stacked_oof = np.column_stack([oof_xgb, oof_lgb])
        oof_y_arr = np.array(oof_y)

        if use_ridge_cv:
            meta_learner = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0, 500.0, 1000.0])
            meta_learner.fit(stacked_oof, oof_y_arr)
            ridge_alpha_used = float(meta_learner.alpha_)
        else:
            meta_learner = Ridge(alpha=1.0)
            meta_learner.fit(stacked_oof, oof_y_arr)
            ridge_alpha_used = 1.0

        if use_early_stopping:
            final_xgb_rounds = max(10, int(round(np.mean(xgb_best_rounds)))) if xgb_best_rounds else 150
            final_lgb_rounds = max(10, int(round(np.mean(lgb_best_rounds)))) if lgb_best_rounds else 150
            final_lgb_params = {k: v for k, v in target_lgb_params.items() if k != "n_estimators"}
        else:
            final_xgb_rounds = 150
            final_lgb_rounds = 500
            final_lgb_params = target_lgb_params

        final_xgb = xgb.XGBRegressor(**{**target_xgb_params, "n_estimators": final_xgb_rounds})
        final_xgb.fit(X_train, y_train, verbose=False)

        dtrain_full = lgb.Dataset(X_train, label=y_train)
        final_lgb = lgb.train(final_lgb_params, dtrain_full, num_boost_round=final_lgb_rounds)

        oof_meta_residuals = meta_learner.predict(stacked_oof)
        oof_df = train_df.iloc[oof_indices]
        mu_vals = np.clip(oof_df["bayesian_roll_stat"].values + oof_meta_residuals, 0.001, None)
        true_vals = oof_df[stat_col].values

        def crps_objective(trial):
            alpha = trial.suggest_float("alpha", 0.01, 3.0)
            n_nb = 1.0 / alpha
            p_nb = 1.0 / (1.0 + alpha * mu_vals)
            if np.any(p_nb <= 0.0) or np.any(p_nb >= 1.0):
                return 1e9
            return float(-np.mean(stats.nbinom.logpmf(true_vals, n_nb, p_nb)))

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize")
        study.optimize(crps_objective, n_trials=20)
        best_alpha = float(study.best_params["alpha"])

        xgb_path = os.path.join(out_dir, f"global_x{target_key}_xgb.json")
        lgb_path = os.path.join(out_dir, f"global_x{target_key}_lgb.txt")
        meta_path = os.path.join(out_dir, f"global_x{target_key}_meta.json")
        compat_path = os.path.join(out_dir, f"global_x{target_key}.json")

        final_xgb.save_model(xgb_path)
        final_xgb.save_model(compat_path)
        final_lgb.save_model(lgb_path)

        meta_data = {
            "target": target_key,
            "stat_column": stat_col,
            "target_type": "residual",
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "feature_columns": active_features,
            "train_seasons": list(train_seasons),
            "intercept": float(meta_learner.intercept_),
            "weights": [float(w) for w in meta_learner.coef_],
            "ridge_alpha": ridge_alpha_used,
            "xgb_rounds": int(final_xgb_rounds),
            "lgb_rounds": int(final_lgb.current_iteration()),
            "features": ["xgb_residual", "lgb_residual"],
            "alpha": best_alpha,
            "n_splits": n_splits,
            "training_samples": int(len(train_df)),
        }
        with open(meta_path, "w") as f:
            json.dump(meta_data, f, indent=2)


def _evaluate_variant_fold(
    raw_eval_dfs: List[pd.DataFrame],
    test_seasons: List[str],
    player_positions: Dict[str, str],
    model_dir: str,
    variant_cfg: Dict[str, Any],
    league: str = "NBA",
) -> Dict[str, Any]:
    cached_priors = get_default_priors(league=league)
    flat_ewma_alpha = variant_cfg.get("flat_ewma_alpha")

    models: Dict[str, Dict[str, Any]] = {}
    for target_key in EVAL_STATS:
        xgb_path = os.path.join(model_dir, f"global_x{target_key}_xgb.json")
        lgb_path = os.path.join(model_dir, f"global_x{target_key}_lgb.txt")
        meta_path = os.path.join(model_dir, f"global_x{target_key}_meta.json")
        bst = xgb.XGBRegressor()
        bst.load_model(xgb_path)
        with open(meta_path, "r") as f:
            meta = json.load(f)
        models[target_key] = {
            "xgb": bst,
            "xgb_categories": extract_xgb_categories_from_json(xgb_path),
            "lgb": lgb.Booster(model_file=lgb_path),
            "meta": meta,
        }

    causal_dfs = []
    for df_p in raw_eval_dfs:
        featured = _prepare_player_features(df_p, flat_ewma_alpha=flat_ewma_alpha, league=league)
        if "season" in featured.columns and test_seasons:
            if featured[featured["season"].isin(test_seasons)].empty:
                continue
        causal_dfs.append(featured)

    by_stat: Dict[str, Dict[str, Any]] = {}
    all_actual, all_meta, all_ewma, all_crps = [], [], [], []
    stat_meta_info: Dict[str, Any] = {}

    for target_key in EVAL_STATS:
        stat_col = TARGET_METRICS[target_key]
        bundle = models[target_key]
        meta_data = bundle["meta"]
        model_features = meta_data["feature_columns"]
        full_contract_cols = get_active_feature_columns(league=league, stat_col=stat_col)

        stat_dfs = []
        for df_p in causal_dfs:
            if stat_col not in df_p.columns:
                continue
            pid = str(df_p["player_id"].iloc[0])
            pos = player_positions.get(pid, "SG")
            stat_dfs.append(
                compute_target_rolling_features(
                    df_p,
                    stat_col=stat_col,
                    position=pos,
                    cached_priors=cached_priors,
                    league=league,
                )
            )

        full_df = pd.concat(stat_dfs, ignore_index=True)
        test_df = full_df[full_df["season"].isin(test_seasons)].copy()
        test_df = test_df.dropna(subset=full_contract_cols + ["target_residual", stat_col, "bayesian_roll_stat"]).copy()
        test_df = test_df.sort_values(["player_id", "start_time", "game_id"], kind="mergesort").reset_index(drop=True)

        ewma_baseline = test_df["bayesian_roll_stat"].to_numpy(dtype=float)
        preds = predict_meta_ensemble(
            test_df,
            model_features,
            xgb_model=bundle["xgb"],
            lgb_model=bundle["lgb"],
            meta_data=meta_data,
            ewma_baseline=ewma_baseline,
            league=league,
            stat_col=stat_col,
            roll_min=test_df["roll_min"].to_numpy(dtype=float),
            xgb_categories=bundle["xgb_categories"],
        )
        meta_pred = preds["meta_prediction"]
        actuals = test_df[stat_col].to_numpy(dtype=float)
        alpha_nb = float(meta_data.get("alpha", 0.1))
        crps = _compute_crps_batch(meta_pred, actuals, alpha_nb, n_iters=10000, rng_seed=42)

        meta_mae = float(np.mean(np.abs(actuals - meta_pred)))
        ewma_mae = float(np.mean(np.abs(actuals - ewma_baseline)))
        sim_crps = float(np.mean(crps))
        mean_bias = float(np.mean(actuals - meta_pred))
        imp_pct = ((ewma_mae - meta_mae) / ewma_mae) * 100.0 if ewma_mae > 0 else 0.0

        by_stat[target_key] = {
            "count": int(len(actuals)),
            "meta_mae": meta_mae,
            "ewma_mae": ewma_mae,
            "mae_imp_pct": imp_pct,
            "sim_crps": sim_crps,
            "mean_residual_bias": mean_bias,
            "nb_alpha": alpha_nb,
        }
        stat_meta_info[target_key] = {
            "xgb_rounds": meta_data.get("xgb_rounds"),
            "lgb_rounds": meta_data.get("lgb_rounds"),
            "ridge_alpha": meta_data.get("ridge_alpha"),
            "weights": meta_data.get("weights"),
            "intercept": meta_data.get("intercept"),
            "alpha": alpha_nb,
        }

        all_actual.append(actuals)
        all_meta.append(meta_pred)
        all_ewma.append(ewma_baseline)
        all_crps.append(crps)

    act_cat = np.concatenate(all_actual)
    meta_cat = np.concatenate(all_meta)
    ewma_cat = np.concatenate(all_ewma)
    crps_cat = np.concatenate(all_crps)

    ov_mae = float(np.mean(np.abs(act_cat - meta_cat)))
    ov_ewma = float(np.mean(np.abs(act_cat - ewma_cat)))
    ov_crps = float(np.mean(crps_cat))
    ov_bias = float(np.mean(act_cat - meta_cat))
    ov_imp = ((ov_ewma - ov_mae) / ov_ewma) * 100.0 if ov_ewma > 0 else 0.0

    return {
        "overall": {
            "count": int(len(act_cat)),
            "meta_mae": ov_mae,
            "ewma_mae": ov_ewma,
            "mae_imp_pct": ov_imp,
            "sim_crps": ov_crps,
            "mean_residual_bias": ov_bias,
        },
        "by_stat": by_stat,
        "model_meta": stat_meta_info,
    }


SINGLE_FACTOR_VARIANTS: List[Tuple[str, Dict[str, Any], str]] = [
    ("control_exp13", {}, "Experiment 13 leak-free baseline (deep trees, native IDs, Ridge(1.0), L1, cohort alphas)"),
    ("ablation_a_early_stopping", {"use_early_stopping": True}, "Date-grouped CV best_iteration early stopping"),
    ("ablation_b_no_cat_ids", {"drop_cat_ids": True}, "Drop player_id, team_id, opponent_team_id categoricals"),
    ("ablation_c_ridge_cv", {"use_ridge_cv": True}, "Adaptive RidgeCV(alphas=[0.1..1000]) on OOF residual stack"),
    ("ablation_d_l2_objective", {"use_l2_objective": True}, "L2 squared-error base learners (reg:squarederror / regression)"),
    ("ablation_e_flat_ewma", {"flat_ewma_alpha": 0.30}, "Uniform EWMA alpha=0.30 baseline instead of 4-tier cohort alphas"),
]


def run_ablation() -> Dict[str, Any]:
    t_start = time.time()
    db_uri = _db_uri()

    if os.path.exists(ABLATION_TMP_DIR):
        shutil.rmtree(ABLATION_TMP_DIR)
    os.makedirs(ABLATION_TMP_DIR, exist_ok=True)

    # Compile strict point-in-time raw DB snapshots ONCE per boundary
    raw_f1_train, pos_map, cutoff_f1_train = _compile_fold_datasets(db_uri, ["2023-24"])
    raw_f1_eval, _, cutoff_f1_eval = _compile_fold_datasets(db_uri, ["2023-24", "2024-25"])
    raw_f2_eval, _, cutoff_f2_eval = _compile_fold_datasets(db_uri, ["2023-24", "2024-25", "2025-26"])

    results: Dict[str, Any] = {}

    def _run_one(name: str, cfg: Dict[str, Any], desc: str) -> Dict[str, Any]:
        logger.info(f"\n==================== RUNNING VARIANT: {name} ====================")
        f1_dir = os.path.join(ABLATION_TMP_DIR, name, "fold1")
        f2_dir = os.path.join(ABLATION_TMP_DIR, name, "fold2")
        _train_variant_fold(raw_f1_train, ["2023-24"], pos_map, f1_dir, cfg)
        fold1 = _evaluate_variant_fold(raw_f1_eval, ["2024-25"], pos_map, f1_dir, cfg)
        _train_variant_fold(raw_f1_eval, ["2023-24", "2024-25"], pos_map, f2_dir, cfg)
        fold2 = _evaluate_variant_fold(raw_f2_eval, ["2025-26"], pos_map, f2_dir, cfg)
        logger.info(
            f"[{name}] Fold1 MAE={fold1['overall']['meta_mae']:.4f} CRPS={fold1['overall']['sim_crps']:.4f} Bias={fold1['overall']['mean_residual_bias']:+.4f} | "
            f"Fold2 MAE={fold2['overall']['meta_mae']:.4f} CRPS={fold2['overall']['sim_crps']:.4f} Bias={fold2['overall']['mean_residual_bias']:+.4f}"
        )
        return {"description": desc, "config": cfg, "fold1": fold1, "fold2": fold2}

    for name, cfg, desc in SINGLE_FACTOR_VARIANTS:
        results[name] = _run_one(name, cfg, desc)

    # Identify single-factor variants that strictly improve BOTH Fold 1 and Fold 2
    ctrl = results["control_exp13"]
    c_f1_mae = ctrl["fold1"]["overall"]["meta_mae"]
    c_f1_crps = ctrl["fold1"]["overall"]["sim_crps"]
    c_f2_mae = ctrl["fold2"]["overall"]["meta_mae"]
    c_f2_crps = ctrl["fold2"]["overall"]["sim_crps"]

    strict_winners: List[str] = []
    mae_or_composite_winners: List[str] = []
    for name, cfg, _ in SINGLE_FACTOR_VARIANTS[1:]:
        r = results[name]
        f1_mae, f1_crps = r["fold1"]["overall"]["meta_mae"], r["fold1"]["overall"]["sim_crps"]
        f2_mae, f2_crps = r["fold2"]["overall"]["meta_mae"], r["fold2"]["overall"]["sim_crps"]

        # Composite score: mean normalized change across MAE & CRPS on both folds (< 0 is better)
        score = 0.25 * (
            (f1_mae - c_f1_mae) / c_f1_mae
            + (f1_crps - c_f1_crps) / c_f1_crps
            + (f2_mae - c_f2_mae) / c_f2_mae
            + (f2_crps - c_f2_crps) / c_f2_crps
        )
        strict_both = (
            f1_mae < c_f1_mae and f1_crps < c_f1_crps and f2_mae < c_f2_mae and f2_crps < c_f2_crps
        )
        composite_both = (
            0.5 * ((f1_mae - c_f1_mae) / c_f1_mae + (f1_crps - c_f1_crps) / c_f1_crps) < 0.0
            and 0.5 * ((f2_mae - c_f2_mae) / c_f2_mae + (f2_crps - c_f2_crps) / c_f2_crps) < 0.0
        )
        r["delta_vs_control"] = {
            "fold1_mae_delta": f1_mae - c_f1_mae,
            "fold1_crps_delta": f1_crps - c_f1_crps,
            "fold2_mae_delta": f2_mae - c_f2_mae,
            "fold2_crps_delta": f2_crps - c_f2_crps,
            "weighted_composite_delta": score,
            "strict_both_folds_mae_and_crps": strict_both,
            "composite_both_folds_improved": composite_both,
        }
        if strict_both:
            strict_winners.append(name)
        if composite_both:
            mae_or_composite_winners.append((score, name))

    ctrl["delta_vs_control"] = {
        "fold1_mae_delta": 0.0,
        "fold1_crps_delta": 0.0,
        "fold2_mae_delta": 0.0,
        "fold2_crps_delta": 0.0,
        "weighted_composite_delta": 0.0,
        "strict_both_folds_mae_and_crps": False,
        "composite_both_folds_improved": False,
    }

    selected_names = strict_winners if strict_winners else [n for _, n in sorted(mae_or_composite_winners)]
    combo_cfg: Dict[str, Any] = {}
    for wname in selected_names:
        combo_cfg.update(results[wname]["config"])

    if not combo_cfg:
        # Select single best composite candidate as challenger fallback test to exercise registry gate
        best_single = min(
            SINGLE_FACTOR_VARIANTS[1:],
            key=lambda item: results[item[0]]["delta_vs_control"]["weighted_composite_delta"],
        )[0]
        combo_cfg = dict(results[best_single]["config"])
        combo_desc = f"Fallback best single candidate ({best_single}) tested because zero variants improved both folds on MAE+CRPS"
        selected_names = [best_single]
    else:
        combo_desc = f"Combined strict/composite winners on Fold 1 & Fold 2: {selected_names}"

    combo_res = _run_one("challenger_best_combo", combo_cfg, combo_desc)
    cm_f1, cc_f1 = combo_res["fold1"]["overall"]["meta_mae"], combo_res["fold1"]["overall"]["sim_crps"]
    cm_f2, cc_f2 = combo_res["fold2"]["overall"]["meta_mae"], combo_res["fold2"]["overall"]["sim_crps"]
    combo_res["delta_vs_control"] = {
        "fold1_mae_delta": cm_f1 - c_f1_mae,
        "fold1_crps_delta": cc_f1 - c_f1_crps,
        "fold2_mae_delta": cm_f2 - c_f2_mae,
        "fold2_crps_delta": cc_f2 - c_f2_crps,
        "weighted_composite_delta": 0.25
        * (
            (cm_f1 - c_f1_mae) / c_f1_mae
            + (cc_f1 - c_f1_crps) / c_f1_crps
            + (cm_f2 - c_f2_mae) / c_f2_mae
            + (cc_f2 - c_f2_crps) / c_f2_crps
        ),
        "selected_factors": selected_names,
    }
    results["challenger_best_combo"] = combo_res

    # Strict promotion criteria from task spec:
    # challenger_best_combo strictly beats control_exp13 on Fold 2 MAE AND CRPS AND does not regress Fold 1
    beats_f2 = (cm_f2 < c_f2_mae) and (cc_f2 < c_f2_crps)
    no_f1_regress = (cm_f1 <= c_f1_mae) and (cc_f1 <= c_f1_crps)
    should_promote = beats_f2 and no_f1_regress

    registry_decision: Dict[str, Any] = {
        "strict_winners": strict_winners,
        "selected_for_combo": selected_names,
        "beats_fold2_mae_and_crps": beats_f2,
        "no_fold1_regression": no_f1_regress,
        "eligible_for_promotion": should_promote,
    }

    # Test against ModelRegistry promotion gate
    registry = ModelRegistry()
    incumbent = registry.get_champion()
    registry_decision["incumbent_champion"] = incumbent["version_id"] if incumbent else None

    if should_promote:
        # Train all 7 targets into a persistent versioned bundle under models/wf_fold2_exp14
        champ_dir = "models/wf_fold2_exp14"
        logger.info(f"Challenger cleared Exp 14 gates! Retraining all 7 targets into {champ_dir}...")
        _train_variant_fold(
            raw_f1_eval,
            ["2023-24", "2024-25"],
            pos_map,
            champ_dir,
            combo_cfg,
            target_keys=tuple(TARGET_METRICS.keys()),
        )
        challenger_manifest = registry.register_version(
            model_dir=champ_dir,
            version_id=f"v2026.11.25-exp14",
            metrics={
                "overall_mae": cm_f2,
                "overall_crps": cc_f2,
                "ewma_mae": combo_res["fold2"]["overall"]["ewma_mae"],
                "by_stat": {
                    k: {
                        "meta_mae": v["meta_mae"],
                        "ewma_mae": v["ewma_mae"],
                        "sim_crps": v["sim_crps"],
                    }
                    for k, v in combo_res["fold2"]["by_stat"].items()
                },
            },
            notes=f"Exp 14 winning combo ({selected_names})",
        )
        promoted = registry.promote_challenger(challenger_manifest["version_id"])
        registry_decision["gate_outcome"] = "PROMOTED"
        registry_decision["challenger_version_id"] = promoted["version_id"]
        registry_decision["gate_message"] = f"Promoted {promoted['version_id']} to champion."
    else:
        # Evaluate the challenger against the real ModelRegistry gate in an isolated temp registry or dry-check so champion.json stays clean unless registered
        temp_reg_dir = os.path.join(ABLATION_TMP_DIR, "registry_gate_check")
        shutil.copytree("models/registry", temp_reg_dir, dirs_exist_ok=True)
        temp_reg = ModelRegistry(registry_dir=temp_reg_dir)
        f2_tmp_dir = os.path.join(ABLATION_TMP_DIR, "challenger_best_combo", "fold2")
        temp_manifest = temp_reg.register_version(
            model_dir=f2_tmp_dir,
            version_id="v2026.11.25-exp14-rejected",
            metrics={
                "overall_mae": cm_f2,
                "overall_crps": cc_f2,
                "ewma_mae": combo_res["fold2"]["overall"]["ewma_mae"],
            },
            notes="Exp 14 challenger evaluated against gate",
        )
        try:
            temp_reg.promote_challenger(temp_manifest["version_id"])
            # Fold 2 beat champion on MAE+CRPS, yet Fold 1 regressed -> explicitly reject by multi-fold rule
            registry_decision["gate_outcome"] = "REJECTED"
            registry_decision["gate_message"] = (
                f"Rejected by Exp 14 multi-fold non-regression rule (beats_f2={beats_f2}, no_f1_regress={no_f1_regress}); incumbent champion '{registry_decision['incumbent_champion']}' preserved."
            )
        except PromotionRejectedError as exc:
            registry_decision["gate_outcome"] = "REJECTED"
            registry_decision["gate_message"] = str(exc)

    report = {
        "experiment": "Experiment 14: Clean Leak-Free Ablation Study",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "cutoffs": {
            "fold1_train": cutoff_f1_train,
            "fold1_eval": cutoff_f1_eval,
            "fold2_eval": cutoff_f2_eval,
        },
        "elapsed_seconds": round(time.time() - t_start, 2),
        "variants": results,
        "registry_decision": registry_decision,
    }

    os.makedirs(os.path.dirname(os.path.abspath(REPORT_PATH)), exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved Experiment 14 ablation report to {REPORT_PATH} in {report['elapsed_seconds']}s.")

    # Clean up temporary ablation model weights after archiving metrics
    if os.path.exists(ABLATION_TMP_DIR):
        shutil.rmtree(ABLATION_TMP_DIR)
        logger.info(f"Cleaned temporary ablation directory {ABLATION_TMP_DIR}.")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Experiment 14 clean ablation matrix")
    parser.parse_args()
    run_ablation()
