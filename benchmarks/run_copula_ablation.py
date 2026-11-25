"""Experiment 15: Walk-Forward Joint Copula Calibration Ablation (Conditional theta(x) vs Static & Independent).

Trains and evaluates pairwise dependence models strictly chronologically:
  - Fold 1: Train on 2023-24 -> Evaluate on unseen 2024-25 games
  - Fold 2: Train on 2023-24, 2024-25 -> Evaluate on unseen 2025-26 games
Includes game_id cluster-bootstrap significance testing (1,000 resamples) and sub-regime
diagnostics (same-team, cross-team, same-position substitute, high-spread blowout risk).
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import sqlalchemy
import xgboost as xgb
from dotenv import load_dotenv
from joblib import Parallel, delayed
from scipy import stats
from sklearn.linear_model import Ridge

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.conditional_copula import (
    TAU_MAX,
    ConditionalCopulaModel,
    build_pair_feature_matrix,
    canonical_direction_pair,
    canonical_stat_pair,
    clean_stat_key,
    vectorized_copula_joint_prob,
)
from src.drift_monitor import compute_brier_score, compute_ece, compute_log_loss
from src.evaluate_fast import load_models
from src.execution import DEFAULT_CORRELATIONS
from src.feature_contract import (
    CATEGORICAL_ID_COLUMNS,
    FEATURE_SCHEMA_VERSION,
    TARGET_METRICS,
    compute_causal_player_features,
    compute_target_rolling_features,
    get_active_feature_columns,
    predict_meta_ensemble,
)
from src.features import ContextualFeatureEngine
from src.models import get_default_priors
from src.serving.predictor import InferencePredictor
from src.serving.schemas import CopulaTypeEnum
from src.train_global_model import _iter_grouped_time_splits

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("CopulaAblationExp15")

REPORT_PATH = "benchmarks/copula_ablation_exp15_report.json"
COPULA_MODEL_OUT_PATH = "models/registry/conditional_copula_fold2.json"
EVAL_STATS = ("pts", "reb", "ast")
SEED = 42


def _db_uri() -> str:
    uri = os.getenv("POSTGRES_URI", "postgresql+psycopg2://postgres:password@localhost:5432/sports_db")
    if uri.startswith("postgres://"):
        uri = uri.replace("postgres://", "postgresql+psycopg2://", 1)
    return uri


def _compile_fold_datasets(
    db_uri: str, seasons: List[str], league: str = "NBA"
) -> Tuple[List[pd.DataFrame], Dict[str, str], Optional[str]]:
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
        player_positions = {str(r[0]): (r[1] or "SG") for r in pos_res}

    def _fetch(pid: str) -> Optional[pd.DataFrame]:
        global _cop_fe
        if "_cop_fe" not in globals():
            _cop_fe = ContextualFeatureEngine(db_uri=db_uri, league=league)
        try:
            df_p = _cop_fe.build_player_feature_matrix(pid, pd.DataFrame(), target_date=cutoff)
            if df_p is None or df_p.empty or len(df_p) < 5:
                return None
            return df_p
        except Exception:
            return None

    logger.info(f"Compiling {len(player_ids)} player matrices (seasons={seasons}, cutoff={cutoff})...")
    compiled = Parallel(n_jobs=8)(delayed(_fetch)(pid) for pid in player_ids)
    valid = [df for df in compiled if df is not None]
    logger.info(f"Compiled {len(valid)} player matrices.")
    return valid, player_positions, cutoff


def _compute_causal_team_strength_map(player_dfs: List[pd.DataFrame]) -> Dict[Tuple[int, int], float]:
    """Computes strictly causal pre-game team net point differential EWMA for every (game_id, team_id)."""
    rows = []
    for df_p in player_dfs:
        cols = [c for c in ("game_id", "start_time", "season", "home_team_id", "away_team_id", "team_id", "opponent_team_id", "points") if c in df_p.columns]
        if len(cols) == 8:
            rows.append(df_p[cols].copy())
    if not rows:
        return {}

    all_logs = pd.concat(rows, ignore_index=True)
    all_logs["start_time"] = pd.to_datetime(all_logs["start_time"], utc=True)
    all_logs["team_id"] = pd.to_numeric(all_logs["team_id"], errors="coerce")
    all_logs["opponent_team_id"] = pd.to_numeric(all_logs["opponent_team_id"], errors="coerce")
    all_logs = all_logs.dropna(subset=["game_id", "team_id", "opponent_team_id"]).copy()
    all_logs["team_id"] = all_logs["team_id"].astype(int)
    all_logs["opponent_team_id"] = all_logs["opponent_team_id"].astype(int)

    team_game_pts = (
        all_logs.groupby(["game_id", "start_time", "season", "team_id", "opponent_team_id"], as_index=False)["points"]
        .sum()
        .rename(columns={"points": "team_pts"})
    )
    opp_pts = team_game_pts[["game_id", "team_id", "team_pts"]].rename(
        columns={"team_id": "opponent_team_id", "team_pts": "opp_pts"}
    )
    merged = team_game_pts.merge(opp_pts, on=["game_id", "opponent_team_id"], how="inner")
    merged["margin"] = merged["team_pts"] - merged["opp_pts"]
    merged = merged.sort_values(["team_id", "start_time", "game_id"], kind="mergesort").reset_index(drop=True)

    # Strictly shifted EWMA net point margin within each season (0..T-1 ONLY)
    merged["pre_team_strength"] = (
        merged.groupby(["team_id", "season"], sort=False)["margin"]
        .transform(lambda s: s.shift(1).ewm(alpha=0.15, adjust=False).mean())
        .fillna(0.0)
    )
    return {
        (int(r.game_id), int(r.team_id)): float(r.pre_team_strength)
        for r in merged.itertuples(index=False)
    }


def _prepare_zinb_legs_from_model(
    player_dfs: List[pd.DataFrame],
    models: Optional[dict],
    target_seasons: List[str],
    player_positions: Dict[str, str],
    team_strength_map: Dict[Tuple[int, int], float],
    use_oof_internal: bool = False,
) -> pd.DataFrame:
    """Computes strictly causal/OOS ZINB marginal predictions and half-point natural line hits."""
    priors = get_default_priors(league="NBA")
    causal_dfs = []
    for df_p in player_dfs:
        feat = compute_causal_player_features(df_p, league="NBA")
        if "season" in feat.columns and target_seasons:
            if feat[feat["season"].isin(target_seasons)].empty:
                continue
        causal_dfs.append(feat)

    leg_frames = []
    for target_key in EVAL_STATS:
        stat_col = TARGET_METRICS[target_key]
        active_cols = get_active_feature_columns(league="NBA", stat_col=stat_col)
        stat_dfs = []
        for df_p in causal_dfs:
            pid = str(df_p["player_id"].iloc[0])
            pos = player_positions.get(pid, "SG")
            df_s = compute_target_rolling_features(
                df_p, stat_col=stat_col, position=pos, cached_priors=priors, league="NBA"
            )
            # Causal zero inflation rate up to T-1 strictly
            zeros = (df_s[stat_col] == 0.0).astype(float)
            df_s["causal_zero_rate"] = zeros.shift(1).expanding(min_periods=1).mean().fillna(0.0)
            df_s["position"] = pos
            stat_dfs.append(df_s)

        full_df = pd.concat(stat_dfs, ignore_index=True)
        sub_df = full_df[full_df["season"].isin(target_seasons)].copy()
        req = active_cols + ["target_residual", stat_col, "bayesian_roll_stat"]
        sub_df = sub_df.dropna(subset=req).copy()
        sub_df = sub_df.sort_values(["start_time", "game_id", "player_id"], kind="mergesort").reset_index(drop=True)

        ewma_arr = sub_df["bayesian_roll_stat"].to_numpy(dtype=float)
        actuals = sub_df[stat_col].to_numpy(dtype=float)

        if use_oof_internal:
            # Generate strictly out-of-fold predictions inside target_seasons (used for Fold 1 train copula)
            df_tr = sub_df.copy()
            for c in CATEGORICAL_ID_COLUMNS:
                df_tr[c] = df_tr[c].astype(str if c == "player_id" else int).astype("category")
            X_tr = df_tr[active_cols]
            y_tr = df_tr["target_residual"].to_numpy(dtype=float)
            oof_res = np.zeros(len(df_tr), dtype=float)
            for tr_i, va_i in _iter_grouped_time_splits(df_tr, n_splits=3):
                m_lgb = lgb.train(
                    {"objective": "regression_l1", "learning_rate": 0.05, "num_leaves": 31, "verbose": -1, "random_state": SEED},
                    lgb.Dataset(X_tr.iloc[tr_i], label=y_tr[tr_i]),
                    num_boost_round=120,
                )
                oof_res[va_i] = m_lgb.predict(X_tr.iloc[va_i])
            # Initial fold before first validation split uses pure causal ewma (residual=0)
            mu_pred = np.maximum(0.001, ewma_arr + oof_res)
            alpha_disp = 0.25 if target_key != "pts" else 0.40
        else:
            bundle = models[target_key]
            preds = predict_meta_ensemble(
                sub_df,
                active_cols,
                xgb_model=bundle.get("xgb"),
                lgb_model=bundle.get("lgb"),
                meta_data=bundle.get("meta", {}),
                ewma_baseline=ewma_arr,
                league="NBA",
                stat_col=stat_col,
                roll_min=sub_df["roll_min"].to_numpy(dtype=float),
                xgb_categories=bundle.get("xgb_categories"),
            )
            mu_pred = preds["meta_prediction"]
            alpha_disp = float((bundle.get("meta") or {}).get("alpha", 0.25))

        n_nb = 1.0 / max(alpha_disp, 1e-6)
        p_nb = n_nb / (n_nb + np.maximum(mu_pred, 1e-6))
        p_zero_nb = np.power(p_nb, n_nb)
        pi_raw = sub_df["causal_zero_rate"].to_numpy(dtype=float)
        pi_val = np.clip((pi_raw - p_zero_nb) / np.maximum(1e-9, 1.0 - p_zero_nb), 0.0, 0.90)

        lines = np.floor(np.maximum(0.5, ewma_arr)) + 0.5
        cutoffs = np.floor(lines).astype(int)
        nb_cdf = stats.nbinom.cdf(cutoffs, n_nb, p_nb)
        cdf = pi_val + (1.0 - pi_val) * nb_cdf
        prob_over = np.clip(1.0 - cdf, 1e-4, 1.0 - 1e-4)
        prob_under = np.clip(1.0 - prob_over, 1e-4, 1.0 - 1e-4)
        actual_over = (actuals > lines).astype(int)

        team_ids = pd.to_numeric(sub_df["team_id"], errors="coerce").fillna(0).astype(int).to_numpy()
        opp_ids = pd.to_numeric(sub_df["opponent_team_id"], errors="coerce").fillna(0).astype(int).to_numpy()
        game_ids = pd.to_numeric(sub_df["game_id"], errors="coerce").fillna(0).astype(int).to_numpy()

        team_str = np.array([team_strength_map.get((int(g), int(t)), 0.0) for g, t in zip(game_ids, team_ids)], dtype=float)
        opp_str = np.array([team_strength_map.get((int(g), int(o)), 0.0) for g, o in zip(game_ids, opp_ids)], dtype=float)

        leg_frames.append(
            pd.DataFrame(
                {
                    "game_id": game_ids,
                    "start_time": pd.to_datetime(sub_df["start_time"], utc=True).astype("int64") // 10**9,
                    "player_id": sub_df["player_id"].astype(str).values,
                    "team_id": team_ids,
                    "opponent_team_id": opp_ids,
                    "position": sub_df["position"].astype(str).values,
                    "stat": target_key,
                    "roll_min": sub_df["roll_min"].to_numpy(dtype=float),
                    "vacated_usage": sub_df["vacated_usage"].to_numpy(dtype=float),
                    "opp_pace_factor": sub_df["opp_pace_factor"].to_numpy(dtype=float),
                    "abs_team_strength_diff": np.abs(team_str - opp_str),
                    "ewma_baseline": ewma_arr,
                    "mu_pred": mu_pred,
                    "actual_value": actuals,
                    "line": lines,
                    "prob_over": prob_over,
                    "prob_under": prob_under,
                    "actual_over": actual_over,
                }
            )
        )

    return pd.concat(leg_frames, ignore_index=True)


def _construct_game_pairs(
    legs_df: pd.DataFrame,
    max_pairs_per_game: int = 48,
    min_ewma_minutes: float = 12.0,
    seed: int = SEED,
) -> pd.DataFrame:
    """Builds balanced same-game player pairs across teammate, opponent, same-pos, and OO/UU/OU regimes."""
    filtered = legs_df[legs_df["roll_min"] >= min_ewma_minutes].copy()
    filtered = filtered.sort_values(["game_id", "roll_min", "player_id", "stat"], ascending=[True, False, True, True]).reset_index(drop=True)

    pair_records: List[Dict[str, Any]] = []
    directions = (("Over", "Over"), ("Under", "Under"), ("Over", "Under"))

    for gid, g in filtered.groupby("game_id", sort=True):
        # Keep top 10 rotation legs sorted by minutes; dedup at player level first to ensure balanced cross-team matchups
        top_players = g.drop_duplicates(subset=["player_id"]).head(10)["player_id"].tolist()
        g_sub = g[g["player_id"].isin(top_players)]
        recs = g_sub.to_dict("records")
        m = len(recs)
        if m < 2:
            continue

        # Enumerate cross-player unordered pairs in game
        candidate_indices = []
        for i in range(m):
            for j in range(i + 1, m):
                if recs[i]["player_id"] != recs[j]["player_id"]:
                    candidate_indices.append((i, j))
        if not candidate_indices:
            continue

        rng = np.random.default_rng(seed + int(gid))
        n_pick = min(len(candidate_indices), max_pairs_per_game)
        chosen_pos = rng.choice(len(candidate_indices), size=n_pick, replace=False) if len(candidate_indices) > n_pick else np.arange(len(candidate_indices))

        for k_idx, c_idx in enumerate(chosen_pos):
            i, j = candidate_indices[int(c_idx)]
            l1, l2 = recs[i], recs[j]
            s1_side, s2_side = directions[k_idx % 3]

            u = float(l1["prob_over"] if s1_side == "Over" else l1["prob_under"])
            v = float(l2["prob_over"] if s2_side == "Over" else l2["prob_under"])
            y1 = int(l1["actual_over"] if s1_side == "Over" else (1 - l1["actual_over"]))
            y2 = int(l2["actual_over"] if s2_side == "Over" else (1 - l2["actual_over"]))

            same_team = int(l1["team_id"] == l2["team_id"])
            same_pos = int(l1["position"] == l2["position"])

            # Heuristic static production rho & copula output (identical to InferencePredictor.allocate)
            rel = "teammate" if same_team else "opponent"
            stat_full1 = TARGET_METRICS[l1["stat"]]
            stat_full2 = TARGET_METRICS[l2["stat"]]
            rel_map = DEFAULT_CORRELATIONS[rel]
            base_rho = float(rel_map.get((stat_full1, stat_full2)) or rel_map.get((stat_full2, stat_full1)) or 0.0)
            rho_static = base_rho if s1_side == s2_side else -base_rho
            p_static, _ = InferencePredictor._copula_joint_probability(
                u, v, rho_static, s1_side.lower(), s2_side.lower(), CopulaTypeEnum.auto
            )

            pair_records.append(
                {
                    "game_id": int(gid),
                    "game_timestamp": int(l1["start_time"]),
                    "is_same_team": same_team,
                    "is_same_position": same_pos,
                    "projected_pace_diff": abs(float(l1["opp_pace_factor"]) - float(l2["opp_pace_factor"])),
                    "mean_projected_pace": 0.5 * (float(l1["opp_pace_factor"]) + float(l2["opp_pace_factor"])),
                    "abs_team_strength_diff": float(l1["abs_team_strength_diff"]),
                    "combined_vacated_usage": float(l1["vacated_usage"]) + float(l2["vacated_usage"]),
                    "min_player_ewma_minutes": min(float(l1["roll_min"]), float(l2["roll_min"])),
                    "stat_pair_type": canonical_stat_pair(l1["stat"], l2["stat"]),
                    "direction_pair": canonical_direction_pair(s1_side, s2_side),
                    "side1": s1_side,
                    "side2": s2_side,
                    "u": u,
                    "v": v,
                    "y_joint": float(y1 & y2),
                    "p_independent": float(np.clip(u * v, 1e-6, 1.0 - 1e-6)),
                    "p_static_copula": float(np.clip(p_static, 1e-6, 1.0 - 1e-6)),
                }
            )

    return pd.DataFrame(pair_records)


def _metrics_dict(y: np.ndarray, p: np.ndarray) -> Dict[str, Any]:
    ece_val, _ = compute_ece(p, y, n_bins=10)
    return {
        "count": int(len(y)),
        "joint_logloss": round(compute_log_loss(p, y), 6),
        "joint_brier": round(compute_brier_score(p, y), 6),
        "joint_ece": round(ece_val, 6),
        "mean_bias": round(float(np.mean(y - p)), 6),
        "empirical_joint_rate": round(float(np.mean(y)), 6),
        "mean_pred_prob": round(float(np.mean(p)), 6),
    }


def _cluster_bootstrap_pvalues(
    game_ids: np.ndarray,
    y: np.ndarray,
    p_cond: np.ndarray,
    p_base: np.ndarray,
    n_resamples: int = 1000,
    seed: int = SEED,
) -> Dict[str, Any]:
    """Paired cluster-bootstrap by game_id on Joint Log-Loss & Joint Brier difference (cond - base < 0 means cond wins)."""
    eps = 1e-12
    pc = np.clip(p_cond, eps, 1.0 - eps)
    pb = np.clip(p_base, eps, 1.0 - eps)
    ll_diff = -(y * np.log(pc) + (1.0 - y) * np.log(1.0 - pc)) - (
        -(y * np.log(pb) + (1.0 - y) * np.log(1.0 - pb))
    )
    br_diff = ((pc - y) ** 2) - ((pb - y) ** 2)

    df_tmp = pd.DataFrame({"gid": game_ids, "ll_sum": ll_diff, "br_sum": br_diff, "n": 1})
    grp = df_tmp.groupby("gid", sort=True)[["ll_sum", "br_sum", "n"]].sum()
    ll_arr = grp["ll_sum"].to_numpy(dtype=float)
    br_arr = grp["br_sum"].to_numpy(dtype=float)
    n_arr = grp["n"].to_numpy(dtype=float)
    n_games = len(grp)

    rng = np.random.default_rng(seed)
    boot_ll = np.empty(n_resamples, dtype=float)
    boot_br = np.empty(n_resamples, dtype=float)
    for b in range(n_resamples):
        idx = rng.integers(0, n_games, size=n_games)
        denom = max(1.0, float(np.sum(n_arr[idx])))
        boot_ll[b] = float(np.sum(ll_arr[idx]) / denom)
        boot_br[b] = float(np.sum(br_arr[idx]) / denom)

    obs_ll = float(np.mean(ll_diff))
    obs_br = float(np.mean(br_diff))
    # One-sided test: H1 is delta < 0 (conditional beats baseline)
    p_ll = float(np.mean(boot_ll >= 0.0))
    p_br = float(np.mean(boot_br >= 0.0))
    return {
        "n_clusters_games": int(n_games),
        "n_resamples": int(n_resamples),
        "logloss_delta": round(obs_ll, 6),
        "logloss_ci95": [round(float(np.percentile(boot_ll, 2.5)), 6), round(float(np.percentile(boot_ll, 97.5)), 6)],
        "logloss_p_value": round(p_ll, 4),
        "brier_delta": round(obs_br, 6),
        "brier_ci95": [round(float(np.percentile(boot_br, 2.5)), 6), round(float(np.percentile(boot_br, 97.5)), 6)],
        "brier_p_value": round(p_br, 4),
    }


def _evaluate_pairs_suite(
    pairs_df: pd.DataFrame, copula_model: ConditionalCopulaModel
) -> Dict[str, Any]:
    X = build_pair_feature_matrix(pairs_df)
    u = pairs_df["u"].to_numpy(dtype=float)
    v = pairs_df["v"].to_numpy(dtype=float)
    dp = pairs_df["direction_pair"].astype(str).to_numpy()
    p_cond, tau_hat = copula_model.predict_joint_probs(X, u, v, dp == "OO", dp == "UU")

    y = pairs_df["y_joint"].to_numpy(dtype=float)
    p_ind = pairs_df["p_independent"].to_numpy(dtype=float)
    p_stat = pairs_df["p_static_copula"].to_numpy(dtype=float)
    gids = pairs_df["game_id"].to_numpy(dtype=int)

    overall = {
        "independent_zinb": _metrics_dict(y, p_ind),
        "static_copula": _metrics_dict(y, p_stat),
        "conditional_copula": _metrics_dict(y, p_cond),
        "tau_hat_stats": {
            "mean": round(float(np.mean(tau_hat)), 5),
            "std": round(float(np.std(tau_hat)), 5),
            "min": round(float(np.min(tau_hat)), 5),
            "max": round(float(np.max(tau_hat)), 5),
        },
    }

    q75_spread = float(np.quantile(pairs_df["abs_team_strength_diff"].to_numpy(dtype=float), 0.75))
    regime_masks = {
        "teammate_same_dir": (pairs_df["is_same_team"] == 1) & (pairs_df["direction_pair"].isin(["OO", "UU"])),
        "cross_team": pairs_df["is_same_team"] == 0,
        "same_pos_substitute": (pairs_df["is_same_team"] == 1) & (pairs_df["is_same_position"] == 1),
        "blowout_high_spread": pairs_df["abs_team_strength_diff"] >= q75_spread,
    }
    sub_regimes: Dict[str, Any] = {}
    for rname, mask in regime_masks.items():
        m_idx = mask.to_numpy(dtype=bool)
        if not np.any(m_idx):
            continue
        sub_regimes[rname] = {
            "independent_zinb": _metrics_dict(y[m_idx], p_ind[m_idx]),
            "static_copula": _metrics_dict(y[m_idx], p_stat[m_idx]),
            "conditional_copula": _metrics_dict(y[m_idx], p_cond[m_idx]),
        }

    bootstrap = {
        "vs_static_copula": _cluster_bootstrap_pvalues(gids, y, p_cond, p_stat),
        "vs_independent_zinb": _cluster_bootstrap_pvalues(gids, y, p_cond, p_ind),
    }
    return {"overall": overall, "sub_regimes": sub_regimes, "bootstrap": bootstrap}


def run_ablation() -> Dict[str, Any]:
    t0 = time.time()
    db_uri = _db_uri()

    raw_f1_train, pos_map, cutoff_f1_tr = _compile_fold_datasets(db_uri, ["2023-24"])
    raw_f1_eval, _, cutoff_f1_ev = _compile_fold_datasets(db_uri, ["2023-24", "2024-25"])
    raw_f2_eval, _, cutoff_f2_ev = _compile_fold_datasets(db_uri, ["2023-24", "2024-25", "2025-26"])

    str_f1_tr = _compute_causal_team_strength_map(raw_f1_train)
    str_f1_ev = _compute_causal_team_strength_map(raw_f1_eval)
    str_f2_ev = _compute_causal_team_strength_map(raw_f2_eval)

    models_f1 = load_models("models/wf_fold1")
    models_f2 = load_models("models/wf_fold2")

    # ---- Fold 1: Train copula on 2023-24 OOF ZINB, Evaluate on unseen 2024-25 ----
    logger.info("Building Fold 1 legs & pairs...")
    legs_f1_train = _prepare_zinb_legs_from_model(
        raw_f1_train, None, ["2023-24"], pos_map, str_f1_tr, use_oof_internal=True
    )
    legs_f1_test = _prepare_zinb_legs_from_model(
        raw_f1_eval, models_f1, ["2024-25"], pos_map, str_f1_ev, use_oof_internal=False
    )
    pairs_f1_train = _construct_game_pairs(legs_f1_train, max_pairs_per_game=48, seed=SEED)
    pairs_f1_test = _construct_game_pairs(legs_f1_test, max_pairs_per_game=48, seed=SEED)

    X_f1_tr = build_pair_feature_matrix(pairs_f1_train)
    dp_f1_tr = pairs_f1_train["direction_pair"].astype(str).to_numpy()
    copula_f1 = ConditionalCopulaModel.fit(
        X_f1_tr,
        pairs_f1_train["u"].to_numpy(dtype=float),
        pairs_f1_train["v"].to_numpy(dtype=float),
        dp_f1_tr == "OO",
        dp_f1_tr == "UU",
        pairs_f1_train["y_joint"].to_numpy(dtype=float),
        game_timestamps=pairs_f1_train["game_timestamp"].to_numpy(dtype=int),
        seed=SEED,
    )
    res_f1 = _evaluate_pairs_suite(pairs_f1_test, copula_f1)
    logger.info(
        f"[Fold 1 OOS 2024-25] Pairs={len(pairs_f1_test)} | "
        f"Indep LL={res_f1['overall']['independent_zinb']['joint_logloss']:.5f} "
        f"Static LL={res_f1['overall']['static_copula']['joint_logloss']:.5f} "
        f"Cond LL={res_f1['overall']['conditional_copula']['joint_logloss']:.5f} "
        f"(p={res_f1['bootstrap']['vs_static_copula']['logloss_p_value']:.4f})"
    )

    # ---- Fold 2: Train copula on 2023-24 + 2024-25 (clean F1 OOS for 24-25!), Evaluate on unseen 2025-26 ----
    logger.info("Building Fold 2 legs & pairs...")
    legs_f2_train = pd.concat([legs_f1_train, legs_f1_test], ignore_index=True)
    legs_f2_test = _prepare_zinb_legs_from_model(
        raw_f2_eval, models_f2, ["2025-26"], pos_map, str_f2_ev, use_oof_internal=False
    )
    pairs_f2_train = pd.concat([pairs_f1_train, pairs_f1_test], ignore_index=True)
    pairs_f2_test = _construct_game_pairs(legs_f2_test, max_pairs_per_game=48, seed=SEED)

    X_f2_tr = build_pair_feature_matrix(pairs_f2_train)
    dp_f2_tr = pairs_f2_train["direction_pair"].astype(str).to_numpy()
    copula_f2 = ConditionalCopulaModel.fit(
        X_f2_tr,
        pairs_f2_train["u"].to_numpy(dtype=float),
        pairs_f2_train["v"].to_numpy(dtype=float),
        dp_f2_tr == "OO",
        dp_f2_tr == "UU",
        pairs_f2_train["y_joint"].to_numpy(dtype=float),
        game_timestamps=pairs_f2_train["game_timestamp"].to_numpy(dtype=int),
        seed=SEED,
    )
    res_f2 = _evaluate_pairs_suite(pairs_f2_test, copula_f2)
    logger.info(
        f"[Fold 2 OOS 2025-26] Pairs={len(pairs_f2_test)} | "
        f"Indep LL={res_f2['overall']['independent_zinb']['joint_logloss']:.5f} "
        f"Static LL={res_f2['overall']['static_copula']['joint_logloss']:.5f} "
        f"Cond LL={res_f2['overall']['conditional_copula']['joint_logloss']:.5f} "
        f"(p={res_f2['bootstrap']['vs_static_copula']['logloss_p_value']:.4f})"
    )

    # Compute exact clean Fold 2 marginal calibration Brier / LogLoss / ECE for ModelRegistry champion metadata
    marg_y = legs_f2_test["actual_over"].to_numpy(dtype=float)
    marg_p = legs_f2_test["prob_over"].to_numpy(dtype=float)
    marg_ece, _ = compute_ece(marg_p, marg_y, n_bins=10)
    marginal_fold2_calibration = {
        "count": int(len(marg_y)),
        "brier_score": round(compute_brier_score(marg_p, marg_y), 6),
        "log_loss": round(compute_log_loss(marg_p, marg_y), 6),
        "ece": round(marg_ece, 6),
    }

    beats_f1 = (
        res_f1["overall"]["conditional_copula"]["joint_logloss"] < res_f1["overall"]["static_copula"]["joint_logloss"]
        and res_f1["overall"]["conditional_copula"]["joint_brier"] < res_f1["overall"]["static_copula"]["joint_brier"]
    )
    beats_f2 = (
        res_f2["overall"]["conditional_copula"]["joint_logloss"] < res_f2["overall"]["static_copula"]["joint_logloss"]
        and res_f2["overall"]["conditional_copula"]["joint_brier"] < res_f2["overall"]["static_copula"]["joint_brier"]
    )
    default_conditional_active = bool(beats_f1 and beats_f2)

    copula_f2.metadata["default_enabled_by_ablation"] = default_conditional_active
    copula_f2.metadata["train_seasons"] = ["2023-24", "2024-25"]
    copula_f2.metadata["feature_schema_version"] = FEATURE_SCHEMA_VERSION
    copula_f2.save(COPULA_MODEL_OUT_PATH)

    # Top magnitude learned feature weights for interpretability
    top_weights = sorted(
        zip(copula_f2.feature_names, copula_f2.coef.tolist()),
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    report = {
        "experiment": "Experiment 15: Conditional Copula theta(x) Walk-Forward Joint Calibration Ablation",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "elapsed_seconds": round(time.time() - t0, 2),
        "cutoffs": {
            "fold1_train": cutoff_f1_tr,
            "fold1_eval": cutoff_f1_ev,
            "fold2_eval": cutoff_f2_ev,
        },
        "default_conditional_active": default_conditional_active,
        "beats_fold1_static": beats_f1,
        "beats_fold2_static": beats_f2,
        "fold2_marginal_calibration": marginal_fold2_calibration,
        "copula_artifact_path": COPULA_MODEL_OUT_PATH,
        "fold2_top_learned_weights": [
            {"feature": name, "weight": round(float(w), 6)} for name, w in top_weights[:12]
        ],
        "fold1": res_f1,
        "fold2": res_f2,
    }

    os.makedirs(os.path.dirname(os.path.abspath(REPORT_PATH)), exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(
        f"Saved Experiment 15 report to {REPORT_PATH} ({report['elapsed_seconds']}s). "
        f"default_conditional_active={default_conditional_active}"
    )
    return report


if __name__ == "__main__":
    run_ablation()
