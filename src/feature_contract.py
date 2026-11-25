"""Unified offline-online feature parity contract and strictly point-in-time feature transforms.

Both batch training (`src/train_global_model.py`), vectorized evaluation (`src/evaluate_fast.py`),
sequential replay (`src/historical_replay_core.py`), and online inference (`src/serving/`) call the
exact same functions defined here to guarantee zero training-serving skew and zero future leakage.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import LEAGUE_CONFIG
from src.models import apply_bayesian_shrinkage, get_default_priors, get_positional_prior

FEATURE_SCHEMA_VERSION = "1.0.0"

TARGET_METRICS: Dict[str, str] = {
    "pts": "points",
    "reb": "rebounds",
    "ast": "assists",
    "fg3m": "three_pointers_made",
    "stl": "steals",
    "blk": "blocks",
    "tov": "turnovers",
}

STAT_COL_TO_TARGET: Dict[str, str] = {v: k for k, v in TARGET_METRICS.items()}

# Cohort-bucketed EWMA tiers: alphas based on structural game-to-game variance.
PLAYER_TIERS: Dict[str, Dict[str, float]] = {
    "star": {"min_minutes": 30.0, "min_usage": 0.25, "ewma_alpha": 0.20},
    "starter": {"min_minutes": 22.0, "min_usage": 0.18, "ewma_alpha": 0.25},
    "rotation": {"min_minutes": 12.0, "min_usage": 0.12, "ewma_alpha": 0.30},
    "fringe": {"min_minutes": 0.0, "min_usage": 0.0, "ewma_alpha": 0.35},
}

TIER_ORDER: List[str] = ["star", "starter", "rotation", "fringe"]
ALL_TIER_ALPHAS: Tuple[float, ...] = (0.20, 0.25, 0.30, 0.35)

FEATURE_COLUMNS: List[str] = [
    "roll_min",
    "roll_stat",
    "roll_usg",
    "days_rest",
    "is_back_to_back",
    "opp_def_rating",
    "opp_pace_factor",
    "opp_pos_pts_allowed",
    "vacated_usage",
    "roll_potential_assists",
    "roll_touches",
    "roll_passes_made",
    "roll_time_of_possession_sec",
    "roll_defender_distance",
    "is_home_game",
    "pace_differential",
    "roll_stat_var",
    "roll_residual",
    "roll_ts_pct",
    "roll_reb_rate",
    "roll_ast_rate",
    "roll_shorthanded_points",
    "roll_plus_minus",
    "roll_usg_efficiency",
    "player_id",
    "team_id",
    "opponent_team_id",
    "player_tier",
]

WNBA_FEATURE_COLUMNS: List[str] = [
    "roll_min",
    "roll_stat",
    "roll_usg",
    "days_rest",
    "is_back_to_back",
    "opp_def_rating",
    "opp_pace_factor",
    "opp_pos_pts_allowed",
    "is_home_game",
    "pace_differential",
    "roll_stat_var",
    "roll_residual",
    "roll_ts_pct",
    "roll_reb_rate",
    "roll_ast_rate",
    "roll_plus_minus",
    "roll_usg_efficiency",
    "roll_fga",
    "roll_fta",
    "roll_expected_points",
    "player_tier",
]

CATEGORICAL_ID_COLUMNS: Tuple[str, ...] = ("player_id", "team_id", "opponent_team_id")

OUTCOME_COLUMNS: Tuple[str, ...] = (
    "minutes_played",
    "points",
    "rebounds",
    "assists",
    "turnovers",
    "steals",
    "blocks",
    "three_pointers_made",
    "field_goals_attempted",
    "free_throws_attempted",
    "plus_minus",
    "usage_rate",
    "potential_assists",
    "touches",
    "passes_made",
    "time_of_possession_sec",
    "avg_defender_distance_ft",
    "shorthanded_points",
)

PREGAME_CONTEXT_COLUMNS: Tuple[str, ...] = (
    "game_id",
    "start_time",
    "season",
    "home_team_id",
    "away_team_id",
    "player_id",
    "team_id",
    "opponent_team_id",
    "days_rest",
    "is_back_to_back",
    "is_home_game",
    "opp_def_rating",
    "opp_pace_factor",
    "opp_pos_pts_allowed",
    "pace_differential",
    "vacated_usage",
)


def assign_player_tier(avg_minutes: float, avg_usage: float) -> str:
    """Assigns a player to a structural tier using point-in-time historical averages."""
    for tier_name in TIER_ORDER:
        tier = PLAYER_TIERS[tier_name]
        if avg_minutes >= tier["min_minutes"] and avg_usage >= tier["min_usage"]:
            return tier_name
    return "fringe"


def get_tier_alpha(tier_name: str) -> float:
    """Returns the EWMA smoothing alpha for a given structural player tier."""
    return PLAYER_TIERS.get(tier_name, PLAYER_TIERS["fringe"])["ewma_alpha"]


def get_active_feature_columns(league: str = "NBA", stat_col: str = "points") -> List[str]:
    """Returns the canonical ordered feature list for a league and target statistic."""
    league_key = (league or "NBA").upper()
    if league_key == "WNBA":
        cols = list(WNBA_FEATURE_COLUMNS)
        if stat_col == "points":
            cols.extend(["points_rolling_std_dev", "usage_x_minutes_interaction"])
        return cols
    return list(FEATURE_COLUMNS)


def infer_season_from_timestamp(dt: Any) -> str:
    """Infers NBA/WNBA season tag from a timestamp."""
    if pd.isna(dt):
        return "2025-26"
    y, m = dt.year, dt.month
    return f"{y}-{str(y + 1)[-2:]}" if m >= 9 else f"{y - 1}-{str(y)[-2:]}"


def _causal_tier_ewma(
    df: pd.DataFrame,
    col: str,
    default: Optional[float] = None,
    var: bool = False,
) -> pd.Series:
    """Computes a strictly shifted, season-partitioned EWMA using each row's causal tier_alpha."""
    tier_alphas = df["tier_alpha"].to_numpy(dtype=float)
    out = np.full(len(df), np.nan, dtype=float)

    for alpha in ALL_TIER_ALPHAS:
        mask = np.isclose(tier_alphas, alpha, atol=1e-9)
        if not np.any(mask):
            continue
        if var:
            vals = df.groupby("season", sort=False)[col].transform(
                lambda s: s.shift(1).ewm(alpha=alpha, adjust=False).var()
            )
        else:
            vals = df.groupby("season", sort=False)[col].transform(
                lambda s: s.shift(1).ewm(alpha=alpha, adjust=False).mean()
            )
        vals_arr = vals.to_numpy(dtype=float)
        out[mask] = vals_arr[mask]

    res = pd.Series(out, index=df.index, dtype=float)
    if default is not None:
        res = res.fillna(default)
    return res


def compute_causal_player_features(df_player: pd.DataFrame, league: str = "NBA") -> pd.DataFrame:
    """Computes strictly point-in-time (T-1) player rolling, tier, rate, and schedule features.

    All rolling windows, tier assignments, and usage/rate features for game T depend strictly on
    games 0..T-1 and pre-game metadata of game T. Perturbing ground-truth outcome columns at
    game T or T+1..T+k leaves row T's derived features bitwise unchanged.
    """
    if df_player.empty:
        return df_player.copy()

    league_key = (league or "NBA").upper()
    reg_min = float(LEAGUE_CONFIG.get(league_key, LEAGUE_CONFIG["NBA"])["regulation_minutes"])

    df = df_player.copy()
    if "start_time" in df.columns:
        df["start_time"] = pd.to_datetime(df["start_time"], utc=True)
        sort_cols = ["start_time", "game_id"] if "game_id" in df.columns else ["start_time"]
        df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    if "season" not in df.columns:
        if "start_time" in df.columns:
            df["season"] = df["start_time"].apply(infer_season_from_timestamp)
        else:
            df["season"] = "2025-26"

    # Compute days_rest / is_back_to_back only where missing, preserving pre-supplied values.
    if "days_rest" not in df.columns or df["days_rest"].isna().any():
        if "start_time" in df.columns:
            computed_rest = []
            for i in range(len(df)):
                if (
                    i == 0
                    or (df.loc[i, "season"] != df.loc[i - 1, "season"])
                    or pd.isna(df.loc[i, "start_time"])
                    or pd.isna(df.loc[i - 1, "start_time"])
                ):
                    computed_rest.append(3.0)
                else:
                    computed_rest.append(float((df.loc[i, "start_time"] - df.loc[i - 1, "start_time"]).days))
            rest_series = pd.Series(computed_rest, index=df.index, dtype=float)
            if "days_rest" not in df.columns:
                df["days_rest"] = rest_series
            else:
                df["days_rest"] = pd.to_numeric(df["days_rest"], errors="coerce").fillna(rest_series)
        else:
            df["days_rest"] = 3.0
    else:
        df["days_rest"] = pd.to_numeric(df["days_rest"], errors="coerce").fillna(3.0)

    if "is_back_to_back" not in df.columns or df["is_back_to_back"].isna().any():
        b2b_calc = (df["days_rest"] == 1.0).astype(float)
        if "is_back_to_back" not in df.columns:
            df["is_back_to_back"] = b2b_calc
        else:
            df["is_back_to_back"] = pd.to_numeric(df["is_back_to_back"], errors="coerce").fillna(b2b_calc)
    else:
        df["is_back_to_back"] = pd.to_numeric(df["is_back_to_back"], errors="coerce").fillna(0.0)

    # Ensure raw stat / tracking columns exist (fillna on historical rows; row T is shifted out anyway).
    raw_defaults = [
        ("points", 0.0),
        ("rebounds", 0.0),
        ("assists", 0.0),
        ("turnovers", 0.0),
        ("steals", 0.0),
        ("blocks", 0.0),
        ("three_pointers_made", 0.0),
        ("minutes_played", 0.0),
        ("potential_assists", 0.0),
        ("touches", 0.0),
        ("passes_made", 0.0),
        ("time_of_possession_sec", 0.0),
        ("avg_defender_distance_ft", 4.5),
        ("shorthanded_points", 0.0),
        ("plus_minus", 0.0),
        ("field_goals_attempted", 0.0),
        ("free_throws_attempted", 0.0),
    ]
    for col, default in raw_defaults:
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)

    # Ensure contextual columns have safe defaults if absent.
    context_defaults = [
        ("opp_def_rating", 100.0 if league_key == "WNBA" else 115.0),
        ("opp_pace_factor", 80.0 if league_key == "WNBA" else 99.0),
        ("opp_pos_pts_allowed", 22.0),
        ("vacated_usage", 0.0),
        ("is_home_game", 0.0),
        ("pace_differential", 0.0),
    ]
    for col, default in context_defaults:
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)

    # Usage rate and per-game derived efficiency metrics (only used via .shift(1)).
    df["usage_rate"] = (
        (df["points"] + 1.5 * df["assists"] + df["turnovers"])
        / (df["minutes_played"] * 2.0 + 1e-5)
    ).clip(0.05, 0.45)

    df["ts_pct"] = df["points"] / (
        2.0 * (df["field_goals_attempted"] + 0.44 * df["free_throws_attempted"]) + 1e-5
    )
    df["reb_rate"] = (df["rebounds"] / (df["minutes_played"] + 1e-5)) * reg_min
    df["ast_rate"] = (df["assists"] / (df["minutes_played"] + 1e-5)) * reg_min
    df["usg_efficiency"] = df["points"] / ((df["usage_rate"] * df["minutes_played"]) + 1e-5)

    # Strictly point-in-time cohort tiering (0..T-1 expanding means ONLY).
    prior_avg_min = df["minutes_played"].shift(1).expanding(min_periods=1).mean().fillna(0.0)
    prior_avg_usg = df["usage_rate"].shift(1).expanding(min_periods=1).mean().fillna(0.0)

    tier_names = [
        assign_player_tier(float(m), float(u))
        for m, u in zip(prior_avg_min.to_numpy(), prior_avg_usg.to_numpy())
    ]
    df["player_tier"] = [TIER_ORDER.index(t) for t in tier_names]
    df["tier_alpha"] = [get_tier_alpha(t) for t in tier_names]

    # Causal season-partitioned EWMAs driven by each row's own point-in-time tier_alpha.
    df["roll_min"] = _causal_tier_ewma(df, "minutes_played", default=None)
    df["roll_usg"] = _causal_tier_ewma(df, "usage_rate", default=None)

    df["roll_potential_assists"] = _causal_tier_ewma(df, "potential_assists", default=0.0)
    df["roll_touches"] = _causal_tier_ewma(df, "touches", default=0.0)
    df["roll_passes_made"] = _causal_tier_ewma(df, "passes_made", default=0.0)
    df["roll_time_of_possession_sec"] = _causal_tier_ewma(df, "time_of_possession_sec", default=0.0)
    df["roll_defender_distance"] = _causal_tier_ewma(df, "avg_defender_distance_ft", default=4.5)
    df["roll_shorthanded_points"] = _causal_tier_ewma(df, "shorthanded_points", default=0.0)
    df["roll_plus_minus"] = _causal_tier_ewma(df, "plus_minus", default=0.0)

    df["roll_ts_pct"] = _causal_tier_ewma(df, "ts_pct", default=0.0)
    df["roll_reb_rate"] = _causal_tier_ewma(df, "reb_rate", default=0.0)
    df["roll_ast_rate"] = _causal_tier_ewma(df, "ast_rate", default=0.0)
    df["roll_usg_efficiency"] = _causal_tier_ewma(df, "usg_efficiency", default=0.0)

    df["roll_fga"] = _causal_tier_ewma(df, "field_goals_attempted", default=0.0)
    df["roll_fta"] = _causal_tier_ewma(df, "free_throws_attempted", default=0.0)
    df["roll_expected_points"] = 2.0 * df["roll_ts_pct"] * (df["roll_fga"] + 0.44 * df["roll_fta"])

    df["points_rolling_std_dev"] = (
        df.groupby("season", sort=False)["points"]
        .transform(lambda s: s.shift(1).rolling(5, min_periods=1).std())
        .fillna(0.0)
    )
    df["usage_x_minutes_interaction"] = df["roll_usg"].fillna(0.0) * df["roll_min"].fillna(0.0)

    return df


def compute_target_rolling_features(
    df_player: pd.DataFrame,
    stat_col: str,
    position: str = "SG",
    cached_priors: Optional[dict] = None,
    league: str = "NBA",
) -> pd.DataFrame:
    """Computes target-specific causal rolling stats, Bayesian baseline, and causal roll_residual."""
    df = df_player.copy()
    if df.empty:
        return df

    league_key = (league or "NBA").upper()
    if "tier_alpha" not in df.columns or "roll_min" not in df.columns:
        df = compute_causal_player_features(df, league=league_key)

    if stat_col not in df.columns:
        df[stat_col] = 0.0
    else:
        df[stat_col] = pd.to_numeric(df[stat_col], errors="coerce")

    df["roll_stat"] = _causal_tier_ewma(df, stat_col, default=None)
    df["roll_stat_var"] = _causal_tier_ewma(df, stat_col, default=0.0, var=True)

    priors = cached_priors or get_default_priors(league=league_key)
    prior_per_36 = get_positional_prior(stat_col, position or "SG", cached_priors=priors)

    df["roll_stat_per_36"] = (df["roll_stat"] / (df["roll_min"] + 1e-5)) * 36.0
    # Strictly shifted career cumulative minutes up to T-1 (zero current-game leakage).
    cum_min = df["minutes_played"].shift(1).cumsum().fillna(0.0)

    bayesian_per_36 = []
    for rs_36, cm in zip(df["roll_stat_per_36"].to_numpy(), cum_min.to_numpy()):
        if np.isnan(rs_36):
            bayesian_per_36.append(np.nan)
        else:
            bayesian_per_36.append(apply_bayesian_shrinkage(float(rs_36), float(cm), prior_per_36, K=500.0))

    df["bayesian_roll_per_36"] = bayesian_per_36
    df["bayesian_roll_stat"] = (df["bayesian_roll_per_36"] / 36.0) * df["roll_min"]

    if league_key == "WNBA":
        if stat_col == "points":
            raw_residual = (df[stat_col] / (df["minutes_played"] + 1e-5)) - (
                df["roll_stat"] / (df["roll_min"] + 1e-5)
            )
            df["target_residual"] = np.clip(raw_residual, -2.0, 2.0)
        else:
            df["target_residual"] = df[stat_col] - df["roll_stat"]
    else:
        df["target_residual"] = df[stat_col] - df["bayesian_roll_stat"]

    df["roll_residual"] = _causal_tier_ewma(df, "target_residual", default=0.0)
    return df


def build_online_feature_row(
    player_id: str,
    history_df: pd.DataFrame,
    target_context: Dict[str, Any],
    stat_col: str = "points",
    position: str = "SG",
    cached_priors: Optional[dict] = None,
    league: str = "NBA",
) -> Tuple[pd.Series, List[str]]:
    """Constructs a single online inference feature row from prior history + pre-game context.

    Appends a target row whose ground-truth outcome fields are masked to NaN so no current-game
    outcome can ever be read, then runs the exact same shared causal pipeline used in batch.
    """
    league_key = (league or "NBA").upper()
    active_cols = get_active_feature_columns(league=league_key, stat_col=stat_col)

    hist = history_df.copy() if history_df is not None and not history_df.empty else pd.DataFrame()
    target_row: Dict[str, Any] = {col: np.nan for col in OUTCOME_COLUMNS}
    for ctx_col in PREGAME_CONTEXT_COLUMNS:
        if ctx_col in target_context:
            target_row[ctx_col] = target_context[ctx_col]
    target_row.setdefault("player_id", str(player_id))
    if "start_time" not in target_row or pd.isna(target_row["start_time"]):
        target_row["start_time"] = pd.Timestamp.now(tz="UTC")
    if "team_id" not in target_row or pd.isna(target_row["team_id"]):
        target_row["team_id"] = (
            int(hist["team_id"].dropna().iloc[-1])
            if not hist.empty and "team_id" in hist.columns and not hist["team_id"].dropna().empty
            else 0
        )
    if "opponent_team_id" not in target_row or pd.isna(target_row["opponent_team_id"]):
        target_row["opponent_team_id"] = 0
    if "season" not in target_row or pd.isna(target_row["season"]):
        if "start_time" in target_row and not pd.isna(target_row["start_time"]):
            target_row["season"] = infer_season_from_timestamp(pd.to_datetime(target_row["start_time"], utc=True))
        elif not hist.empty and "season" in hist.columns:
            target_row["season"] = hist["season"].iloc[-1]
        else:
            target_row["season"] = "2025-26"

    combined = pd.concat([hist, pd.DataFrame([target_row])], ignore_index=True)
    if "player_id" not in combined.columns:
        combined["player_id"] = str(player_id)
    else:
        combined["player_id"] = combined["player_id"].fillna(str(player_id))

    if "team_id" in target_context and "team_id" in combined.columns:
        combined.loc[combined.index[-1], "team_id"] = target_context["team_id"]
    if "opponent_team_id" in target_context and "opponent_team_id" in combined.columns:
        combined.loc[combined.index[-1], "opponent_team_id"] = target_context["opponent_team_id"]

    featured = compute_causal_player_features(combined, league=league_key)
    featured = compute_target_rolling_features(
        featured,
        stat_col=stat_col,
        position=position,
        cached_priors=cached_priors,
        league=league_key,
    )
    return featured.iloc[-1], active_cols


def extract_xgb_categories_from_json(xgb_json_path: str) -> Dict[str, List[Any]]:
    """Extracts categorical vocabularies from a saved XGBoost JSON artifact when present."""
    if not xgb_json_path or not os.path.exists(xgb_json_path):
        return {}
    try:
        with open(xgb_json_path, "r") as f:
            payload = json.load(f)
        learner = payload.get("learner", {})
        feature_names = learner.get("feature_names", [])
        feature_types = learner.get("feature_types", [])
        cats = learner.get("gradient_booster", {}).get("model", {}).get("cats")
        if not cats:
            return {}
        enc = cats.get("enc", [])
        result: Dict[str, List[Any]] = {}
        for idx, (fname, ftype) in enumerate(zip(feature_names, feature_types)):
            if ftype != "c" or idx >= len(enc):
                continue
            entry = enc[idx]
            offsets = entry.get("offsets", [])
            values = entry.get("values", [])
            if offsets and values:
                decoded = []
                for start, end in zip(offsets[:-1], offsets[1:]):
                    decoded.append(bytes(values[start:end]).decode("utf-8"))
                result[fname] = decoded
            elif values:
                result[fname] = list(values)
        return result
    except Exception:
        return {}


def prepare_xgb_lgb_frames(
    df: pd.DataFrame,
    feature_cols: List[str],
    xgb_model: Any = None,
    lgb_model: Any = None,
    xgb_categories: Optional[Dict[str, List[Any]]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Formats feature DataFrames to match loaded XGBoost and LightGBM schema expectations.

    Handles both categorical-trained artifacts (mapping unseen cold-start IDs cleanly to NaN)
    and float-trained artifacts seamlessly.
    """
    base = df[feature_cols].copy()

    # Determine XGBoost expected feature types
    xgb_ft_map: Dict[str, str] = {}
    if xgb_model is not None and hasattr(xgb_model, "get_booster"):
        try:
            booster = xgb_model.get_booster()
            names = booster.feature_names or feature_cols
            types = booster.feature_types or []
            xgb_ft_map = dict(zip(names, types))
        except Exception:
            xgb_ft_map = {}

    lgb_cats = getattr(lgb_model, "pandas_categorical", None) or []

    # Build XGB frame
    xgb_df = base.copy()
    for col in feature_cols:
        if col in CATEGORICAL_ID_COLUMNS:
            if xgb_ft_map.get(col) == "c":
                known = (xgb_categories or {}).get(col)
                if not known and lgb_cats:
                    cat_idx = [c for c in CATEGORICAL_ID_COLUMNS if c in feature_cols].index(col)
                    if cat_idx < len(lgb_cats):
                        known = lgb_cats[cat_idx]
                if known:
                    if isinstance(known[0], str):
                        xgb_df[col] = pd.Categorical(xgb_df[col].astype(str), categories=known)
                    else:
                        xgb_df[col] = pd.Categorical(
                            pd.to_numeric(xgb_df[col], errors="coerce"), categories=known
                        )
                else:
                    xgb_df[col] = xgb_df[col].astype(str).astype("category")
            else:
                xgb_df[col] = pd.to_numeric(xgb_df[col], errors="coerce").fillna(0.0).astype(float)
        else:
            xgb_df[col] = pd.to_numeric(xgb_df[col], errors="coerce").astype(float)

    # Build LGB frame
    lgb_df = base.copy()
    cat_cols_present = [c for c in CATEGORICAL_ID_COLUMNS if c in feature_cols]
    for col in feature_cols:
        if col in CATEGORICAL_ID_COLUMNS:
            if lgb_cats and len(lgb_cats) >= len(cat_cols_present):
                cat_idx = cat_cols_present.index(col)
                known = lgb_cats[cat_idx]
                if known and isinstance(known[0], str):
                    lgb_df[col] = pd.Categorical(lgb_df[col].astype(str), categories=known)
                elif known:
                    lgb_df[col] = pd.Categorical(
                        pd.to_numeric(lgb_df[col], errors="coerce"), categories=known
                    )
                else:
                    lgb_df[col] = lgb_df[col].astype("category")
            else:
                lgb_df[col] = pd.to_numeric(lgb_df[col], errors="coerce").fillna(0.0).astype(float)
        else:
            lgb_df[col] = pd.to_numeric(lgb_df[col], errors="coerce").astype(float)

    return xgb_df, lgb_df


def predict_meta_ensemble(
    df: pd.DataFrame,
    feature_cols: List[str],
    xgb_model: Any,
    lgb_model: Any,
    meta_data: Optional[dict],
    ewma_baseline: np.ndarray,
    league: str = "NBA",
    stat_col: str = "points",
    roll_min: Optional[np.ndarray] = None,
    xgb_categories: Optional[Dict[str, List[Any]]] = None,
) -> Dict[str, np.ndarray]:
    """Runs identical Meta-Ensemble prediction across batch evaluation and online serving."""
    xgb_df, lgb_df = prepare_xgb_lgb_frames(
        df,
        feature_cols,
        xgb_model=xgb_model,
        lgb_model=lgb_model,
        xgb_categories=xgb_categories,
    )

    n = len(df)
    xgb_res = np.asarray(xgb_model.predict(xgb_df), dtype=float) if xgb_model is not None else np.zeros(n, dtype=float)
    lgb_res = np.asarray(lgb_model.predict(lgb_df), dtype=float) if lgb_model is not None else xgb_res.copy()

    ewma_arr = np.asarray(ewma_baseline, dtype=float)
    meta_dict = meta_data or {}
    if "weights" in meta_dict and "intercept" in meta_dict:
        w = meta_dict["weights"]
        intercept = float(meta_dict["intercept"])
        if len(w) == 3:
            meta_residual = intercept + (float(w[0]) * xgb_res) + (float(w[1]) * lgb_res) + (float(w[2]) * ewma_arr)
        else:
            meta_residual = intercept + (float(w[0]) * xgb_res) + (float(w[1]) * lgb_res)
    else:
        meta_residual = 0.5 * (xgb_res + lgb_res)

    league_key = (league or "NBA").upper()
    if league_key == "WNBA" and stat_col == "points" and roll_min is not None:
        meta_pred = ewma_arr + (meta_residual * np.asarray(roll_min, dtype=float))
    else:
        meta_pred = ewma_arr + meta_residual

    meta_pred = np.maximum(0.001, meta_pred)

    return {
        "meta_prediction": meta_pred,
        "meta_residual": meta_residual,
        "xgb_residual": xgb_res,
        "lgb_residual": lgb_res,
        "ewma_baseline": ewma_arr,
    }

