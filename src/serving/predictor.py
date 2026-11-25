"""Inference predictor engine orchestrating feature parity, Meta-Ensemble, ZINB, Copula, and C++ Eigen Kelly."""

from __future__ import annotations

import json
import logging
import math
import os
import subprocess
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats

from src.execution import DEFAULT_CORRELATIONS
from src.feature_contract import (
    FEATURE_COLUMNS,
    FEATURE_SCHEMA_VERSION,
    TARGET_METRICS,
    build_online_feature_row,
    extract_xgb_categories_from_json,
    get_active_feature_columns,
    predict_meta_ensemble,
)
from src.models import MarketMicrostructure, get_default_priors, get_positional_prior
from src.risk import RiskManager, _HAS_CPP_KELLY, _quant_core
from src.serving.schemas import (
    AllocateCandidateOutcome,
    AllocateRequest,
    AllocateResponse,
    BatchPredictRequest,
    BatchPredictResponse,
    CopulaTypeEnum,
    HealthResponse,
    JointPairPricing,
    PredictionItemRequest,
    PredictionItemResponse,
    ThresholdSurvival,
    ZINBParams,
)

logger = logging.getLogger("InferencePredictor")


def _resolve_git_sha() -> str:
    env_sha = os.getenv("GIT_SHA")
    if env_sha:
        return env_sha.strip()
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        return out.decode("utf-8").strip() or "unknown"
    except Exception:
        return "unknown"


class InferencePredictor:
    """Stateless-per-request predictor holding loaded ensemble artifacts and Redis connection."""

    def __init__(
        self,
        model_dir: Optional[str] = None,
        redis_client: Any = None,
        auto_connect_redis: bool = True,
    ):
        resolved_dir = model_dir or os.getenv("MODEL_DIR")
        if not resolved_dir:
            resolved_dir = "models/wf_fold2" if os.path.exists("models/wf_fold2") else "models"
        self.model_dir = resolved_dir
        self.git_sha = _resolve_git_sha()
        self.model_version = os.getenv(
            "MODEL_VERSION", f"{os.path.basename(os.path.abspath(self.model_dir))}:{self.git_sha}"
        )
        self.models: Dict[str, Dict[str, Any]] = {}
        self.known_players: Dict[str, set] = {}
        self.cached_priors = {
            "NBA": get_default_priors("NBA"),
            "WNBA": get_default_priors("WNBA"),
        }
        self._load_artifacts()

        self.redis_client = redis_client
        if self.redis_client is None and auto_connect_redis:
            self.redis_client = self._init_default_redis()

    @staticmethod
    def _init_default_redis() -> Any:
        try:
            import redis

            return redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", 6379)),
                password=os.getenv("REDIS_PASSWORD") or None,
                decode_responses=True,
                socket_connect_timeout=0.5,
                socket_timeout=0.5,
            )
        except Exception as exc:
            logger.warning(f"Failed initializing Redis client: {exc}")
            return None

    def _load_artifacts(self) -> None:
        for target_key in TARGET_METRICS:
            entry: Dict[str, Any] = {}
            xgb_path = os.path.join(self.model_dir, f"global_x{target_key}_xgb.json")
            if not os.path.exists(xgb_path):
                xgb_path = os.path.join(self.model_dir, f"global_x{target_key}.json")
            lgb_path = os.path.join(self.model_dir, f"global_x{target_key}_lgb.txt")
            meta_path = os.path.join(self.model_dir, f"global_x{target_key}_meta.json")

            known_set: set = set()
            if os.path.exists(xgb_path):
                try:
                    bst = xgb.XGBRegressor()
                    bst.load_model(xgb_path)
                    entry["xgb"] = bst
                    xgb_cats = extract_xgb_categories_from_json(xgb_path)
                    entry["xgb_categories"] = xgb_cats
                    known_set.update(str(p) for p in xgb_cats.get("player_id", []))
                except Exception as exc:
                    logger.warning(f"Unable to load XGBoost model for '{target_key}': {exc}")

            if os.path.exists(lgb_path):
                try:
                    lgb_model = lgb.Booster(model_file=lgb_path)
                    entry["lgb"] = lgb_model
                    lgb_cats = getattr(lgb_model, "pandas_categorical", None) or []
                    if lgb_cats:
                        known_set.update(str(p) for p in lgb_cats[0])
                except Exception as exc:
                    logger.warning(f"Unable to load LightGBM model for '{target_key}': {exc}")

            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r") as f:
                        entry["meta"] = json.load(f)
                except Exception as exc:
                    logger.warning(f"Unable to load meta JSON for '{target_key}': {exc}")

            if entry:
                self.models[target_key] = entry
                self.known_players[target_key] = known_set

    def check_redis_health(self) -> Tuple[bool, str]:
        if self.redis_client is None:
            return False, "unconfigured"
        try:
            pong = self.redis_client.ping()
            return (True, "connected") if pong else (False, "no_pong")
        except Exception as exc:
            return False, f"disconnected:{type(exc).__name__}"

    def health(self) -> HealthResponse:
        redis_ok, redis_status = self.check_redis_health()
        loaded = sorted(self.models.keys())
        status = "healthy" if loaded else "degraded"
        return HealthResponse(
            status=status,
            model_version=self.model_version,
            git_sha=self.git_sha,
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            loaded_targets=loaded,
            redis_connected=redis_ok,
            redis_status=redis_status,
            cpp_eigen_available=bool(_HAS_CPP_KELLY),
        )

    def fetch_player_history_from_redis(self, player_id: str) -> pd.DataFrame:
        if self.redis_client is None:
            return pd.DataFrame()
        try:
            raw = self.redis_client.get(f"player:history:{player_id}")
            if not raw:
                return pd.DataFrame()
            records = json.loads(raw) if isinstance(raw, str) else raw
            if not records:
                return pd.DataFrame()
            return pd.DataFrame(records)
        except Exception as exc:
            logger.warning(f"Redis hydration failed for player_id={player_id}: {exc}")
            return pd.DataFrame()

    @staticmethod
    def _estimate_zero_inflation(history_df: Optional[pd.DataFrame], stat_col: str, mu: float, n_nb: float) -> float:
        if history_df is None or history_df.empty or stat_col not in history_df.columns:
            return 0.0
        vals = pd.to_numeric(history_df[stat_col], errors="coerce").dropna().to_numpy(dtype=float)
        if len(vals) == 0:
            return 0.0
        pi_raw = float(np.mean(vals == 0.0))
        p_nb = n_nb / (n_nb + max(mu, 1e-6))
        p_zero_nb = p_nb ** n_nb
        pi = max(0.0, (pi_raw - p_zero_nb) / max(1e-9, 1.0 - p_zero_nb))
        return float(min(pi, 0.95))

    @staticmethod
    def _compute_survival(mu: float, n_nb: float, pi: float, thresholds: List[float]) -> List[ThresholdSurvival]:
        if not thresholds:
            return []
        p_nb = n_nb / (n_nb + max(mu, 1e-9))
        out: List[ThresholdSurvival] = []
        for threshold in thresholds:
            cutoff = int(math.floor(threshold))
            nb_cdf = float(stats.nbinom.cdf(cutoff, n_nb, p_nb))
            cdf = pi + (1.0 - pi) * nb_cdf
            prob_over = float(np.clip(1.0 - cdf, 0.0, 1.0))
            if abs(threshold - cutoff) < 1e-9:
                nb_pmf = float(stats.nbinom.pmf(cutoff, n_nb, p_nb))
                pmf_k = (pi + (1.0 - pi) * nb_pmf) if cutoff == 0 else ((1.0 - pi) * nb_pmf)
                prob_under = float(np.clip(cdf - pmf_k, 0.0, 1.0))
            else:
                prob_under = float(np.clip(cdf, 0.0, 1.0))
            out.append(
                ThresholdSurvival(
                    threshold=float(threshold),
                    prob_over=prob_over,
                    prob_under=prob_under,
                )
            )
        return out

    def _prepare_item(
        self, idx: int, item: PredictionItemRequest
    ) -> Tuple[int, Optional[pd.Series], float, str, bool, float, Optional[float], Optional[pd.DataFrame]]:
        """Resolves a single request item into either an engineered feature row or a cold-start fallback."""
        league_key = item.league.value
        target_key = item.stat.value
        stat_col = TARGET_METRICS[target_key]
        priors = self.cached_priors.get(league_key, self.cached_priors["NBA"])
        known_set = self.known_players.get(target_key, set())

        if item.features is not None:
            feat_dict = item.features.model_dump()
            pid = str(feat_dict["player_id"])
            ewma_baseline = (
                float(feat_dict["bayesian_roll_stat"])
                if feat_dict.get("bayesian_roll_stat") is not None
                else float(feat_dict["roll_stat"])
            )
            alpha_override = feat_dict.get("alpha_disp_override")
            pi_val = float(feat_dict["zero_inflation_prob"]) if feat_dict.get("zero_inflation_prob") is not None else 0.0
            cold_start = bool(known_set and pid not in known_set)
            active_cols = get_active_feature_columns(league=league_key, stat_col=stat_col)
            row_dict = {col: feat_dict.get(col, 0.0) for col in active_cols}
            row_dict["roll_min"] = feat_dict["roll_min"]
            return idx, pd.Series(row_dict), ewma_baseline, pid, cold_start, pi_val, alpha_override, None

        pid = str(item.player_id).strip()
        ctx = item.game_context
        ctx_dict = ctx.model_dump(exclude_none=True) if ctx is not None else {}
        position = ctx_dict.get("position", "SG")

        if item.history_logs is not None:
            hist_df = pd.DataFrame([entry.model_dump() for entry in item.history_logs])
        else:
            hist_df = self.fetch_player_history_from_redis(pid)

        if not hist_df.empty:
            feat_series, active_cols = build_online_feature_row(
                player_id=pid,
                history_df=hist_df,
                target_context=ctx_dict,
                stat_col=stat_col,
                position=position,
                cached_priors=priors,
                league=league_key,
            )
            has_nan_features = any(pd.isna(feat_series[c]) for c in active_cols) or pd.isna(
                feat_series.get("bayesian_roll_stat")
            )
            if not has_nan_features:
                ewma_baseline = (
                    float(feat_series["roll_stat"])
                    if league_key == "WNBA"
                    else float(feat_series["bayesian_roll_stat"])
                )
                cold_start = bool(known_set and pid not in known_set)
                return idx, feat_series, ewma_baseline, pid, cold_start, 0.0, None, hist_df

        # Cold-start unseen player / insufficient current-season history fallback
        prior_per_36 = get_positional_prior(stat_col, position, cached_priors=priors)
        fallback_mins = 20.0 if league_key == "WNBA" else 24.0
        ewma_baseline = max(0.001, (prior_per_36 / 36.0) * fallback_mins)
        return idx, None, ewma_baseline, pid, True, 0.0, None, hist_df

    def predict_batch(self, req: BatchPredictRequest) -> BatchPredictResponse:
        t0 = time.perf_counter_ns()
        prepared = [self._prepare_item(i, item) for i, item in enumerate(req.requests)]

        # Group non-fallback items by (league, stat_key) for vectorized batch inference
        grouped: Dict[Tuple[str, str], List[tuple]] = defaultdict(list)
        responses: List[Optional[PredictionItemResponse]] = [None] * len(req.requests)

        for entry in prepared:
            idx, feat_series, ewma_baseline, pid, cold_start, pi_override, alpha_override, hist_df = entry
            item = req.requests[idx]
            target_key = item.stat.value
            if feat_series is not None and target_key in self.models:
                grouped[(item.league.value, target_key)].append(entry)
            else:
                meta_data = self.models.get(target_key, {}).get("meta", {})
                alpha_disp = float(alpha_override or meta_data.get("alpha", 0.25))
                n_nb = 1.0 / max(alpha_disp, 1e-6)
                pi_val = (
                    pi_override
                    if item.features is not None and item.features.zero_inflation_prob is not None
                    else self._estimate_zero_inflation(hist_df, TARGET_METRICS[target_key], ewma_baseline, n_nb)
                )
                mu_val = max(0.001, float(ewma_baseline))
                responses[idx] = PredictionItemResponse(
                    player_id=pid,
                    stat=item.stat,
                    league=item.league,
                    schema_version=FEATURE_SCHEMA_VERSION,
                    predicted_mean=mu_val,
                    residual=0.0,
                    xgb_residual=0.0,
                    lgb_residual=0.0,
                    ewma_baseline=mu_val,
                    zinb=ZINBParams(
                        mu=mu_val,
                        alpha_disp=alpha_disp,
                        zero_inflation_prob=pi_val,
                        n_nb=n_nb,
                    ),
                    survival_probabilities=self._compute_survival(mu_val, n_nb, pi_val, item.thresholds),
                    cold_start=cold_start,
                )

        for (league_key, target_key), entries in grouped.items():
            stat_col = TARGET_METRICS[target_key]
            active_cols = get_active_feature_columns(league=league_key, stat_col=stat_col)
            bundle = self.models[target_key]
            xgb_model = bundle.get("xgb")
            lgb_model = bundle.get("lgb")
            meta_data = bundle.get("meta", {})
            xgb_cats = bundle.get("xgb_categories")

            batch_df = pd.DataFrame([e[1] for e in entries]).reset_index(drop=True)
            ewma_arr = np.array([e[2] for e in entries], dtype=float)
            roll_min_arr = batch_df["roll_min"].to_numpy(dtype=float)

            preds = predict_meta_ensemble(
                batch_df,
                active_cols,
                xgb_model=xgb_model,
                lgb_model=lgb_model,
                meta_data=meta_data,
                ewma_baseline=ewma_arr,
                league=league_key,
                stat_col=stat_col,
                roll_min=roll_min_arr,
                xgb_categories=xgb_cats,
            )

            default_alpha = float(meta_data.get("alpha", 0.25))
            for local_i, entry in enumerate(entries):
                idx, _, ewma_baseline, pid, cold_start, pi_override, alpha_override, hist_df = entry
                item = req.requests[idx]
                mu_val = float(preds["meta_prediction"][local_i])
                alpha_disp = float(alpha_override or default_alpha)
                n_nb = 1.0 / max(alpha_disp, 1e-6)
                pi_val = (
                    pi_override
                    if item.features is not None and item.features.zero_inflation_prob is not None
                    else self._estimate_zero_inflation(hist_df, stat_col, mu_val, n_nb)
                )
                responses[idx] = PredictionItemResponse(
                    player_id=pid,
                    stat=item.stat,
                    league=item.league,
                    schema_version=FEATURE_SCHEMA_VERSION,
                    predicted_mean=mu_val,
                    residual=float(preds["meta_residual"][local_i]),
                    xgb_residual=float(preds["xgb_residual"][local_i]),
                    lgb_residual=float(preds["lgb_residual"][local_i]),
                    ewma_baseline=float(ewma_baseline),
                    zinb=ZINBParams(
                        mu=mu_val,
                        alpha_disp=alpha_disp,
                        zero_inflation_prob=pi_val,
                        n_nb=n_nb,
                    ),
                    survival_probabilities=self._compute_survival(mu_val, n_nb, pi_val, item.thresholds),
                    cold_start=cold_start,
                )

        latency_ms = (time.perf_counter_ns() - t0) / 1e6
        return BatchPredictResponse(
            schema_version=FEATURE_SCHEMA_VERSION,
            model_version=self.model_version,
            predictions=[r for r in responses if r is not None],
            latency_ms=latency_ms,
        )

    @staticmethod
    def _clean_stat_name(stat_str: str) -> str:
        s = stat_str.lower().strip()
        if "point" in s or s == "pts":
            return "points"
        if "rebound" in s or s == "reb":
            return "rebounds"
        if "assist" in s or s == "ast":
            return "assists"
        return s

    @classmethod
    def _determine_relationship(cls, c1: AllocateCandidateOutcome, c2: AllocateCandidateOutcome) -> str:
        if c1.game_id is not None and c2.game_id is not None and c1.game_id != c2.game_id:
            return "independent"
        if c1.team_id is not None and c2.team_id is not None:
            return "teammate" if c1.team_id == c2.team_id else "opponent"
        return "independent"

    @classmethod
    def _copula_joint_probability(
        cls,
        u: float,
        v: float,
        adjusted_rho: float,
        side1: str,
        side2: str,
        copula_type: CopulaTypeEnum,
    ) -> Tuple[float, str]:
        u_c = float(np.clip(u, 1e-6, 1.0 - 1e-6))
        v_c = float(np.clip(v, 1e-6, 1.0 - 1e-6))
        if abs(adjusted_rho) < 1e-9:
            return u_c * v_c, "independent"

        chosen = copula_type
        if chosen == CopulaTypeEnum.auto:
            if side1 == "over" and side2 == "over" and adjusted_rho > 0:
                chosen = CopulaTypeEnum.gumbel
            elif side1 == "under" and side2 == "under" and adjusted_rho > 0:
                chosen = CopulaTypeEnum.clayton
            else:
                chosen = CopulaTypeEnum.frank

        if chosen == CopulaTypeEnum.gumbel and adjusted_rho > 0:
            tau = float(np.clip(adjusted_rho, 1e-5, 0.99))
            theta = max(1.0, 1.0 / (1.0 - tau))
            u_g = -math.log(u_c)
            v_g = -math.log(v_c)
            jp = math.exp(-((u_g ** theta + v_g ** theta) ** (1.0 / theta)))
            return float(np.clip(jp, 0.0, min(u_c, v_c))), "gumbel"

        if chosen == CopulaTypeEnum.clayton and adjusted_rho > 0:
            tau = float(np.clip(adjusted_rho, 1e-5, 0.99))
            theta = max(1e-5, (2.0 * tau) / (1.0 - tau))
            jp = max(u_c ** (-theta) + v_c ** (-theta) - 1.0, 0.0) ** (-1.0 / theta)
            return float(np.clip(jp, 0.0, min(u_c, v_c))), "clayton"

        if chosen == CopulaTypeEnum.frank:
            theta = float(np.clip(adjusted_rho * 9.0, -25.0, 25.0))
            jp = MarketMicrostructure.frank_copula_joint_prob(u_c, v_c, theta)
            return float(np.clip(jp, 0.0, 1.0)), "frank"

        # Gaussian copula fallback for mixed/negative dependencies
        z1 = float(stats.norm.ppf(u_c))
        z2 = float(stats.norm.ppf(v_c))
        rho_clip = float(np.clip(math.sin(math.pi / 2.0 * adjusted_rho), -0.99, 0.99))
        jp = float(stats.multivariate_normal.cdf([z1, z2], mean=[0.0, 0.0], cov=[[1.0, rho_clip], [rho_clip, 1.0]]))
        return float(np.clip(jp, 0.0, 1.0)), "gaussian"

    def allocate(self, req: AllocateRequest) -> AllocateResponse:
        t0 = time.perf_counter_ns()
        n = len(req.candidates)
        edges = np.array(
            [
                float(c.expected_edge)
                if c.expected_edge is not None
                else (float(c.marginal_prob) * float(c.payout_odds) - 1.0)
                for c in req.candidates
            ],
            dtype=float,
        )

        joint_pairs: List[JointPairPricing] = []
        if req.covariance_matrix is not None:
            cov_raw = np.array(req.covariance_matrix, dtype=float)
            cov_raw = 0.5 * (cov_raw + cov_raw.T)
        else:
            probs = np.array([float(c.marginal_prob) for c in req.candidates], dtype=float)
            cov_raw = np.zeros((n, n), dtype=float)
            for i in range(n):
                cov_raw[i, i] = probs[i] * (1.0 - probs[i])
            for i in range(n):
                for j in range(i + 1, n):
                    c1 = req.candidates[i]
                    c2 = req.candidates[j]
                    rel = self._determine_relationship(c1, c2)
                    if rel == "independent":
                        rho = 0.0
                    else:
                        s1 = self._clean_stat_name(c1.stat)
                        s2 = self._clean_stat_name(c2.stat)
                        rel_map = DEFAULT_CORRELATIONS.get(rel, {})
                        base_rho = float(rel_map.get((s1, s2)) or rel_map.get((s2, s1)) or 0.0)
                        side1 = c1.side.value.lower()
                        side2 = c2.side.value.lower()
                        rho = base_rho if side1 == side2 else -base_rho
                    jp, cop_name = self._copula_joint_probability(
                        probs[i],
                        probs[j],
                        rho,
                        c1.side.value.lower(),
                        c2.side.value.lower(),
                        req.copula_type,
                    )
                    cov_ij = jp - (probs[i] * probs[j])
                    cov_raw[i, j] = cov_ij
                    cov_raw[j, i] = cov_ij
                    if rel != "independent" or abs(rho) > 0.0:
                        joint_pairs.append(
                            JointPairPricing(
                                leg_i=i,
                                leg_j=j,
                                relationship=rel,
                                copula_used=cop_name,
                                rho=float(rho),
                                joint_prob=float(jp),
                                independent_prob=float(probs[i] * probs[j]),
                            )
                        )

        # Numerical stability telemetry on raw covariance matrix
        sv_raw = np.linalg.svd(cov_raw, compute_uv=False)
        tol_raw = max(cov_raw.shape) * np.finfo(float).eps * (sv_raw[0] if sv_raw[0] > 0 else 1.0)
        numerical_rank = int(np.sum(sv_raw > tol_raw))
        is_rank_deficient = bool(numerical_rank < n)
        cond_raw = float(sv_raw[0] / sv_raw[-1]) if sv_raw[-1] > 1e-15 else 1e16

        # Ledoit-Wolf linear shrinkage
        delta = float(req.shrinkage_delta)
        cov_shrunk = (1.0 - delta) * cov_raw + delta * (np.trace(cov_raw) / n) * np.eye(n)
        cov_shrunk = 0.5 * (cov_shrunk + cov_shrunk.T)
        sv_shrunk = np.linalg.svd(cov_shrunk, compute_uv=False)
        cond_shrunk = float(sv_shrunk[0] / sv_shrunk[-1]) if sv_shrunk[-1] > 1e-15 else 1e16

        # Unconstrained QR/lstsq reference residual norm ||Sigma_shrunk * w_unc - mu||
        w_unc, _, _, _ = np.linalg.lstsq(cov_shrunk, edges, rcond=None)
        solve_residual_norm = float(np.linalg.norm(cov_shrunk @ w_unc - edges))

        # C++ Eigen ColPivHouseholderQR solver (with clean NumPy fallback)
        if _HAS_CPP_KELLY and _quant_core is not None:
            alloc_vec = np.array(
                _quant_core.optimize_multivariate_kelly(
                    edges.tolist(),
                    cov_shrunk.tolist(),
                    float(req.max_bankroll_fraction),
                    float(req.kelly_fraction),
                ),
                dtype=float,
            )
            solver_backend = "cpp_eigen_colpiv_qr"
        else:
            f = np.linalg.lstsq(cov_shrunk, edges, rcond=None)[0]
            f = np.clip(f, 0.0, None)
            total = float(np.sum(f))
            if total > req.max_bankroll_fraction and total > 1e-12:
                f = f * (req.max_bankroll_fraction / total)
            alloc_vec = f * float(req.kelly_fraction)
            solver_backend = "numpy_lstsq_fallback"

        alloc_vec = np.nan_to_num(alloc_vec, nan=0.0, posinf=req.max_bankroll_fraction, neginf=0.0)
        residual_norm = float(np.linalg.norm(cov_shrunk @ alloc_vec - edges))

        dollar_stakes = [float(a * req.bankroll_usd) for a in alloc_vec]
        masked_stakes = [
            RiskManager.mask_bet_size(s, league=req.league.value) if s > 0.0 else 0 for s in dollar_stakes
        ]
        latency_ms = (time.perf_counter_ns() - t0) / 1e6

        return AllocateResponse(
            schema_version=FEATURE_SCHEMA_VERSION,
            league=req.league,
            allocations=[float(x) for x in alloc_vec],
            dollar_stakes=dollar_stakes,
            masked_stakes=masked_stakes,
            expected_edges=[float(x) for x in edges],
            joint_pairs=joint_pairs,
            condition_number_raw=cond_raw,
            condition_number_shrunk=cond_shrunk,
            residual_norm=residual_norm,
            solve_residual_norm=solve_residual_norm,
            solver_backend=solver_backend,
            is_rank_deficient=is_rank_deficient,
            numerical_rank=numerical_rank,
            latency_ms=latency_ms,
        )

