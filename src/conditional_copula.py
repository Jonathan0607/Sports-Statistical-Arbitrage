"""Context-dependent Conditional Copula dependence model theta(x) with analytical output layer.

Maps pairwise pre-game game-script context features to bounded Kendall's tau(x) in [-0.75, 0.75]
and evaluates exact analytical copulas (Gumbel for double-Over, Clayton for double-Under, Frank/Gaussian
for negative or mixed concordance), guaranteeing strict copula axioms and Frechet-Hoeffding bounds.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import optimize, stats

TAU_MAX = 0.75

PAIR_FEATURE_NAMES: List[str] = [
    "is_same_team",
    "is_same_position",
    "same_team_same_pos",
    "same_team_opp_pos",
    "projected_pace_diff",
    "mean_projected_pace_centered",
    "abs_team_strength_diff",
    "combined_vacated_usage",
    "min_player_ewma_minutes_scaled",
    "pair_pts_pts",
    "pair_pts_reb",
    "pair_pts_ast",
    "pair_reb_reb",
    "pair_reb_ast",
    "pair_ast_ast",
    "dir_OO",
    "dir_UU",
    "dir_OU",
    "teammate_same_dir",
    "teammate_opp_dir",
    "opp_same_dir",
    "blowout_x_uu",
    "blowout_x_oo",
    "teammate_pts_ast_dir",
    "teammate_pts_pts_same_pos_dir",
    "teammate_reb_reb_dir",
]

# Validated Chebyshev/polynomial approximation for Frank copula Kendall's tau -> theta on [-0.75, 0.75]
_FRANK_POLY_COEFFS = (
    9.005388624601167,
    6.8113409824909805,
    14.150496170981532,
    -20.326295101165716,
    51.44594856636756,
)


def clean_stat_key(stat_name: str) -> str:
    s = str(stat_name or "").strip().lower()
    if "point" in s or s == "pts":
        return "pts"
    if "rebound" in s or s == "reb":
        return "reb"
    if "assist" in s or s == "ast":
        return "ast"
    return s


def canonical_stat_pair(stat1: str, stat2: str) -> str:
    s1, s2 = clean_stat_key(stat1), clean_stat_key(stat2)
    order = {"pts": 0, "reb": 1, "ast": 2}
    if order.get(s1, 99) <= order.get(s2, 99):
        return f"{s1}-{s2}"
    return f"{s2}-{s1}"


def canonical_direction_pair(side1: str, side2: str) -> str:
    d1 = "O" if str(side1).strip().lower() == "over" else "U"
    d2 = "O" if str(side2).strip().lower() == "over" else "U"
    if d1 == "O" and d2 == "O":
        return "OO"
    if d1 == "U" and d2 == "U":
        return "UU"
    return "OU"


def tau_to_frank_theta_scalar(tau: float) -> float:
    t = float(np.clip(tau, -TAU_MAX, TAU_MAX))
    if abs(t) < 1e-9:
        return 0.0
    t2 = t * t
    c0, c1, c2, c3, c4 = _FRANK_POLY_COEFFS
    scale = c0 + t2 * (c1 + t2 * (c2 + t2 * (c3 + t2 * c4)))
    return float(np.clip(t * scale, -25.0, 25.0))


def tau_to_frank_theta_vec(tau: np.ndarray) -> np.ndarray:
    t = np.clip(np.asarray(tau, dtype=float), -TAU_MAX, TAU_MAX)
    t2 = t * t
    c0, c1, c2, c3, c4 = _FRANK_POLY_COEFFS
    scale = c0 + t2 * (c1 + t2 * (c2 + t2 * (c3 + t2 * c4)))
    return np.clip(t * scale, -25.0, 25.0)


def frechet_hoeffding_clamp(jp: float, u: float, v: float) -> float:
    fh_low = max(0.0, float(u) + float(v) - 1.0)
    fh_high = min(float(u), float(v))
    return float(min(fh_high, max(fh_low, float(jp))))


def copula_joint_probability(
    u: float,
    v: float,
    tau_hat: float,
    side1: str,
    side2: str,
    negative_copula: str = "frank",
    force_family: Optional[str] = None,
) -> Tuple[float, str]:
    """Evaluates analytical copula C(u, v; tau_hat) satisfying strict Frechet-Hoeffding bounds."""
    u_c = float(np.clip(u, 1e-6, 1.0 - 1e-6))
    v_c = float(np.clip(v, 1e-6, 1.0 - 1e-6))
    tau = float(np.clip(tau_hat, -TAU_MAX, TAU_MAX))
    if abs(tau) < 1e-12:
        return frechet_hoeffding_clamp(u_c * v_c, u_c, v_c), "independent"

    d_pair = canonical_direction_pair(side1, side2)
    family = force_family
    if family is None or family == "auto":
        if tau > 0.0:
            if d_pair == "OO":
                family = "gumbel"
            elif d_pair == "UU":
                family = "clayton"
            else:
                family = "frank"
        else:
            family = "gaussian" if negative_copula == "gaussian" else "frank"

    if family == "gumbel" and tau > 0.0:
        theta = max(1.0, 1.0 / max(1e-6, 1.0 - tau))
        lu = -math.log(u_c)
        lv = -math.log(v_c)
        jp = math.exp(-((lu ** theta + lv ** theta) ** (1.0 / theta)))
        return frechet_hoeffding_clamp(jp, u_c, v_c), "gumbel"

    if family == "clayton" and tau > 0.0:
        theta = max(1e-5, (2.0 * tau) / max(1e-6, 1.0 - tau))
        inner = max(u_c ** (-theta) + v_c ** (-theta) - 1.0, 1e-15)
        jp = inner ** (-1.0 / theta)
        return frechet_hoeffding_clamp(jp, u_c, v_c), "clayton"

    if family == "gaussian":
        rho = float(np.clip(math.sin((math.pi / 2.0) * tau), -0.99, 0.99))
        z1 = float(stats.norm.ppf(u_c))
        z2 = float(stats.norm.ppf(v_c))
        jp = float(
            stats.multivariate_normal.cdf(
                [z1, z2], mean=[0.0, 0.0], cov=[[1.0, rho], [rho, 1.0]]
            )
        )
        return frechet_hoeffding_clamp(jp, u_c, v_c), "gaussian"

    # Analytical Frank copula handles both positive (OU) and negative concordance fast
    theta = tau_to_frank_theta_scalar(tau)
    if abs(theta) < 1e-11:
        return frechet_hoeffding_clamp(u_c * v_c, u_c, v_c), "independent"
    num = math.expm1(-theta * u_c) * math.expm1(-theta * v_c)
    den = math.expm1(-theta)
    ratio = num / den if abs(den) > 1e-15 else 0.0
    if ratio <= -1.0 + 1e-15:
        jp = 0.0
    else:
        jp = (-1.0 / theta) * math.log1p(ratio)
    return frechet_hoeffding_clamp(jp, u_c, v_c), "frank"


def vectorized_copula_joint_prob(
    u: np.ndarray,
    v: np.ndarray,
    tau_hat: np.ndarray,
    is_oo: np.ndarray,
    is_uu: np.ndarray,
) -> np.ndarray:
    """Vectorized joint C(u, v; tau_hat) routing OO+ -> Gumbel, UU+ -> Clayton, others -> Frank."""
    u_c = np.clip(np.asarray(u, dtype=float), 1e-6, 1.0 - 1e-6)
    v_c = np.clip(np.asarray(v, dtype=float), 1e-6, 1.0 - 1e-6)
    tau = np.clip(np.asarray(tau_hat, dtype=float), -TAU_MAX, TAU_MAX)

    jp = u_c * v_c

    gumbel_mask = (tau > 1e-12) & is_oo
    if np.any(gumbel_mask):
        t_g = tau[gumbel_mask]
        th_g = 1.0 / np.maximum(1e-6, 1.0 - t_g)
        lu = -np.log(u_c[gumbel_mask])
        lv = -np.log(v_c[gumbel_mask])
        jp[gumbel_mask] = np.exp(-((np.power(lu, th_g) + np.power(lv, th_g)) ** (1.0 / th_g)))

    clayton_mask = (tau > 1e-12) & is_uu
    if np.any(clayton_mask):
        t_c = tau[clayton_mask]
        th_c = (2.0 * t_c) / np.maximum(1e-6, 1.0 - t_c)
        inner_c = np.maximum(
            np.power(u_c[clayton_mask], -th_c) + np.power(v_c[clayton_mask], -th_c) - 1.0,
            1e-15,
        )
        jp[clayton_mask] = np.power(inner_c, -1.0 / th_c)

    frank_mask = (np.abs(tau) > 1e-12) & (~gumbel_mask) & (~clayton_mask)
    if np.any(frank_mask):
        th_f = tau_to_frank_theta_vec(tau[frank_mask])
        uf = u_c[frank_mask]
        vf = v_c[frank_mask]
        num = np.expm1(-th_f * uf) * np.expm1(-th_f * vf)
        den = np.expm1(-th_f)
        safe_den = np.where(np.abs(den) < 1e-15, -1e-15, den)
        ratio = np.maximum(-1.0 + 1e-15, num / safe_den)
        safe_th = np.where(np.abs(th_f) < 1e-15, 1e-15, th_f)
        jp[frank_mask] = (-1.0 / safe_th) * np.log1p(ratio)

    fh_low = np.maximum(0.0, u_c + v_c - 1.0)
    fh_high = np.minimum(u_c, v_c)
    return np.clip(jp, fh_low, fh_high)


def ensure_psd_covariance(cov: np.ndarray, min_eig: float = 1e-9) -> np.ndarray:
    """Projects a symmetric matrix onto the PSD cone via spectral eigenvalue clipping."""
    mat = 0.5 * (np.asarray(cov, dtype=float) + np.asarray(cov, dtype=float).T)
    eigvals, eigvecs = np.linalg.eigh(mat)
    if float(np.min(eigvals)) >= min_eig:
        return mat
    clipped = np.maximum(eigvals, min_eig)
    repaired = (eigvecs * clipped) @ eigvecs.T
    return 0.5 * (repaired + repaired.T)


def build_pair_feature_row(leg1: Dict[str, Any], leg2: Dict[str, Any]) -> np.ndarray:
    """Extracts swap-invariant pairwise pre-game context features in exact PAIR_FEATURE_NAMES order."""
    t1 = leg1.get("team_id")
    t2 = leg2.get("team_id")
    same_team = 1.0 if (t1 is not None and t2 is not None and int(t1) == int(t2)) else 0.0

    p1 = str(leg1.get("position") or "").strip().upper()
    p2 = str(leg2.get("position") or "").strip().upper()
    same_pos = 1.0 if (p1 and p2 and p1 == p2) else 0.0
    same_team_same_pos = same_team * same_pos
    same_team_opp_pos = same_team * (1.0 - same_pos)

    pace1 = float(leg1.get("opp_pace_factor") if leg1.get("opp_pace_factor") is not None else 99.5)
    pace2 = float(leg2.get("opp_pace_factor") if leg2.get("opp_pace_factor") is not None else 99.5)
    pace_diff_raw = leg1.get("projected_pace_diff")
    if pace_diff_raw is not None:
        proj_pace_diff = abs(float(pace_diff_raw))
    else:
        proj_pace_diff = abs(pace1 - pace2)
    mean_pace_c = 0.5 * (pace1 + pace2) - 99.5

    sd_raw = leg1.get("abs_team_strength_diff")
    if sd_raw is None:
        sd_raw = leg2.get("abs_team_strength_diff")
    if sd_raw is not None:
        abs_strength_diff = abs(float(sd_raw))
    else:
        s1_net = float(leg1.get("team_strength") or 0.0)
        s2_net = float(leg2.get("team_strength") or 0.0)
        abs_strength_diff = abs(s1_net - s2_net)

    vac1 = float(leg1.get("vacated_usage") or 0.0)
    vac2 = float(leg2.get("vacated_usage") or 0.0)
    combined_vac = max(0.0, vac1 + vac2)

    m1 = float(leg1.get("ewma_minutes") if leg1.get("ewma_minutes") is not None else 24.0)
    m2 = float(leg2.get("ewma_minutes") if leg2.get("ewma_minutes") is not None else 24.0)
    min_min_scaled = (min(m1, m2) - 24.0) / 10.0

    sp = canonical_stat_pair(leg1.get("stat", "pts"), leg2.get("stat", "pts"))
    p_pts_pts = 1.0 if sp == "pts-pts" else 0.0
    p_pts_reb = 1.0 if sp == "pts-reb" else 0.0
    p_pts_ast = 1.0 if sp == "pts-ast" else 0.0
    p_reb_reb = 1.0 if sp == "reb-reb" else 0.0
    p_reb_ast = 1.0 if sp == "reb-ast" else 0.0
    p_ast_ast = 1.0 if sp == "ast-ast" else 0.0

    dp = canonical_direction_pair(leg1.get("side", "Over"), leg2.get("side", "Over"))
    dir_oo = 1.0 if dp == "OO" else 0.0
    dir_uu = 1.0 if dp == "UU" else 0.0
    dir_ou = 1.0 if dp == "OU" else 0.0
    dir_cong = 1.0 if dp in ("OO", "UU") else -1.0

    teammate_same_dir = same_team * (1.0 if dir_cong > 0 else 0.0)
    teammate_opp_dir = same_team * (1.0 if dir_cong < 0 else 0.0)
    opp_same_dir = (1.0 - same_team) * dir_cong
    blowout_x_uu = (abs_strength_diff / 10.0) * dir_uu
    blowout_x_oo = (abs_strength_diff / 10.0) * dir_oo
    tm_pts_ast_dir = same_team * p_pts_ast * dir_cong
    tm_pts_pts_same_pos_dir = same_team_same_pos * p_pts_pts * dir_cong
    tm_reb_reb_dir = same_team * p_reb_reb * dir_cong

    return np.array(
        [
            same_team,
            same_pos,
            same_team_same_pos,
            same_team_opp_pos,
            proj_pace_diff,
            mean_pace_c,
            abs_strength_diff,
            combined_vac,
            min_min_scaled,
            p_pts_pts,
            p_pts_reb,
            p_pts_ast,
            p_reb_reb,
            p_reb_ast,
            p_ast_ast,
            dir_oo,
            dir_uu,
            dir_ou,
            teammate_same_dir,
            teammate_opp_dir,
            opp_same_dir,
            blowout_x_uu,
            blowout_x_oo,
            tm_pts_ast_dir,
            tm_pts_pts_same_pos_dir,
            tm_reb_reb_dir,
        ],
        dtype=float,
    )


def build_pair_feature_matrix(pairs_df: pd.DataFrame) -> np.ndarray:
    """Vectorized construction of PAIR_FEATURE_NAMES matrix from a DataFrame of pairwise contexts."""
    same_team = pairs_df["is_same_team"].to_numpy(dtype=float)
    same_pos = pairs_df["is_same_position"].to_numpy(dtype=float)
    same_team_same_pos = same_team * same_pos
    same_team_opp_pos = same_team * (1.0 - same_pos)
    proj_pace_diff = np.abs(pairs_df["projected_pace_diff"].to_numpy(dtype=float))
    mean_pace_c = pairs_df["mean_projected_pace"].to_numpy(dtype=float) - 99.5
    abs_strength = np.abs(pairs_df["abs_team_strength_diff"].to_numpy(dtype=float))
    comb_vac = np.maximum(0.0, pairs_df["combined_vacated_usage"].to_numpy(dtype=float))
    min_min_scaled = (pairs_df["min_player_ewma_minutes"].to_numpy(dtype=float) - 24.0) / 10.0

    sp = pairs_df["stat_pair_type"].astype(str).to_numpy()
    p_pts_pts = (sp == "pts-pts").astype(float)
    p_pts_reb = (sp == "pts-reb").astype(float)
    p_pts_ast = (sp == "pts-ast").astype(float)
    p_reb_reb = (sp == "reb-reb").astype(float)
    p_reb_ast = (sp == "reb-ast").astype(float)
    p_ast_ast = (sp == "ast-ast").astype(float)

    dp = pairs_df["direction_pair"].astype(str).to_numpy()
    dir_oo = (dp == "OO").astype(float)
    dir_uu = (dp == "UU").astype(float)
    dir_ou = (dp == "OU").astype(float)
    dir_cong = np.where(dp == "OU", -1.0, 1.0)

    return np.column_stack(
        [
            same_team,
            same_pos,
            same_team_same_pos,
            same_team_opp_pos,
            proj_pace_diff,
            mean_pace_c,
            abs_strength,
            comb_vac,
            min_min_scaled,
            p_pts_pts,
            p_pts_reb,
            p_pts_ast,
            p_reb_reb,
            p_reb_ast,
            p_ast_ast,
            dir_oo,
            dir_uu,
            dir_ou,
            same_team * (dir_cong > 0).astype(float),
            same_team * (dir_cong < 0).astype(float),
            (1.0 - same_team) * dir_cong,
            (abs_strength / 10.0) * dir_uu,
            (abs_strength / 10.0) * dir_oo,
            same_team * p_pts_ast * dir_cong,
            same_team_same_pos * p_pts_pts * dir_cong,
            same_team * p_reb_reb * dir_cong,
        ]
    )


class ConditionalCopulaModel:
    """L2-regularized context-dependent Kendall's tau(x) copula model trained on Joint Binary Log-Loss."""

    def __init__(
        self,
        coef: Optional[np.ndarray] = None,
        intercept: float = 0.0,
        scaler_mean: Optional[np.ndarray] = None,
        scaler_scale: Optional[np.ndarray] = None,
        reg_lambda: float = 1e-3,
        feature_names: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.feature_names = list(feature_names or PAIR_FEATURE_NAMES)
        d = len(self.feature_names)
        self.coef = np.zeros(d, dtype=float) if coef is None else np.asarray(coef, dtype=float)
        self.intercept = float(intercept)
        self.scaler_mean = (
            np.zeros(d, dtype=float) if scaler_mean is None else np.asarray(scaler_mean, dtype=float)
        )
        self.scaler_scale = (
            np.ones(d, dtype=float) if scaler_scale is None else np.asarray(scaler_scale, dtype=float)
        )
        self.reg_lambda = float(reg_lambda)
        self.metadata = metadata or {}

    def predict_tau(self, X: np.ndarray) -> np.ndarray:
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        z = (arr - self.scaler_mean) / np.where(self.scaler_scale > 1e-8, self.scaler_scale, 1.0)
        raw = z @ self.coef + self.intercept
        return TAU_MAX * np.tanh(raw)

    def predict_tau_single(self, leg1: Dict[str, Any], leg2: Dict[str, Any]) -> float:
        feat = build_pair_feature_row(leg1, leg2)
        z = (feat - self.scaler_mean) / np.where(self.scaler_scale > 1e-8, self.scaler_scale, 1.0)
        raw = float(np.dot(z, self.coef) + self.intercept)
        return float(TAU_MAX * math.tanh(raw))

    def predict_joint_probs(
        self,
        X: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        is_oo: np.ndarray,
        is_uu: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        tau_hat = self.predict_tau(X)
        jp = vectorized_copula_joint_prob(u, v, tau_hat, is_oo, is_uu)
        return jp, tau_hat

    @staticmethod
    def _joint_logloss(y_joint: np.ndarray, p_joint: np.ndarray, eps: float = 1e-12) -> float:
        pc = np.clip(p_joint, eps, 1.0 - eps)
        return float(-np.mean(y_joint * np.log(pc) + (1.0 - y_joint) * np.log(1.0 - pc)))

    @classmethod
    def fit(
        cls,
        X_train: np.ndarray,
        u_train: np.ndarray,
        v_train: np.ndarray,
        is_oo: np.ndarray,
        is_uu: np.ndarray,
        y_joint: np.ndarray,
        *,
        game_timestamps: Optional[np.ndarray] = None,
        reg_grid: Sequence[float] = (1e-6, 1e-5, 1e-4, 5e-4, 1e-3, 5e-3),
        max_iter: int = 120,
        seed: int = 42,
    ) -> "ConditionalCopulaModel":
        """Fits scaler + L2-regularized tau(x) weights on Joint Binary Log-Loss using chronological CV."""
        X = np.asarray(X_train, dtype=float)
        u = np.clip(np.asarray(u_train, dtype=float), 1e-6, 1.0 - 1e-6)
        v = np.clip(np.asarray(v_train, dtype=float), 1e-6, 1.0 - 1e-6)
        oo = np.asarray(is_oo, dtype=bool)
        uu = np.asarray(is_uu, dtype=bool)
        y = np.asarray(y_joint, dtype=float)

        mean = np.mean(X, axis=0)
        scale = np.std(X, axis=0)
        scale = np.where(scale > 1e-6, scale, 1.0)
        Z = (X - mean) / scale
        d = Z.shape[1]

        def _fit_params(Z_sub, u_sub, v_sub, oo_sub, uu_sub, y_sub, reg_lam: float) -> Tuple[np.ndarray, float]:
            def _loss(params: np.ndarray) -> float:
                w = params[:d]
                b = params[d]
                tau = TAU_MAX * np.tanh(Z_sub @ w + b)
                jp = vectorized_copula_joint_prob(u_sub, v_sub, tau, oo_sub, uu_sub)
                nll = cls._joint_logloss(y_sub, jp)
                return nll + reg_lam * float(np.dot(w, w))

            init = np.zeros(d + 1, dtype=float)
            res = optimize.minimize(
                _loss,
                init,
                method="L-BFGS-B",
                options={"maxiter": max_iter, "ftol": 1e-11, "gtol": 1e-7, "eps": 1e-6},
            )
            return res.x[:d], float(res.x[d])

        best_lam = float(reg_grid[0])
        if len(reg_grid) > 1 and len(Z) >= 200:
            if game_timestamps is not None:
                order = np.argsort(game_timestamps, kind="mergesort")
            else:
                order = np.arange(len(Z))
            split = int(len(order) * 0.80)
            tr_idx, va_idx = order[:split], order[split:]
            best_val_ll = float("inf")
            for lam in reg_grid:
                w_cv, b_cv = _fit_params(
                    Z[tr_idx], u[tr_idx], v[tr_idx], oo[tr_idx], uu[tr_idx], y[tr_idx], float(lam)
                )
                tau_va = TAU_MAX * np.tanh(Z[va_idx] @ w_cv + b_cv)
                jp_va = vectorized_copula_joint_prob(u[va_idx], v[va_idx], tau_va, oo[va_idx], uu[va_idx])
                ll_va = cls._joint_logloss(y[va_idx], jp_va)
                if ll_va < best_val_ll:
                    best_val_ll = ll_va
                    best_lam = float(lam)

        w_final, b_final = _fit_params(Z, u, v, oo, uu, y, best_lam)
        tau_tr = TAU_MAX * np.tanh(Z @ w_final + b_final)
        jp_tr = vectorized_copula_joint_prob(u, v, tau_tr, oo, uu)
        train_ll = cls._joint_logloss(y, jp_tr)

        return cls(
            coef=w_final,
            intercept=b_final,
            scaler_mean=mean,
            scaler_scale=scale,
            reg_lambda=best_lam,
            feature_names=PAIR_FEATURE_NAMES,
            metadata={
                "seed": int(seed),
                "training_pairs": int(len(Z)),
                "selected_reg_lambda": float(best_lam),
                "train_joint_logloss": float(train_ll),
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_type": "conditional_copula_l2_joint_logloss",
            "tau_max": TAU_MAX,
            "feature_names": self.feature_names,
            "coef": [float(x) for x in self.coef],
            "intercept": float(self.intercept),
            "scaler_mean": [float(x) for x in self.scaler_mean],
            "scaler_scale": [float(x) for x in self.scaler_scale],
            "reg_lambda": float(self.reg_lambda),
            "metadata": self.metadata,
        }

    def save(self, path: str) -> None:
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ConditionalCopulaModel":
        with open(path, "r") as f:
            data = json.load(f)
        return cls(
            coef=np.asarray(data["coef"], dtype=float),
            intercept=float(data["intercept"]),
            scaler_mean=np.asarray(data["scaler_mean"], dtype=float),
            scaler_scale=np.asarray(data["scaler_scale"], dtype=float),
            reg_lambda=float(data.get("reg_lambda", 1e-3)),
            feature_names=data.get("feature_names", PAIR_FEATURE_NAMES),
            metadata=data.get("metadata", {}),
        )
