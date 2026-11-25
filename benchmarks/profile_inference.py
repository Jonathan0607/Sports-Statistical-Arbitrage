"""Reproducible stage-by-stage latency benchmark for the StatArb inference & risk pipeline.

Runs 100 warmup iterations followed by 1,000 timed iterations using `time.perf_counter_ns()`
across:
  1. Redis feature hydration (live local Redis or labeled mock fallback)
  2. Single & 32-leg batch XGBoost + LightGBM + Ridge residual inference
  3. ZINB survival + Clayton/Gumbel Copula evaluation
  4. C++ Eigen ColPivHouseholderQR Kelly solve vs. pure NumPy/SciPy baselines (N=5,15,30,50,
     including ill-conditioned and rank-deficient matrices)
  5. End-to-end FastAPI POST /v1/predict and POST /v1/allocate round-trips
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Callable, Dict, List
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import scipy.linalg
from fastapi.testclient import TestClient
from scipy import stats

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluate_fast import load_models
from src.feature_contract import FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION, predict_meta_ensemble
from src.risk import _HAS_CPP_KELLY, _quant_core
from src.serving.app import create_app
from src.serving.predictor import InferencePredictor
from src.serving.schemas import CopulaTypeEnum

WARMUP_ITERS = 100
TIMED_ITERS = 1000


def _benchmark_ns(fn: Callable[[], Any], warmup: int = WARMUP_ITERS, iters: int = TIMED_ITERS) -> Dict[str, float]:
    for _ in range(warmup):
        fn()
    samples_ns = np.empty(iters, dtype=np.int64)
    for i in range(iters):
        t0 = time.perf_counter_ns()
        fn()
        samples_ns[i] = time.perf_counter_ns() - t0

    samples_us = samples_ns / 1_000.0
    return {
        "mean_us": float(np.mean(samples_us)),
        "p50_us": float(np.percentile(samples_us, 50)),
        "p95_us": float(np.percentile(samples_us, 95)),
        "p99_us": float(np.percentile(samples_us, 99)),
        "mean_ms": float(np.mean(samples_us) / 1_000.0),
        "p50_ms": float(np.percentile(samples_us, 50) / 1_000.0),
        "p95_ms": float(np.percentile(samples_us, 95) / 1_000.0),
        "p99_ms": float(np.percentile(samples_us, 99) / 1_000.0),
        "warmup_iters": warmup,
        "timed_iters": iters,
    }


def _sample_feature_dict(player_id: str = "2544", idx: int = 0) -> Dict[str, Any]:
    return {
        "roll_min": 34.0 + (idx % 5),
        "roll_stat": 24.5 + (idx % 7),
        "roll_usg": 0.28,
        "days_rest": 2.0,
        "is_back_to_back": 0.0,
        "opp_def_rating": 113.0,
        "opp_pace_factor": 100.2,
        "opp_pos_pts_allowed": 22.5,
        "vacated_usage": 0.0,
        "roll_potential_assists": 10.5,
        "roll_touches": 76.0,
        "roll_passes_made": 50.0,
        "roll_time_of_possession_sec": 305.0,
        "roll_defender_distance": 4.3,
        "is_home_game": float(idx % 2),
        "pace_differential": 1.2,
        "roll_stat_var": 30.0,
        "roll_residual": 0.5,
        "roll_ts_pct": 0.60,
        "roll_reb_rate": 8.0,
        "roll_ast_rate": 8.5,
        "roll_shorthanded_points": 3.0,
        "roll_plus_minus": 4.2,
        "roll_usg_efficiency": 2.4,
        "player_id": str(player_id),
        "team_id": 14,
        "opponent_team_id": 2 + (idx % 10),
        "player_tier": 0,
    }


def _bench_redis_hydration() -> Dict[str, Any]:
    sample_records = [
        {
            "game_id": 100 + i,
            "start_time": f"2025-11-{(i % 28) + 1:02d}T19:00:00Z",
            "season": "2025-26",
            "home_team_id": 14,
            "away_team_id": 2 + (i % 10),
            "team_id": 14,
            "minutes_played": 33.0,
            "points": 24.0,
            "rebounds": 7.0,
            "assists": 8.0,
            "turnovers": 3.0,
        }
        for i in range(50)
    ]
    serialized = json.dumps(sample_records)
    key = "bench:player:history:2544"

    backend_label = "mock_redis_fallback"
    redis_client = None
    try:
        import redis

        live = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            decode_responses=True,
            socket_connect_timeout=0.5,
            socket_timeout=0.5,
        )
        if live.ping():
            live.set(key, serialized)
            redis_client = live
            backend_label = "live_local_redis"
    except Exception:
        redis_client = None

    if redis_client is None:
        mock = MagicMock()
        mock.ping.return_value = True
        mock.get.return_value = serialized
        redis_client = mock

    def _hydrate_once():
        raw = redis_client.get(key)
        _ = json.loads(raw)

    stats_dict = _benchmark_ns(_hydrate_once)
    if backend_label == "live_local_redis":
        redis_client.delete(key)

    return {"backend": backend_label, **stats_dict}


def _bench_ml_inference(models: dict) -> Dict[str, Any]:
    bundle = models["pts"]
    xgb_model = bundle["xgb"]
    lgb_model = bundle["lgb"]
    meta_data = bundle["meta"]
    xgb_cats = bundle.get("xgb_categories")

    df_single = pd.DataFrame([_sample_feature_dict("2544", 0)])
    ewma_single = np.array([24.5], dtype=float)
    roll_min_single = np.array([34.0], dtype=float)

    df_batch32 = pd.DataFrame([_sample_feature_dict("2544", i) for i in range(32)])
    ewma_batch32 = np.full(32, 24.5, dtype=float)
    roll_min_batch32 = np.full(32, 34.0, dtype=float)

    single_stats = _benchmark_ns(
        lambda: predict_meta_ensemble(
            df_single,
            FEATURE_COLUMNS,
            xgb_model=xgb_model,
            lgb_model=lgb_model,
            meta_data=meta_data,
            ewma_baseline=ewma_single,
            roll_min=roll_min_single,
            xgb_categories=xgb_cats,
        )
    )
    batch32_stats = _benchmark_ns(
        lambda: predict_meta_ensemble(
            df_batch32,
            FEATURE_COLUMNS,
            xgb_model=xgb_model,
            lgb_model=lgb_model,
            meta_data=meta_data,
            ewma_baseline=ewma_batch32,
            roll_min=roll_min_batch32,
            xgb_categories=xgb_cats,
        )
    )
    return {"single_leg": single_stats, "batch_32_legs": batch32_stats}


def _bench_zinb_and_copula() -> Dict[str, Any]:
    def _single_zinb_copula():
        mu, alpha_d, pi = 24.8, 0.2626, 0.02
        n_nb = 1.0 / alpha_d
        p_nb = n_nb / (n_nb + mu)
        cdf = pi + (1.0 - pi) * float(stats.nbinom.cdf(24, n_nb, p_nb))
        u = float(np.clip(1.0 - cdf, 1e-6, 1.0 - 1e-6))
        v = 0.57
        jp_gumbel, _ = InferencePredictor._copula_joint_probability(u, v, 0.15, "over", "over", CopulaTypeEnum.gumbel)
        jp_clayton, _ = InferencePredictor._copula_joint_probability(1.0 - u, 1.0 - v, 0.15, "under", "under", CopulaTypeEnum.clayton)
        return jp_gumbel, jp_clayton

    mus = np.linspace(12.0, 32.0, 32)
    alpha_d = 0.2626
    n_nb = 1.0 / alpha_d
    cutoffs = np.floor(mus - 0.5).astype(int)

    def _slate32_zinb_copula():
        p_nb = n_nb / (n_nb + mus)
        probs = np.clip(1.0 - (0.02 + 0.98 * stats.nbinom.cdf(cutoffs, n_nb, p_nb)), 1e-6, 1.0 - 1e-6)
        u_arr = probs[:-1]
        v_arr = probs[1:]
        tau = 0.15
        theta_g = 1.0 / (1.0 - tau)
        _ = np.exp(-(((-np.log(u_arr)) ** theta_g + (-np.log(v_arr)) ** theta_g) ** (1.0 / theta_g)))
        theta_c = 2.0 * tau / (1.0 - tau)
        _ = np.maximum(u_arr ** (-theta_c) + v_arr ** (-theta_c) - 1.0, 0.0) ** (-1.0 / theta_c)

    return {
        "single_pair": _benchmark_ns(_single_zinb_copula),
        "slate_32_legs": _benchmark_ns(_slate32_zinb_copula),
    }


def _make_kelly_test_case(n: int, regime: str, seed: int = 42):
    rng = np.random.default_rng(seed + n)
    if regime == "well_conditioned":
        A = rng.normal(0.0, 0.15, size=(n, n))
        sigma = (A @ A.T) / n + 0.10 * np.eye(n)
        w_true = rng.uniform(0.01, 0.05, size=n)
        mu = sigma @ w_true
    elif regime == "ill_conditioned":
        U, _ = np.linalg.qr(rng.normal(size=(n, n)))
        singular_vals = np.geomspace(1.0, 1e-10, num=n)
        sigma = (U * singular_vals) @ U.T
        sigma = 0.5 * (sigma + sigma.T)
        w_true = rng.uniform(0.01, 0.05, size=n)
        mu = sigma @ w_true
    elif regime == "rank_deficient":
        rank = max(1, n - 2)
        B = rng.normal(0.0, 0.2, size=(n, rank))
        sigma = (B @ B.T) / max(rank, 1)
        sigma = 0.5 * (sigma + sigma.T)
        w_true = rng.uniform(0.01, 0.05, size=n)
        mu = sigma @ w_true
    else:
        raise ValueError(f"Unknown regime {regime}")
    return sigma, mu


def _python_lstsq_kelly(mu: np.ndarray, sigma: np.ndarray, max_limit: float = 1e6, kelly_fraction: float = 1.0) -> np.ndarray:
    f = np.linalg.lstsq(sigma, mu, rcond=None)[0]
    f = np.clip(f, 0.0, None)
    total = float(np.sum(f))
    if total > max_limit and total > 1e-12:
        f = f * (max_limit / total)
    return f * kelly_fraction


def _scipy_qr_pivoting_kelly(mu: np.ndarray, sigma: np.ndarray, max_limit: float = 1e6, kelly_fraction: float = 1.0) -> np.ndarray:
    Q, R, P = scipy.linalg.qr(sigma, mode="full", pivoting=True)
    tol = max(sigma.shape) * np.finfo(float).eps * (abs(R[0, 0]) if R.size > 0 else 1.0)
    rank = int(np.sum(np.abs(np.diag(R)) > tol))
    qty = Q[:, :rank].T @ mu
    z = np.zeros(sigma.shape[1], dtype=float)
    if rank > 0:
        z[:rank] = scipy.linalg.solve_triangular(R[:rank, :rank], qty, lower=False)
    f = np.zeros_like(z)
    f[P] = z
    f = np.clip(f, 0.0, None)
    total = float(np.sum(f))
    if total > max_limit and total > 1e-12:
        f = f * (max_limit / total)
    return f * kelly_fraction


def _bench_kelly_solvers() -> List[Dict[str, Any]]:
    if not _HAS_CPP_KELLY or _quant_core is None:
        raise RuntimeError("C++ quant_core extension is required for Kelly benchmark")

    rows = []
    for regime in ("well_conditioned", "ill_conditioned", "rank_deficient"):
        for n in (5, 15, 30, 50):
            # All three solvers receive the exact same raw (sigma, mu) pair (no asymmetric shrinkage).
            sigma, mu = _make_kelly_test_case(n, regime=regime)
            sigma_list = sigma.tolist()
            mu_list = mu.tolist()
            assert np.allclose(np.asarray(sigma_list), sigma, rtol=0.0, atol=0.0)
            assert np.allclose(np.asarray(mu_list), mu, rtol=0.0, atol=0.0)

            # Benchmark latency and post-clipping residual ||Sigma * w - mu|| with non-binding cap (max_limit=1e6, kelly_fraction=1.0).
            # Note on rank-deficient matrices: ColPivHouseholderQR computes a sparse basic solution (non-pivot cols=0)
            # that hits ~5e-18 pre-clipping; subsequent long-only max(f, 0) clipping can leave non-zero residual if any
            # basic component was negative, whereas SVD minimum-norm lstsq can stay all-positive on the same nullspace.
            cpp_fn = lambda s=sigma_list, m=mu_list: _quant_core.optimize_multivariate_kelly(m, s, 1e6, 1.0)
            np_fn = lambda s=sigma, m=mu: _python_lstsq_kelly(m, s, 1e6, 1.0)
            scipy_fn = lambda s=sigma, m=mu: _scipy_qr_pivoting_kelly(m, s, 1e6, 1.0)

            cpp_stats = _benchmark_ns(cpp_fn)
            np_stats = _benchmark_ns(np_fn)
            scipy_stats = _benchmark_ns(scipy_fn)

            w_cpp = np.array(cpp_fn(), dtype=float)
            w_np = np_fn()
            w_scipy = scipy_fn()

            res_cpp = float(np.linalg.norm(sigma @ w_cpp - mu))
            res_np = float(np.linalg.norm(sigma @ w_np - mu))
            res_scipy = float(np.linalg.norm(sigma @ w_scipy - mu))
            f_np_raw = np.linalg.lstsq(sigma, mu, rcond=None)[0]
            unclipped_res_np = float(np.linalg.norm(sigma @ f_np_raw - mu))

            speedup_vs_np = np_stats["p50_us"] / max(cpp_stats["p50_us"], 1e-9)
            speedup_vs_scipy = scipy_stats["p50_us"] / max(cpp_stats["p50_us"], 1e-9)

            rows.append(
                {
                    "regime": regime,
                    "N": n,
                    "cpp_eigen": cpp_stats,
                    "numpy_lstsq": np_stats,
                    "scipy_piv_qr": scipy_stats,
                    "speedup_vs_numpy_p50": float(speedup_vs_np),
                    "speedup_vs_scipy_p50": float(speedup_vs_scipy),
                    "residual_norm_cpp": res_cpp,
                    "residual_norm_numpy": res_np,
                    "residual_norm_scipy": res_scipy,
                    "unconstrained_residual_norm_numpy": unclipped_res_np,
                }
            )
    return rows


def _bench_api_roundtrip() -> Dict[str, Any]:
    mock_redis = MagicMock()
    mock_redis.ping.return_value = True
    mock_redis.get.return_value = None
    predictor = InferencePredictor(
        model_dir="models/wf_fold2",
        redis_client=mock_redis,
        auto_connect_redis=False,
        log_predictions=False,
    )
    app = create_app(predictor=predictor)
    client = TestClient(app)

    req_single = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "requests": [
            {
                "schema_version": FEATURE_SCHEMA_VERSION,
                "league": "NBA",
                "stat": "pts",
                "features": _sample_feature_dict("2544", 0),
                "thresholds": [24.5],
            }
        ],
    }
    req_batch32 = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "requests": [
            {
                "schema_version": FEATURE_SCHEMA_VERSION,
                "league": "NBA",
                "stat": "pts",
                "features": _sample_feature_dict("2544", i),
                "thresholds": [24.5],
            }
            for i in range(32)
        ],
    }

    def _make_alloc_payload(n_legs: int) -> dict:
        return {
            "schema_version": FEATURE_SCHEMA_VERSION,
            "league": "NBA",
            "copula_type": "auto",
            "shrinkage_delta": 0.20,
            "candidates": [
                {
                    "leg_id": f"leg_{i}",
                    "player_id": str(2544 + i),
                    "team_id": 1 + (i % 4),
                    "game_id": 100 + (i // 4),
                    "stat": "points" if i % 2 == 0 else "assists",
                    "side": "Over",
                    "marginal_prob": 0.56 + (i % 5) * 0.01,
                    "payout_odds": 1.909,
                }
                for i in range(n_legs)
            ],
        }

    alloc15 = _make_alloc_payload(15)
    alloc32 = _make_alloc_payload(32)

    return {
        "predict_single": _benchmark_ns(lambda: client.post("/v1/predict", json=req_single)),
        "predict_batch_32": _benchmark_ns(lambda: client.post("/v1/predict", json=req_batch32)),
        "allocate_15": _benchmark_ns(lambda: client.post("/v1/allocate", json=alloc15)),
        "allocate_32": _benchmark_ns(lambda: client.post("/v1/allocate", json=alloc32)),
    }


def _format_markdown_report(report: Dict[str, Any]) -> str:
    lines = []
    lines.append("### Stage-by-Stage Inference & Optimization Latency Benchmark")
    lines.append(f"- Warmup: {WARMUP_ITERS} iterations | Timed: {TIMED_ITERS} iterations (`time.perf_counter_ns()`)")
    lines.append(f"- Redis Hydration Backend: `{report['redis_hydration']['backend']}`")
    lines.append("")
    lines.append("| Pipeline Stage | p50 (ms) | p95 (ms) | p99 (ms) | Mean (ms) |")
    lines.append("| :--- | ---: | ---: | ---: | ---: |")

    stage_rows = [
        ("Redis Feature Hydration (50-game JSON)", report["redis_hydration"]),
        ("Meta-Ensemble Inference (Single Leg)", report["ml_inference"]["single_leg"]),
        ("Meta-Ensemble Inference (32-Leg Batch)", report["ml_inference"]["batch_32_legs"]),
        ("ZINB + Clayton/Gumbel Copula (Single Pair)", report["zinb_copula"]["single_pair"]),
        ("ZINB + Clayton/Gumbel Copula (32-Leg Slate)", report["zinb_copula"]["slate_32_legs"]),
        ("API Round-Trip `POST /v1/predict` (1 Leg)", report["api_roundtrip"]["predict_single"]),
        ("API Round-Trip `POST /v1/predict` (32 Legs)", report["api_roundtrip"]["predict_batch_32"]),
        ("API Round-Trip `POST /v1/allocate` (15 Legs)", report["api_roundtrip"]["allocate_15"]),
        ("API Round-Trip `POST /v1/allocate` (32 Legs)", report["api_roundtrip"]["allocate_32"]),
    ]
    for label, s in stage_rows:
        lines.append(
            f"| {label} | {s['p50_ms']:.3f} | {s['p95_ms']:.3f} | {s['p99_ms']:.3f} | {s['mean_ms']:.3f} |"
        )

    lines.append("")
    lines.append("### C++ Eigen ColPivHouseholderQR vs. Pure Python/NumPy/SciPy Kelly Solver")
    lines.append("")
    lines.append(
        "| Matrix Regime | N | C++ Eigen p50 (us) | NumPy `lstsq` p50 (us) | SciPy Pivoted QR p50 (us) | Speedup vs NumPy | Speedup vs SciPy | Residual `||Σw - μ||` (Eigen) | Residual `||Σw - μ||` (NumPy) |"
    )
    lines.append("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in report["kelly_solver_comparison"]:
        lines.append(
            f"| {row['regime']} | {row['N']} | {row['cpp_eigen']['p50_us']:.2f} | "
            f"{row['numpy_lstsq']['p50_us']:.2f} | {row['scipy_piv_qr']['p50_us']:.2f} | "
            f"{row['speedup_vs_numpy_p50']:.2f}x | {row['speedup_vs_scipy_p50']:.2f}x | "
            f"{row['residual_norm_cpp']:.2e} | {row['residual_norm_numpy']:.2e} |"
        )

    return "\n".join(lines)


def main():
    models = load_models("models/wf_fold2")
    if "pts" not in models or "xgb" not in models["pts"]:
        raise RuntimeError("models/wf_fold2/global_xpts artifacts missing")

    report = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "cpp_eigen_available": bool(_HAS_CPP_KELLY),
        "warmup_iters": WARMUP_ITERS,
        "timed_iters": TIMED_ITERS,
        "redis_hydration": _bench_redis_hydration(),
        "ml_inference": _bench_ml_inference(models),
        "zinb_copula": _bench_zinb_and_copula(),
        "kelly_solver_comparison": _bench_kelly_solvers(),
        "api_roundtrip": _bench_api_roundtrip(),
    }

    os.makedirs("benchmarks", exist_ok=True)
    out_path = os.path.join("benchmarks", "latest_latency_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    markdown = _format_markdown_report(report)
    print("\n" + markdown + "\n")
    print(f"Saved structured benchmark report to {out_path}")


if __name__ == "__main__":
    main()
