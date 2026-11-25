"""Deterministic unit tests for C++20 Eigen & pybind11 bindings (`src/quant_core`)."""

import math
import os
import sys

import numpy as np
import pytest
import scipy.linalg

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import quant_core
from src.models import MarketMicrostructure


def _american_to_implied(american: int) -> float:
    return (100.0 / (american + 100.0)) if american > 0 else (abs(american) / (abs(american) + 100.0))


def _recover_shin_z(pi_vec: np.ndarray, p_vec: np.ndarray) -> float:
    """Recovers Shin insider trading parameter z from implied probs pi and true probs p."""
    s = float(np.sum(pi_vec))
    if s <= 1.0 + 1e-12:
        return 0.0
    # From Shin (1991/1993): p_i^2 * (1 - z) + p_i * z - pi_i^2 / s = 0
    # => z * (p_i - p_i^2) = pi_i^2 / s - p_i^2
    num = float((pi_vec[0] ** 2) / s - (p_vec[0] ** 2))
    den = float(p_vec[0] - (p_vec[0] ** 2))
    return num / den if abs(den) > 1e-12 else 0.0


@pytest.mark.parametrize(
    "odds_over,odds_under",
    [
        (-110, -110),
        (-115, -105),
        (-220, 180),
        (-350, 270),
        (140, -170),
    ],
)
def test_calculate_shins_probs_matches_python_reference(odds_over, odds_under):
    pi_over = _american_to_implied(odds_over)
    pi_under = _american_to_implied(odds_under)

    py_res = MarketMicrostructure.shins_method(odds_over, odds_under)
    cpp_from_probs = quant_core.calculate_shins_probs([pi_over, pi_under])
    cpp_from_decimal = quant_core.calculate_shins_probs([1.0 / pi_over, 1.0 / pi_under])

    assert len(cpp_from_probs) == 2
    assert len(cpp_from_decimal) == 2
    assert math.isclose(sum(cpp_from_probs), 1.0, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(sum(cpp_from_decimal), 1.0, rel_tol=0.0, abs_tol=1e-12)
    assert all(0.0 < p < 1.0 for p in cpp_from_probs)

    assert abs(cpp_from_probs[0] - py_res["true_prob_over"]) < 1e-6
    assert abs(cpp_from_probs[1] - py_res["true_prob_under"]) < 1e-6
    assert abs(cpp_from_decimal[0] - py_res["true_prob_over"]) < 1e-6

    z_est = _recover_shin_z(np.array([pi_over, pi_under]), np.array(cpp_from_probs))
    assert z_est >= -1e-7
    assert py_res["z_factor"] >= 0.0


def test_calculate_shins_probs_no_vig_and_validation():
    no_vig = quant_core.calculate_shins_probs([2.10, 2.10])
    assert math.isclose(sum(no_vig), 1.0, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(no_vig[0], 0.5, rel_tol=0.0, abs_tol=1e-12)

    with pytest.raises(ValueError):
        quant_core.calculate_shins_probs([])
    with pytest.raises(ValueError):
        quant_core.calculate_shins_probs([-1.5, 2.0])


def _make_matrix_case(n: int, regime: str, seed: int = 123):
    rng = np.random.default_rng(seed + n)
    if regime == "well_conditioned":
        A = rng.normal(0.0, 0.15, size=(n, n))
        sigma = 0.5 * ((A @ A.T) / n + ((A @ A.T) / n).T) + 0.10 * np.eye(n)
    elif regime == "ill_conditioned":
        U, _ = np.linalg.qr(rng.normal(size=(n, n)))
        singular_vals = np.geomspace(1.0, 1e-9, num=n)
        sigma = (U * singular_vals) @ U.T
        sigma = 0.5 * (sigma + sigma.T)
    elif regime == "rank_deficient":
        rank = max(1, n - 2)
        B = rng.normal(0.0, 0.2, size=(n, rank))
        sigma = 0.5 * ((B @ B.T) / rank + ((B @ B.T) / rank).T)
    else:
        raise ValueError(regime)
    w_true = rng.uniform(0.005, 0.02, size=n)
    mu = sigma @ w_true
    return sigma, mu, w_true


@pytest.mark.parametrize("regime,n", [("well_conditioned", 5), ("well_conditioned", 15), ("ill_conditioned", 15)])
def test_optimize_multivariate_kelly_matches_pivoted_qr(regime, n):
    sigma, mu, _ = _make_matrix_case(n, regime)
    max_limit = 0.50
    kelly_frac = 0.25

    cpp_w = np.asarray(
        quant_core.optimize_multivariate_kelly(mu.tolist(), sigma.tolist(), max_limit, kelly_frac),
        dtype=float,
    )

    Q, R, P = scipy.linalg.qr(sigma, mode="full", pivoting=True)
    tol = max(sigma.shape) * np.finfo(float).eps * abs(R[0, 0])
    rank = int(np.sum(np.abs(np.diag(R)) > tol))
    z = np.zeros(n)
    z[:rank] = scipy.linalg.solve_triangular(R[:rank, :rank], Q[:, :rank].T @ mu, lower=False)
    ref = np.zeros(n)
    ref[P] = z
    ref = np.clip(ref, 0.0, None)
    if np.sum(ref) > max_limit:
        ref *= max_limit / np.sum(ref)
    ref *= kelly_frac

    assert np.all(np.isfinite(cpp_w))
    assert np.all(cpp_w >= 0.0)
    assert np.sum(cpp_w) <= max_limit * kelly_frac + 1e-12
    assert np.allclose(cpp_w, ref, rtol=0.0, atol=1e-7)


def test_optimize_multivariate_kelly_rank_deficient_and_constraints():
    # Exact collinear covariance (rank 1 in 2x2) and rank-deficient 15x15
    collinear_cov = [[0.25, 0.25], [0.25, 0.25]]
    edges = [0.08, 0.06]
    w_col = np.asarray(quant_core.optimize_multivariate_kelly(edges, collinear_cov, 0.05, 0.25), dtype=float)
    assert np.all(np.isfinite(w_col))
    assert np.all(w_col >= 0.0)
    assert np.sum(w_col) <= 0.05 * 0.25 + 1e-12

    sigma15, mu15, _ = _make_matrix_case(15, "rank_deficient")
    w15 = np.asarray(
        quant_core.optimize_multivariate_kelly(mu15.tolist(), sigma15.tolist(), 0.25, 0.5),
        dtype=float,
    )
    assert np.all(np.isfinite(w15))
    assert np.all(w15 >= 0.0)
    assert np.sum(w15) <= 0.25 * 0.5 + 1e-12

    with pytest.raises(ValueError):
        quant_core.optimize_multivariate_kelly([0.1, 0.2], [[0.1]], 0.05, 0.25)
