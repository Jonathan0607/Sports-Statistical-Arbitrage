"""Unit and API integration tests for Phase 3 Conditional Copula theta(x)."""

import os
import sys

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.run_copula_ablation import _compute_causal_team_strength_map
from src.conditional_copula import (
    TAU_MAX,
    ConditionalCopulaModel,
    build_pair_feature_matrix,
    build_pair_feature_row,
    copula_joint_probability,
    ensure_psd_covariance,
    vectorized_copula_joint_prob,
)
from src.feature_contract import FEATURE_SCHEMA_VERSION
from src.serving.app import create_app
from src.serving.predictor import InferencePredictor


class TestFrechetHoeffdingAndAxioms:
    def test_bounds_and_tail_family_routing(self):
        grid = np.linspace(0.05, 0.95, 7)
        for u in grid:
            for v in grid:
                fh_low = max(0.0, u + v - 1.0)
                fh_high = min(u, v)

                jp_oo, fam_oo = copula_joint_probability(u, v, 0.35, "over", "over")
                assert fam_oo == "gumbel"
                assert fh_low - 1e-12 <= jp_oo <= fh_high + 1e-12
                assert jp_oo > u * v - 1e-9

                jp_uu, fam_uu = copula_joint_probability(u, v, 0.35, "under", "under")
                assert fam_uu == "clayton"
                assert fh_low - 1e-12 <= jp_uu <= fh_high + 1e-12
                assert jp_uu > u * v - 1e-9

                # Negative concordance must map to Frank / Gaussian with negative covariance
                jp_neg_f, fam_neg_f = copula_joint_probability(u, v, -0.45, "over", "under")
                assert fam_neg_f == "frank"
                assert fh_low - 1e-12 <= jp_neg_f <= fh_high + 1e-12
                assert jp_neg_f < u * v + 1e-9

                jp_neg_g, fam_neg_g = copula_joint_probability(
                    u, v, -0.45, "over", "over", negative_copula="gaussian"
                )
                assert fam_neg_g == "gaussian"
                assert fh_low - 1e-12 <= jp_neg_g <= fh_high + 1e-12
                assert jp_neg_g < u * v + 1e-9

                # Extreme tau inputs clamp safely within [-TAU_MAX, TAU_MAX]
                jp_clamp, _ = copula_joint_probability(u, v, 5.0, "over", "over")
                jp_max, _ = copula_joint_probability(u, v, TAU_MAX, "over", "over")
                assert abs(jp_clamp - jp_max) < 1e-12

    def test_vectorized_matches_scalar_and_swap_symmetry(self):
        rng = np.random.default_rng(42)
        u = rng.uniform(0.1, 0.9, size=60)
        v = rng.uniform(0.1, 0.9, size=60)
        taus = rng.uniform(-0.70, 0.70, size=60)
        sides = [("over", "over"), ("under", "under"), ("over", "under")]
        is_oo = np.array([i % 3 == 0 for i in range(60)])
        is_uu = np.array([i % 3 == 1 for i in range(60)])

        jp_vec = vectorized_copula_joint_prob(u, v, taus, is_oo, is_uu)
        for i in range(60):
            s1, s2 = sides[i % 3]
            jp_s, _ = copula_joint_probability(u[i], v[i], taus[i], s1, s2)
            assert abs(jp_vec[i] - jp_s) < 1e-8

        leg1 = {
            "team_id": 14,
            "position": "PG",
            "stat": "points",
            "side": "Over",
            "ewma_minutes": 34.0,
            "vacated_usage": 0.15,
            "opp_pace_factor": 102.0,
            "team_strength": 6.5,
        }
        leg2 = {
            "team_id": 2,
            "position": "C",
            "stat": "assists",
            "side": "Under",
            "ewma_minutes": 28.0,
            "vacated_usage": 0.0,
            "opp_pace_factor": 98.0,
            "team_strength": -3.2,
        }
        f12 = build_pair_feature_row(leg1, leg2)
        f21 = build_pair_feature_row(leg2, leg1)
        assert np.allclose(f12, f21, atol=1e-12)


class TestPSDAndTrainingDeterminism:
    def test_psd_covariance_guarantee(self):
        n = 12
        probs = np.linspace(0.35, 0.70, n)
        cov = np.diag(probs * (1.0 - probs))
        for i in range(n):
            for j in range(i + 1, n):
                tau = -0.55 if (i + j) % 2 == 0 else 0.45
                jp, _ = copula_joint_probability(probs[i], probs[j], tau, "over", "under")
                c_ij = jp - probs[i] * probs[j]
                cov[i, j] = c_ij
                cov[j, i] = c_ij

        psd = ensure_psd_covariance(cov, min_eig=1e-9)
        assert np.allclose(psd, psd.T, atol=1e-12)
        eigs = np.linalg.eigvalsh(psd)
        assert float(np.min(eigs)) >= 0.0

    def test_fit_recovers_conditional_structure_deterministically(self, tmp_path):
        rng = np.random.default_rng(42)
        n = 2500
        same_team = rng.integers(0, 2, size=n)
        same_pos = rng.integers(0, 2, size=n)
        dp = np.where(rng.random(n) < 0.5, "OO", "OU")
        pairs_df = pd.DataFrame(
            {
                "is_same_team": same_team,
                "is_same_position": same_pos,
                "projected_pace_diff": rng.uniform(0.0, 6.0, size=n),
                "mean_projected_pace": rng.normal(100.0, 2.5, size=n),
                "abs_team_strength_diff": rng.uniform(0.0, 15.0, size=n),
                "combined_vacated_usage": rng.uniform(0.0, 0.4, size=n),
                "min_player_ewma_minutes": rng.uniform(15.0, 36.0, size=n),
                "stat_pair_type": rng.choice(["pts-pts", "pts-ast", "reb-reb"], size=n),
                "direction_pair": dp,
            }
        )
        u = rng.uniform(0.35, 0.65, size=n)
        v = rng.uniform(0.35, 0.65, size=n)
        true_tau = np.where(dp == "OO", 0.45, -0.35)
        jp_true = vectorized_copula_joint_prob(u, v, true_tau, dp == "OO", np.zeros(n, dtype=bool))
        y_joint = (rng.random(n) < jp_true).astype(float)

        X = build_pair_feature_matrix(pairs_df)
        m1 = ConditionalCopulaModel.fit(X, u, v, dp == "OO", np.zeros(n, dtype=bool), y_joint, seed=42)
        m2 = ConditionalCopulaModel.fit(X, u, v, dp == "OO", np.zeros(n, dtype=bool), y_joint, seed=42)

        assert np.allclose(m1.coef, m2.coef, atol=1e-12)
        tau_pred = m1.predict_tau(X)
        assert float(np.mean(tau_pred[dp == "OO"])) > 0.15
        assert float(np.mean(tau_pred[dp == "OU"])) < -0.10

        out_file = tmp_path / "copula.json"
        m1.save(str(out_file))
        reloaded = ConditionalCopulaModel.load(str(out_file))
        assert np.allclose(m1.predict_tau(X), reloaded.predict_tau(X), atol=1e-12)

    def test_causal_team_strength_temporal_perturbation_invariance(self):
        rows = []
        for g_idx in range(6):
            gid = 1000 + g_idx
            st = pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(days=g_idx * 2)
            rows.append(
                {
                    "game_id": gid,
                    "start_time": st,
                    "season": "2024-25",
                    "home_team_id": 1,
                    "away_team_id": 2,
                    "team_id": 1,
                    "opponent_team_id": 2,
                    "points": 110 + g_idx,
                }
            )
            rows.append(
                {
                    "game_id": gid,
                    "start_time": st,
                    "season": "2024-25",
                    "home_team_id": 1,
                    "away_team_id": 2,
                    "team_id": 2,
                    "opponent_team_id": 1,
                    "points": 100,
                }
            )
        base_df = pd.DataFrame(rows)
        pert_df = base_df.copy()
        # Corrupt current game T (game_id=1004) and future game T+1 (1005) outcomes
        pert_df.loc[pert_df["game_id"] >= 1004, "points"] = 999

        m_base = _compute_causal_team_strength_map([base_df])
        m_pert = _compute_causal_team_strength_map([pert_df])

        assert abs(m_base[(1004, 1)] - m_pert[(1004, 1)]) < 1e-12
        assert abs(m_base[(1004, 2)] - m_pert[(1004, 2)]) < 1e-12


class TestAllocateConditionalCopulaIntegration:
    def test_allocate_endpoint_conditional_vs_static_and_validation(self):
        predictor = InferencePredictor(
            model_dir="models/wf_fold2",
            auto_connect_redis=False,
            log_predictions=False,
        )
        client = TestClient(create_app(predictor=predictor))

        base_payload = {
            "schema_version": FEATURE_SCHEMA_VERSION,
            "league": "NBA",
            "copula_type": "auto",
            "projected_pace_diff": 3.5,
            "abs_team_strength_diff": 11.0,
            "candidates": [
                {
                    "leg_id": "leg_a",
                    "player_id": "2544",
                    "team_id": 14,
                    "game_id": 500,
                    "stat": "points",
                    "side": "Over",
                    "marginal_prob": 0.58,
                    "payout_odds": 1.91,
                    "position": "SF",
                    "ewma_minutes": 34.5,
                    "vacated_usage": 0.12,
                    "opp_pace_factor": 101.2,
                    "team_strength": 6.0,
                },
                {
                    "leg_id": "leg_b",
                    "player_id": "203076",
                    "team_id": 14,
                    "game_id": 500,
                    "stat": "assists",
                    "side": "Over",
                    "marginal_prob": 0.56,
                    "payout_odds": 1.91,
                    "position": "PG",
                    "ewma_minutes": 32.0,
                    "vacated_usage": 0.12,
                    "opp_pace_factor": 101.2,
                    "team_strength": 6.0,
                },
                {
                    "leg_id": "leg_c",
                    "player_id": "1628983",
                    "team_id": 2,
                    "game_id": 500,
                    "stat": "points",
                    "side": "Under",
                    "marginal_prob": 0.57,
                    "payout_odds": 1.91,
                    "position": "SG",
                    "ewma_minutes": 35.0,
                    "vacated_usage": 0.0,
                    "opp_pace_factor": 98.4,
                    "team_strength": -5.0,
                },
            ],
        }

        res_cond = client.post("/v1/allocate", json=base_payload)
        assert res_cond.status_code == 200
        body_cond = res_cond.json()
        assert body_cond["copula_mode"] == "conditional"
        assert len(body_cond["joint_pairs"]) == 3
        for pair in body_cond["joint_pairs"]:
            assert 0.0 <= pair["joint_prob"] <= 1.0

        static_payload = dict(base_payload, use_conditional_copula=False)
        res_stat = client.post("/v1/allocate", json=static_payload)
        assert res_stat.status_code == 200
        assert res_stat.json()["copula_mode"] == "static"
