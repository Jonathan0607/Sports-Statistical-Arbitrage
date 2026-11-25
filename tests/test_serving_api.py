"""FastAPI TestClient test suite for src/serving/ inference and allocation endpoints."""

import json
import os
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_contract import FEATURE_SCHEMA_VERSION
from src.serving.app import create_app
from src.serving.predictor import InferencePredictor


def _sample_feature_payload(player_id: str = "2544") -> dict:
    return {
        "roll_min": 34.5,
        "roll_stat": 24.8,
        "roll_usg": 0.29,
        "days_rest": 2.0,
        "is_back_to_back": 0.0,
        "opp_def_rating": 113.5,
        "opp_pace_factor": 100.2,
        "opp_pos_pts_allowed": 22.5,
        "vacated_usage": 0.0,
        "roll_potential_assists": 11.2,
        "roll_touches": 78.4,
        "roll_passes_made": 52.1,
        "roll_time_of_possession_sec": 310.0,
        "roll_defender_distance": 4.3,
        "is_home_game": 1.0,
        "pace_differential": 1.4,
        "roll_stat_var": 32.5,
        "roll_residual": 0.6,
        "roll_ts_pct": 0.61,
        "roll_reb_rate": 8.4,
        "roll_ast_rate": 8.8,
        "roll_shorthanded_points": 3.2,
        "roll_plus_minus": 4.5,
        "roll_usg_efficiency": 2.45,
        "player_id": player_id,
        "team_id": 14,
        "opponent_team_id": 2,
        "player_tier": 0,
        "bayesian_roll_stat": 24.5,
    }


@pytest.fixture
def mock_redis():
    client = MagicMock()
    client.ping.return_value = True
    history_records = [
        {
            "game_id": 100 + i,
            "start_time": f"2025-11-0{i + 1}T19:00:00Z",
            "season": "2025-26",
            "home_team_id": 14,
            "away_team_id": 2 + i,
            "team_id": 14,
            "minutes_played": 33.0 + (i % 3),
            "points": 24 + i,
            "rebounds": 7,
            "assists": 8,
            "turnovers": 3,
            "field_goals_attempted": 18,
            "free_throws_attempted": 6,
            "plus_minus": 5.0,
            "potential_assists": 12.0,
            "touches": 75.0,
            "passes_made": 48.0,
            "time_of_possession_sec": 290.0,
            "avg_defender_distance_ft": 4.2,
            "shorthanded_points": 2.0,
        }
        for i in range(6)
    ]

    def _get(key):
        if key == "player:history:2544":
            return json.dumps(history_records)
        return None

    client.get.side_effect = _get
    return client


@pytest.fixture
def client(mock_redis):
    predictor = InferencePredictor(
        model_dir="models/wf_fold2",
        redis_client=mock_redis,
        auto_connect_redis=False,
    )
    app = create_app(predictor=predictor)
    return TestClient(app)


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert body["feature_schema_version"] == FEATURE_SCHEMA_VERSION
    assert body["redis_connected"] is True
    assert body["redis_status"] == "connected"
    assert body["cpp_eigen_available"] is True
    assert "pts" in body["loaded_targets"]
    assert "git_sha" in body and len(body["git_sha"]) > 0


def test_valid_batch_predict_features_and_redis_hydration(client):
    payload = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "requests": [
            {
                "schema_version": FEATURE_SCHEMA_VERSION,
                "league": "NBA",
                "stat": "pts",
                "features": _sample_feature_payload("2544"),
                "thresholds": [20.5, 24.5, 29.5],
            },
            {
                "schema_version": FEATURE_SCHEMA_VERSION,
                "league": "NBA",
                "stat": "reb",
                "player_id": "2544",
                "game_context": {
                    "game_id": 999,
                    "start_time": "2025-11-15T19:30:00Z",
                    "season": "2025-26",
                    "home_team_id": 14,
                    "away_team_id": 5,
                    "team_id": 14,
                    "opponent_team_id": 5,
                    "opp_def_rating": 112.0,
                    "opp_pace_factor": 99.5,
                    "opp_pos_pts_allowed": 22.0,
                    "position": "SF",
                },
                "thresholds": [6.5, 8.5],
            },
        ],
    }

    resp = client.post("/v1/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["schema_version"] == FEATURE_SCHEMA_VERSION
    assert data["latency_ms"] >= 0.0
    assert len(data["predictions"]) == 2

    p0, p1 = data["predictions"]
    assert p0["player_id"] == "2544"
    assert p0["stat"] == "pts"
    assert p0["cold_start"] is False
    assert np.isfinite(p0["predicted_mean"]) and p0["predicted_mean"] > 0.0
    assert p0["zinb"]["alpha_disp"] > 0.0
    assert 0.0 <= p0["zinb"]["zero_inflation_prob"] < 1.0
    assert len(p0["survival_probabilities"]) == 3
    # Survival probabilities must be monotonically non-increasing with higher thresholds
    probs_over = [s["prob_over"] for s in p0["survival_probabilities"]]
    assert probs_over[0] >= probs_over[1] >= probs_over[2]

    assert p1["player_id"] == "2544"
    assert p1["stat"] == "reb"
    assert np.isfinite(p1["predicted_mean"]) and p1["predicted_mean"] > 0.0
    assert len(p1["survival_probabilities"]) == 2


def test_cold_start_unseen_player_handling(client):
    payload = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "requests": [
            # Case 1: Unseen player_id with full features
            {
                "schema_version": FEATURE_SCHEMA_VERSION,
                "league": "NBA",
                "stat": "pts",
                "features": _sample_feature_payload("cold_start_rookie_999999"),
                "thresholds": [14.5],
            },
            # Case 2: Unseen player_id with Redis cache miss (0 prior games)
            {
                "schema_version": FEATURE_SCHEMA_VERSION,
                "league": "NBA",
                "stat": "ast",
                "player_id": "completely_unseen_rookie_888888",
                "game_context": {
                    "team_id": 14,
                    "opponent_team_id": 2,
                    "position": "PG",
                },
                "thresholds": [4.5],
            },
        ],
    }

    resp = client.post("/v1/predict", json=payload)
    assert resp.status_code == 200
    preds = resp.json()["predictions"]
    assert len(preds) == 2
    assert preds[0]["cold_start"] is True
    assert np.isfinite(preds[0]["predicted_mean"]) and preds[0]["predicted_mean"] > 0.0
    assert preds[1]["cold_start"] is True
    assert np.isfinite(preds[1]["predicted_mean"]) and preds[1]["predicted_mean"] > 0.0
    assert 0.0 <= preds[1]["survival_probabilities"][0]["prob_over"] <= 1.0


def test_valid_allocate_with_copula_and_cpp_eigen(client):
    payload = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "league": "NBA",
        "copula_type": "auto",
        "shrinkage_delta": 0.20,
        "max_bankroll_fraction": 0.05,
        "kelly_fraction": 0.25,
        "bankroll_usd": 50000.0,
        "candidates": [
            {
                "leg_id": "leg_1",
                "player_id": "2544",
                "team_id": 14,
                "game_id": 1001,
                "stat": "points",
                "side": "Over",
                "marginal_prob": 0.60,
                "payout_odds": 1.909,
            },
            {
                "leg_id": "leg_2",
                "player_id": "203076",
                "team_id": 14,
                "game_id": 1001,
                "stat": "assists",
                "side": "Over",
                "marginal_prob": 0.58,
                "payout_odds": 1.909,
            },
            {
                "leg_id": "leg_3",
                "player_id": "201939",
                "team_id": 10,
                "game_id": 1002,
                "stat": "points",
                "side": "Over",
                "marginal_prob": 0.57,
                "payout_odds": 1.909,
            },
        ],
    }

    resp = client.post("/v1/allocate", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["solver_backend"] == "cpp_eigen_colpiv_qr"
    assert len(data["allocations"]) == 3
    assert all(np.isfinite(x) and x >= 0.0 for x in data["allocations"])
    assert sum(data["allocations"]) <= 0.05 * 0.25 + 1e-9
    assert data["condition_number_raw"] >= 1.0
    assert data["condition_number_shrunk"] >= 1.0
    assert np.isfinite(data["residual_norm"])
    assert data["numerical_rank"] == 3
    assert data["is_rank_deficient"] is False
    # Teammates leg_1 & leg_2 (pts Over + ast Over) should emit Gumbel joint pricing
    assert len(data["joint_pairs"]) >= 1
    assert data["joint_pairs"][0]["copula_used"] == "gumbel"


def test_allocate_rank_deficient_matrix_stability(client):
    # Explicitly singular / collinear 3x3 covariance matrix
    singular_cov = [
        [0.25, 0.25, 0.0],
        [0.25, 0.25, 0.0],
        [0.0, 0.0, 0.20],
    ]
    payload = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "league": "NBA",
        "covariance_matrix": singular_cov,
        "shrinkage_delta": 0.20,
        "candidates": [
            {"player_id": "p1", "stat": "points", "side": "Over", "marginal_prob": 0.58, "payout_odds": 1.91},
            {"player_id": "p2", "stat": "points", "side": "Over", "marginal_prob": 0.58, "payout_odds": 1.91},
            {"player_id": "p3", "stat": "rebounds", "side": "Over", "marginal_prob": 0.56, "payout_odds": 1.91},
        ],
    }
    resp = client.post("/v1/allocate", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_rank_deficient"] is True
    assert data["numerical_rank"] == 2
    assert all(np.isfinite(x) and x >= 0.0 for x in data["allocations"])


@pytest.mark.parametrize(
    "mutator",
    [
        lambda req: req["requests"][0]["features"].update({"roll_stat_var": -0.5}),  # Negative variance
        lambda req: req["requests"][0].update({"league": "NFL"}),                   # Invalid league enum
        lambda req: req["requests"][0].update({"schema_version": "0.0.1"}),         # Schema mismatch
        lambda req: req["requests"][0]["features"].pop("roll_min"),                 # Missing required field
        lambda req: req["requests"][0]["features"].update({"roll_min": 99.0}),      # Out-of-bounds minutes
        lambda req: req["requests"][0]["features"].update({"unknown_field": 1.0}),  # Extra field forbidden
        lambda req: req["requests"][0].pop("features"),                             # Missing features & player_id
    ],
)
def test_predict_validation_rejections(client, mutator):
    req = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "requests": [
            {
                "schema_version": FEATURE_SCHEMA_VERSION,
                "league": "NBA",
                "stat": "pts",
                "features": _sample_feature_payload("2544"),
            }
        ],
    }
    mutator(req)
    resp = client.post("/v1/predict", json=req)
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"] == "validation_error"


def test_reject_nan_and_inf_in_raw_json(client):
    feat = _sample_feature_payload("2544")
    feat["roll_stat"] = "NaN"
    req_nan_str = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "requests": [
            {
                "schema_version": FEATURE_SCHEMA_VERSION,
                "league": "NBA",
                "stat": "pts",
                "features": feat,
            }
        ],
    }
    assert client.post("/v1/predict", json=req_nan_str).status_code == 422

    raw_nan_json = (
        f'{{"schema_version":"{FEATURE_SCHEMA_VERSION}","requests":[{{"schema_version":"{FEATURE_SCHEMA_VERSION}",'
        f'"league":"NBA","stat":"pts","features":'
        + json.dumps(_sample_feature_payload("2544")).replace("24.8", "NaN")
        + "}]}"
    )
    assert (
        client.post(
            "/v1/predict",
            content=raw_nan_json.encode("utf-8"),
            headers={"Content-Type": "application/json"},
        ).status_code
        == 422
    )

    raw_inf_json = raw_nan_json.replace("NaN", "Infinity")
    assert (
        client.post(
            "/v1/predict",
            content=raw_inf_json.encode("utf-8"),
            headers={"Content-Type": "application/json"},
        ).status_code
        == 422
    )


def test_allocate_validation_rejections(client):
    bad_prob = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "league": "NBA",
        "candidates": [
            {"player_id": "2544", "stat": "points", "side": "Over", "marginal_prob": 1.25}
        ],
    }
    assert client.post("/v1/allocate", json=bad_prob).status_code == 422

    bad_cov_shape = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "league": "NBA",
        "covariance_matrix": [[0.25]],
        "candidates": [
            {"player_id": "2544", "stat": "points", "side": "Over", "marginal_prob": 0.58},
            {"player_id": "201939", "stat": "points", "side": "Over", "marginal_prob": 0.58},
        ],
    }
    assert client.post("/v1/allocate", json=bad_cov_shape).status_code == 422


def test_hfp_poller_http_inference_integration(mock_redis):
    import asyncio
    from unittest.mock import patch
    import httpx
    import pandas as pd
    from scrapers.hfp_poller import UnifiedRestPoller

    predictor = InferencePredictor(
        model_dir="models/wf_fold2",
        redis_client=mock_redis,
        auto_connect_redis=False,
    )
    app = create_app(predictor=predictor)
    poller = UnifiedRestPoller(leagues=["NBA"], inference_url="http://testserver")
    assert poller.inference_url == "http://testserver"

    history_df = pd.DataFrame(
        [
            {
                "game_id": 100 + i,
                "start_time": f"2025-11-0{i + 1}T19:00:00Z",
                "season": "2025-26",
                "home_team_id": 14,
                "away_team_id": 2 + i,
                "team_id": 14,
                "minutes_played": 33.0,
                "points": 24 + i,
                "rebounds": 7,
                "assists": 8,
                "turnovers": 3,
            }
            for i in range(6)
        ]
    )

    transport = httpx.ASGITransport(app=app)
    orig_async_client = httpx.AsyncClient

    def _mock_client(*args, **kwargs):
        return orig_async_client(transport=transport, base_url="http://testserver")

    async def _run():
        with patch("httpx.AsyncClient", side_effect=_mock_client):
            prob, zinb = await poller.predict_via_http(
                player_name="LeBron James",
                stat_col="points",
                line=24.5,
                master_id="2544",
                history_df=history_df,
                league="NBA",
            )
        assert 0.0 <= prob <= 1.0
        assert zinb.mu > 0.0
        assert zinb.n > 0.0

    asyncio.run(_run())

