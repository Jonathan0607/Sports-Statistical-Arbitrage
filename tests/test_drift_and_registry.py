"""Unit and API integration tests for Phase 2 DriftMonitor & ModelRegistry."""

import json
import os
import shutil
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.drift_monitor import (
    DEFAULT_MONITORED_FEATURES,
    DriftAction,
    DriftMonitor,
    DriftThresholds,
    build_reference_stats,
    compute_brier_score,
    compute_ece,
    compute_log_loss,
    compute_psi,
)
from src.feature_contract import FEATURE_SCHEMA_VERSION
from src.model_registry import ModelRegistry, PromotionRejectedError, RegistryError
from src.serving.app import create_app
from src.serving.predictor import InferencePredictor


def _make_reference_stats(seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    n = 400
    df = pd.DataFrame(
        {
            "roll_min": np.clip(rng.normal(28.0, 5.0, size=n), 5.0, 44.0),
            "roll_stat": np.clip(rng.normal(16.0, 6.0, size=n), 1.0, 40.0),
            "roll_usg": np.clip(rng.normal(0.22, 0.05, size=n), 0.08, 0.40),
            "days_rest": np.clip(rng.integers(1, 5, size=n), 0, 7).astype(float),
            "opp_def_rating": rng.normal(113.0, 3.0, size=n),
            "opp_pace_factor": rng.normal(99.5, 2.5, size=n),
            "opp_pos_pts_allowed": rng.normal(21.5, 2.5, size=n),
            "vacated_usage": np.clip(rng.exponential(0.05, size=n), 0.0, 0.6),
            "pace_differential": rng.normal(0.0, 2.2, size=n),
            "roll_stat_var": np.clip(rng.normal(25.0, 8.0, size=n), 1.0, 80.0),
            "roll_residual": rng.normal(0.0, 1.5, size=n),
            "roll_ts_pct": np.clip(rng.normal(0.57, 0.05, size=n), 0.35, 0.80),
            "roll_touches": np.clip(rng.normal(52.0, 15.0, size=n), 10.0, 110.0),
        }
    )
    return build_reference_stats(df, features=DEFAULT_MONITORED_FEATURES, seed=seed)


class TestCalibrationAndPSIMetrics:
    def test_ece_brier_logloss_and_psi(self):
        rng = np.random.default_rng(123)
        probs = np.linspace(0.05, 0.95, 500)
        outcomes = (rng.random(500) < probs).astype(float)

        ece_good, bins = compute_ece(probs, outcomes, n_bins=10)
        assert len(bins) == 10
        assert ece_good < 0.07

        # Overconfident predictions (0.95 when true rate is 0.20)
        bad_probs = np.full(200, 0.95)
        bad_y = np.array([1.0] * 40 + [0.0] * 160)
        ece_bad, _ = compute_ece(bad_probs, bad_y, n_bins=10)
        assert ece_bad > 0.70

        assert compute_brier_score(probs, outcomes) < compute_brier_score(bad_probs, bad_y)
        assert compute_log_loss(probs, outcomes) < compute_log_loss(bad_probs, bad_y)

        ref = _make_reference_stats()["features"]["roll_min"]
        in_dist = np.array(ref["sample"])
        shifted = in_dist + 25.0
        assert compute_psi(ref["bin_proportions"], ref["bin_edges"], in_dist) < 0.05
        assert compute_psi(ref["bin_proportions"], ref["bin_edges"], shifted) > 0.25


class TestDriftMonitorAndCircuitBreaker:
    def test_healthy_baseline_stays_healthy(self, tmp_path):
        ref = _make_reference_stats()
        monitor = DriftMonitor(
            db_url=f"sqlite:///{tmp_path}/telemetry.sqlite",
            reference_stats=ref,
        )
        rng = np.random.default_rng(42)
        records = []
        n_samples = 600
        for i in range(n_samples):
            snap = {k: float(ref["features"][k]["sample"][i % len(ref["features"][k]["sample"])]) for k in DEFAULT_MONITORED_FEATURES}
            ewma = float(snap["roll_stat"])
            edge_sig = float(rng.normal(0.0, 1.2))
            meta = float(ewma + edge_sig)
            actual = float(meta + rng.normal(0.0, 0.8))
            prob = float(np.clip(0.5 + 0.35 * np.tanh((meta - ewma) / 1.5), 0.15, 0.85))
            hit = int(rng.random() < prob)
            records.append(
                {
                    "model_version": "v-test",
                    "feature_schema_version": FEATURE_SCHEMA_VERSION,
                    "player_id": str(1000 + i),
                    "stat_key": "pts",
                    "predicted_mean": meta,
                    "ewma_baseline": ewma,
                    "predicted_prob": prob,
                    "line": ewma,
                    "feature_snapshot": snap,
                    "realized_value": actual,
                    "realized_hit": hit,
                }
            )
        monitor.log_predictions(records)
        rep = monitor.evaluate_drift(window_days=7)
        assert rep["action"] == DriftAction.HEALTHY.value, f"Unexpected drift action: {rep['reasons']}"
        assert rep["fallback_active"] is False
        assert rep["settled_predictions"] == n_samples
        assert rep["calibration"]["meta_mae"] < rep["calibration"]["ewma_mae"]
        assert rep["covariate_drift"]["max_psi"] < 0.15

    def test_calibration_drift_trips_circuit_breaker(self, tmp_path):
        ref = _make_reference_stats()
        monitor = DriftMonitor(
            db_url=f"sqlite:///{tmp_path}/cal_drift.sqlite",
            reference_stats=ref,
        )
        records = []
        for i in range(50):
            snap = {k: float(ref["features"][k]["sample"][i % len(ref["features"][k]["sample"])]) for k in DEFAULT_MONITORED_FEATURES}
            actual = 20.0
            ewma = 20.2  # EWMA very accurate (MAE 0.2)
            meta = 25.0  # Meta catastrophically biased (MAE 5.0)
            records.append(
                {
                    "model_version": "v-bad",
                    "feature_schema_version": FEATURE_SCHEMA_VERSION,
                    "player_id": str(i),
                    "stat_key": "pts",
                    "predicted_mean": meta,
                    "ewma_baseline": ewma,
                    "predicted_prob": 0.92,  # Miscalibrated confidence vs 0% hit rate
                    "line": 22.0,
                    "feature_snapshot": snap,
                    "realized_value": actual,
                    "realized_hit": 0,
                }
            )
        monitor.log_predictions(records)
        rep = monitor.evaluate_drift(window_days=7)
        assert rep["action"] == DriftAction.CIRCUIT_BREAK_FALLBACK_EWMA.value
        assert rep["fallback_active"] is True
        assert rep["calibration"]["ece"] >= 0.80
        assert rep["calibration"]["mae_delta"] > 1.0

    def test_covariate_shift_and_settlement_and_discord(self, tmp_path):
        ref = _make_reference_stats()
        monitor = DriftMonitor(
            db_url=f"sqlite:///{tmp_path}/cov_drift.sqlite",
            reference_stats=ref,
            discord_webhook_url="https://discord.com/api/webhooks/mock-drift",
        )
        ids = []
        for i in range(45):
            snap = {k: float(ref["features"][k]["mean"]) for k in DEFAULT_MONITORED_FEATURES}
            # Severe regime shift on minutes and opponent pace
            snap["roll_min"] = 45.0 + (i % 4)
            snap["opp_pace_factor"] = 122.0 + (i % 3)
            ids.append(
                monitor.log_prediction(
                    model_version="v-test",
                    feature_schema_version=FEATURE_SCHEMA_VERSION,
                    player_id=str(i),
                    stat_key="pts",
                    predicted_mean=25.0,
                    ewma_baseline=24.0,
                    predicted_prob=0.58,
                    line=24.5,
                    feature_snapshot=snap,
                )
            )
        # Test record_settlement auto hit inference
        assert monitor.record_settlement(ids[0], realized_value=27.0) is True
        rep = monitor.evaluate_drift(window_days=7)
        assert rep["action"] == DriftAction.CIRCUIT_BREAK_FALLBACK_EWMA.value
        assert rep["fallback_active"] is True
        assert rep["covariate_drift"]["max_psi"] >= 0.25
        assert rep["covariate_drift"]["min_ks_pvalue"] < 0.001

        mock_resp = MagicMock()
        mock_resp.status = 204
        mock_ctx = MagicMock()
        mock_ctx.__enter__.return_value = mock_resp
        with patch("urllib.request.urlopen", return_value=mock_ctx) as mock_open:
            sent = monitor.send_discord_alert(rep)
            assert sent is True
            mock_open.assert_called_once()

    def test_serving_api_ewma_fallback_and_drift_endpoint(self, tmp_path):
        ref = _make_reference_stats()
        monitor = DriftMonitor(
            db_url=f"sqlite:///{tmp_path}/serving_drift.sqlite",
            reference_stats=ref,
        )
        predictor = InferencePredictor(
            model_dir="models/wf_fold2",
            auto_connect_redis=False,
            drift_monitor=monitor,
            log_predictions=False,
        )
        app = create_app(predictor=predictor)
        client = TestClient(app)

        payload = {
            "schema_version": FEATURE_SCHEMA_VERSION,
            "requests": [
                {
                    "league": "NBA",
                    "stat": "pts",
                    "thresholds": [24.5],
                    "features": {
                        "roll_min": 34.0,
                        "roll_stat": 24.0,
                        "roll_usg": 0.28,
                        "days_rest": 2.0,
                        "is_back_to_back": 0.0,
                        "opp_def_rating": 113.0,
                        "opp_pace_factor": 100.0,
                        "opp_pos_pts_allowed": 22.0,
                        "vacated_usage": 0.0,
                        "roll_potential_assists": 10.0,
                        "roll_touches": 70.0,
                        "roll_passes_made": 45.0,
                        "roll_time_of_possession_sec": 290.0,
                        "roll_defender_distance": 4.2,
                        "is_home_game": 1.0,
                        "pace_differential": 1.0,
                        "roll_stat_var": 30.0,
                        "roll_residual": 0.4,
                        "roll_ts_pct": 0.60,
                        "roll_reb_rate": 8.0,
                        "roll_ast_rate": 8.5,
                        "roll_shorthanded_points": 3.0,
                        "roll_plus_minus": 4.0,
                        "roll_usg_efficiency": 2.3,
                        "player_id": "2544",
                        "team_id": 14,
                        "opponent_team_id": 2,
                        "player_tier": 0,
                        "bayesian_roll_stat": 24.5,
                    },
                }
            ],
        }

        h0 = client.get("/health").json()
        assert h0["drift_status"] == DriftAction.HEALTHY.value
        assert h0["fallback_active"] is False

        d0 = client.get("/v1/drift").json()
        assert d0["action"] == DriftAction.HEALTHY.value

        # Force circuit breaker trip via severe covariate shift
        shifted_rows = [
            {
                "model_version": predictor.model_version,
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "player_id": str(i),
                "stat_key": "pts",
                "predicted_mean": 25.0,
                "ewma_baseline": 24.5,
                "feature_snapshot": {"roll_min": 55.0, "opp_pace_factor": 135.0},
            }
            for i in range(40)
        ]
        monitor.log_predictions(shifted_rows)

        h1 = client.get("/health").json()
        assert h1["drift_status"] == DriftAction.CIRCUIT_BREAK_FALLBACK_EWMA.value
        assert h1["fallback_active"] is True

        pred_resp = client.post("/v1/predict", json=payload).json()
        assert pred_resp["fallback_active"] is True
        item = pred_resp["predictions"][0]
        assert item["fallback_active"] is True
        assert item["residual"] == 0.0
        assert abs(item["predicted_mean"] - item["ewma_baseline"]) < 1e-9


class TestModelRegistryLifecycle:
    def _seed_dummy_model_dir(self, path, meta_schema: str = FEATURE_SCHEMA_VERSION, payload_seed: str = "a"):
        os.makedirs(path, exist_ok=True)
        for stat in ("pts", "reb", "ast"):
            with open(os.path.join(path, f"global_x{stat}_xgb.json"), "w") as f:
                f.write(f'{{"dummy_xgb": "{payload_seed}_{stat}"}}')
            with open(os.path.join(path, f"global_x{stat}_lgb.txt"), "w") as f:
                f.write(f"tree_{payload_seed}_{stat}\n")
            with open(os.path.join(path, f"global_x{stat}_meta.json"), "w") as f:
                json.dump(
                    {
                        "target": stat,
                        "feature_schema_version": meta_schema,
                        "train_seasons": ["2023-24", "2024-25"],
                        "training_samples": 55006,
                        "alpha": 0.35,
                    },
                    f,
                )

    def test_register_promote_gates_and_rollback(self, tmp_path):
        reg_dir = os.path.join(tmp_path, "registry")
        m_champ = os.path.join(tmp_path, "model_v1")
        m_bad_schema = os.path.join(tmp_path, "model_bad_schema")
        m_good = os.path.join(tmp_path, "model_v2")
        self._seed_dummy_model_dir(m_champ, FEATURE_SCHEMA_VERSION, "v1")
        self._seed_dummy_model_dir(m_bad_schema, "0.9.0", "v_old")
        self._seed_dummy_model_dir(m_good, FEATURE_SCHEMA_VERSION, "v2")

        registry = ModelRegistry(registry_dir=reg_dir)

        c1 = registry.register_version(
            model_dir=m_champ,
            git_sha="abc1234",
            metrics={"overall_mae": 2.624, "overall_crps": 1.887, "ewma_mae": 2.663, "brier_score": 0.21},
            promote_if_empty=True,
        )
        assert c1["status"] == "champion"
        assert c1["version_id"].endswith("-abc1234")
        assert len(c1["artifact_checksums"]) == 9
        assert registry.verify_checksums(c1)[0] is True

        # Reject 1: schema mismatch
        bad_schema_v = registry.register_version(
            model_dir=m_bad_schema,
            git_sha="bad0001",
            metrics={"overall_mae": 2.50, "overall_crps": 1.80, "ewma_mae": 2.66, "brier_score": 0.19},
        )
        with pytest.raises(PromotionRejectedError, match="Schema mismatch"):
            registry.promote_challenger(bad_schema_v["version_id"])

        # Reject 2: worse MAE than champion
        worse_mae_v = registry.register_version(
            model_dir=m_good,
            git_sha="bad0002",
            metrics={"overall_mae": 2.640, "overall_crps": 1.850, "ewma_mae": 2.663, "brier_score": 0.20},
        )
        with pytest.raises(PromotionRejectedError, match="overall_mae"):
            registry.promote_challenger(worse_mae_v["version_id"])

        # Reject 3: fails EWMA baseline
        worse_ewma_v = registry.register_version(
            model_dir=m_good,
            git_sha="bad0003",
            metrics={"overall_mae": 2.700, "overall_crps": 1.850, "ewma_mae": 2.663},
        )
        with pytest.raises(PromotionRejectedError, match="EWMA baseline"):
            registry.promote_challenger(worse_ewma_v["version_id"])

        # Reject 4: tampered artifact checksum
        tampered_dir = os.path.join(tmp_path, "model_tampered")
        shutil.copytree(m_good, tampered_dir)
        tampered_v = registry.register_version(
            model_dir=tampered_dir,
            git_sha="tamper1",
            metrics={"overall_mae": 2.550, "overall_crps": 1.820, "ewma_mae": 2.663, "brier_score": 0.19},
        )
        with open(os.path.join(tampered_dir, "global_xpts_xgb.json"), "w") as f:
            f.write('{"tampered": true}')
        with pytest.raises(PromotionRejectedError, match="Checksum mismatch"):
            registry.promote_challenger(tampered_v["version_id"])

        # Valid challenger promotion
        v2 = registry.register_version(
            model_dir=m_good,
            git_sha="def5678",
            metrics={"overall_mae": 2.580, "overall_crps": 1.850, "ewma_mae": 2.663, "brier_score": 0.20},
        )
        promoted = registry.promote_challenger(v2["version_id"])
        assert promoted["status"] == "champion"
        assert registry.get_champion()["version_id"] == v2["version_id"]
        assert registry.get_version(c1["version_id"])["status"] == "archived"

        # Atomic rollback
        restored = registry.rollback_to_previous(reason="canary_latency_spike")
        assert restored["version_id"] == c1["version_id"]
        assert restored["status"] == "champion"
        assert registry.get_champion()["version_id"] == c1["version_id"]
        assert registry.get_version(v2["version_id"])["status"] == "rolled_back"

        with pytest.raises(RegistryError, match="no previous champion"):
            registry.rollback_to_previous()
