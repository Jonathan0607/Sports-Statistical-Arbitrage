import os
import sys
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.historical_replay_core import HistoricalReplayCore


def test_init():
    """Test that HistoricalReplayCore initializes successfully."""
    replay = HistoricalReplayCore()
    assert replay.engine is not None
    assert replay.feature_engine is not None
    assert "Points" in replay.market_model_map
    assert "points" in replay.stat_to_market


def test_evaluate_prediction_accuracy_insufficient_history():
    """Test that evaluate_prediction_accuracy returns None if history is too short."""
    replay = HistoricalReplayCore()

    mock_history = pd.DataFrame([
        {"game_id": 101, "start_time": "2025-10-22", "minutes_played": 30.0, "points": 15}
    ])

    with patch.object(replay.engine, "connect"), \
         patch("pandas.read_sql", return_value=mock_history):
        res = replay.evaluate_prediction_accuracy(
            player_id="player_mock",
            player_name="Mock Player",
            stat_category="points",
            game_time="2025-10-25 19:00:00",
            actual_outcome_value=18.0,
        )
        assert res is None


def test_evaluate_prediction_accuracy_success():
    """Test that evaluate_prediction_accuracy computes naive baseline (last 5 mean) and XGBoost mu."""
    replay = HistoricalReplayCore()

    # Mock 6 game logs
    mock_history = pd.DataFrame([
        {"game_id": 1, "start_time": "2025-10-20", "home_team_id": 1, "away_team_id": 2, "minutes_played": 30.0, "points": 10, "rebounds": 5, "assists": 4, "turnovers": 2, "steals": 1, "blocks": 0, "three_pointers_made": 1, "usage_rate": 0.2},
        {"game_id": 2, "start_time": "2025-10-22", "home_team_id": 1, "away_team_id": 3, "minutes_played": 32.0, "points": 20, "rebounds": 6, "assists": 5, "turnovers": 1, "steals": 2, "blocks": 1, "three_pointers_made": 2, "usage_rate": 0.25},
        {"game_id": 3, "start_time": "2025-10-24", "home_team_id": 1, "away_team_id": 4, "minutes_played": 31.0, "points": 15, "rebounds": 4, "assists": 3, "turnovers": 3, "steals": 0, "blocks": 0, "three_pointers_made": 1, "usage_rate": 0.22},
        {"game_id": 4, "start_time": "2025-10-26", "home_team_id": 1, "away_team_id": 5, "minutes_played": 35.0, "points": 25, "rebounds": 7, "assists": 6, "turnovers": 2, "steals": 1, "blocks": 1, "three_pointers_made": 3, "usage_rate": 0.28},
        {"game_id": 5, "start_time": "2025-10-28", "home_team_id": 1, "away_team_id": 6, "minutes_played": 28.0, "points": 30, "rebounds": 8, "assists": 7, "turnovers": 4, "steals": 3, "blocks": 2, "three_pointers_made": 4, "usage_rate": 0.3},
        {"game_id": 6, "start_time": "2025-10-30", "home_team_id": 1, "away_team_id": 7, "minutes_played": 34.0, "points": 35, "rebounds": 9, "assists": 8, "turnovers": 1, "steals": 1, "blocks": 0, "three_pointers_made": 5, "usage_rate": 0.32},
    ])

    # Expected EWMA baseline for points (alpha=0.3, adjust=False)
    expected_ewma = float(mock_history["points"].ewm(alpha=0.3, adjust=False).mean().iloc[-1])

    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([-0.5])
    replay.loaded_models["Points"] = mock_model
    replay.loaded_ensembles["Points"] = {
        "xgb": mock_model,
        "lgb": mock_model,
        "meta": {"intercept": 0.0, "weights": [1.0, 0.0]}
    }

    mock_active_feats = {
        "days_rest": 2,
        "is_back_to_back": 0,
        "opp_def_rating": 110.0,
        "opp_pace_factor": 100.0,
        "opp_pos_pts_allowed": 22.0,
        "vacated_usage": 0.0,
        "roll_potential_assists": 3.5,
        "roll_touches": 55.0,
        "roll_defender_distance": 4.2,
    }

    with patch.object(replay.engine, "connect"), \
         patch("pandas.read_sql", return_value=mock_history), \
         patch.object(replay.feature_engine, "build_active_features", return_value=mock_active_feats):

        res = replay.evaluate_prediction_accuracy(
            player_id="player_test",
            player_name="Test Player",
            stat_category="points",
            game_time="2025-11-01 19:00:00",
            actual_outcome_value=27.0,
        )

        assert res is not None
        assert res["player_id"] == "player_test"
        assert res["player_name"] == "Test Player"
        assert res["stat_category"] == "Points"
        assert res["actual_value"] == 27.0
        assert abs(res["ewma_baseline"] - expected_ewma) < 1e-4
        assert abs(res["xgboost_prediction"] - (expected_ewma - 0.5)) < 1e-4
        assert abs(res["meta_prediction"] - (expected_ewma - 0.5)) < 1e-4
        assert abs(res["xgb_residual"] - (-0.5)) < 1e-4


def test_run_backtest_with_mock_data():
    """Test run_backtest flow with mock game logs and predictions."""
    replay = HistoricalReplayCore()

    mock_logs = pd.DataFrame([
        {"game_id": 1, "player_id": "p1", "player_name": "Player One", "start_time": "2025-11-15 19:00:00", "points": 20, "rebounds": 5, "assists": 4},
        {"game_id": 2, "player_id": "p2", "player_name": "Player Two", "start_time": "2025-11-15 19:30:00", "points": 10, "rebounds": 8, "assists": 2},
    ])

    eval_return = {
        "player_id": "p1",
        "player_name": "Player One",
        "stat_category": "Points",
        "game_time": "2025-11-15 19:00:00",
        "actual_value": 20.0,
        "meta_prediction": 19.8,
        "xgboost_prediction": 19.5,
        "lightgbm_prediction": 20.1,
        "ewma_baseline": 18.0,
        "crps": 1.1,
    }

    with patch.object(replay.engine, "connect"), \
         patch("pandas.read_sql", return_value=mock_logs), \
         patch.object(replay, "evaluate_prediction_accuracy", return_value=eval_return):

        results = replay.run_backtest(
            start_date="2025-11-15 00:00:00",
            end_date="2025-11-17 23:59:59",
            stat_categories=["points"],
        )

        assert len(results) == 2
        assert results[0]["actual_value"] == 20.0
        assert results[0]["meta_prediction"] == 19.8
        assert results[0]["xgboost_prediction"] == 19.5
        assert results[0]["ewma_baseline"] == 18.0
        assert results[0]["crps"] == 1.1


@pytest.mark.db
def test_run_backtest_integration():
    """Integration test that runs backtest over real/mock database logs without crashing."""
    replay = HistoricalReplayCore()

    results = replay.run_backtest(
        start_date="2025-11-15 00:00:00",
        end_date="2025-11-17 23:59:59",
        stat_categories=["points", "rebounds", "assists"],
    )

    assert isinstance(results, list)
