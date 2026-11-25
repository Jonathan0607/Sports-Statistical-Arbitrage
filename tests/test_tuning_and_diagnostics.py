import os
import sys
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import ContextualFeatureEngine
from src.tune_models import tune_xgboost_hyperparameters, load_feature_matrix_for_tuning
from src.historical_replay_core import HistoricalReplayCore


def test_vacated_usage_elasticity_calculation():
    """Verify that vacated_usage correctly sums usage rates of sidelined co-stars."""
    with patch.object(ContextualFeatureEngine, "_load_player_info", return_value=None):
        fe = ContextualFeatureEngine()
        fe.player_teams = {"star_p1": 1, "co_star_a": 1, "co_star_b": 1}
        fe.player_positions = {"star_p1": "PG", "co_star_a": "SG", "co_star_b": "SF"}

    # Mock co-stars returning player_id and avg_usg
    mock_co_stars = [
        {"player_id": "co_star_a", "avg_usg": 0.28},
        {"player_id": "co_star_b", "avg_usg": 0.22},
    ]
    fe.get_co_stars = MagicMock(return_value=mock_co_stars)

    # Historical game log
    history_df = pd.DataFrame([
        {"game_id": 101, "start_time": "2025-11-01 19:00:00", "home_team_id": 1, "away_team_id": 2, "minutes_played": 34.0, "points": 25, "rebounds": 5, "assists": 6, "turnovers": 2, "steals": 1, "blocks": 0, "three_pointers_made": 2, "usage_rate": 0.25},
    ])

    # Game 101: co_star_a recorded 0.0 mins (DNP -> pre-game ruled-out proxy), co_star_b played 32.0 mins (active)
    # Expected vacated_usage = 0.28 + 0.0 = 0.28
    mock_played_logs = [
        (101, "co_star_a", 0.0),
        (101, "co_star_b", 32.0),
    ]

    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchall.return_value = mock_played_logs

    with patch.object(fe.engine, "connect") as mock_engine_conn:
        mock_engine_conn.return_value.__enter__.return_value = mock_conn
        res_df = fe.build_player_feature_matrix("star_p1", history_df)

    assert "vacated_usage" in res_df.columns
    assert abs(res_df.iloc[0]["vacated_usage"] - 0.28) < 1e-4


def test_optuna_tuning_execution():
    """Verify Optuna study creates trials and finds optimal hyperparameter dictionary."""
    study = tune_xgboost_hyperparameters(target_key="pts", n_trials=3)
    assert study is not None
    assert len(study.trials) == 3
    assert "max_depth" in study.best_params
    assert "learning_rate" in study.best_params
    assert "subsample" in study.best_params
    assert "colsample_bytree" in study.best_params
    assert "reg_alpha" in study.best_params
    assert "reg_lambda" in study.best_params


def test_residual_diagnostics_generation(tmp_path):
    """Verify that run_backtest creates diagnostic_residuals.png and computes metrics."""
    replay = HistoricalReplayCore()

    mock_logs = pd.DataFrame([
        {"game_id": 1, "player_id": "p1", "player_name": "Player One", "start_time": "2025-10-25 19:00:00", "points": 25, "rebounds": 6, "assists": 5},
        {"game_id": 2, "player_id": "p2", "player_name": "Player Two", "start_time": "2025-11-15 19:30:00", "points": 18, "rebounds": 8, "assists": 3},
        {"game_id": 3, "player_id": "p3", "player_name": "Player Three", "start_time": "2025-12-05 20:00:00", "points": 30, "rebounds": 10, "assists": 7},
    ])

    def mock_eval(player_id, player_name, stat_category, game_time, actual_outcome_value):
        return {
            "player_id": player_id,
            "player_name": player_name,
            "stat_category": "Points",
            "game_time": game_time,
            "actual_value": actual_outcome_value,
            "meta_prediction": actual_outcome_value + 1.2,
            "xgboost_prediction": actual_outcome_value + 1.2,
            "ewma_baseline": actual_outcome_value + 2.5,
            "crps": 0.95,
        }

    with patch.object(replay.engine, "connect"), \
         patch("pandas.read_sql", return_value=mock_logs), \
         patch.object(replay, "evaluate_prediction_accuracy", side_effect=mock_eval):

        results = replay.run_backtest(
            start_date="2025-10-01 00:00:00",
            end_date="2025-12-31 23:59:59",
            stat_categories=["points"],
        )

        assert len(results) == 3
        assert os.path.exists("logs/diagnostic_residuals.png")


def test_stat_specific_regularization_search_space():
    """Verify that reb and ast use aggressive regularization search spaces compared to pts."""
    from src.tune_models import get_stat_specific_search_space
    import optuna

    study_reb = optuna.create_study()
    trial_reb = study_reb.ask()
    params_reb = get_stat_specific_search_space("reb", trial_reb)

    # Rebounds should have restricted max_depth in [1, 3] and elevated penalties
    assert 1 <= params_reb["max_depth"] <= 3
    assert 5.0 <= params_reb["reg_alpha"] <= 100.0
    assert 10.0 <= params_reb["reg_lambda"] <= 200.0
    assert 0.005 <= params_reb["learning_rate"] <= 0.05

    study_pts = optuna.create_study()
    trial_pts = study_pts.ask()
    params_pts = get_stat_specific_search_space("pts", trial_pts)

    assert 1 <= params_pts["max_depth"] <= 3
    assert 5.0 <= params_pts["reg_alpha"] <= 100.0
    assert 15.0 <= params_pts["reg_lambda"] <= 250.0

    study_gen = optuna.create_study()
    trial_gen = study_gen.ask()
    params_gen = get_stat_specific_search_space("generic", trial_gen)

    assert 3 <= params_gen["max_depth"] <= 6
    assert 0.1 <= params_gen["reg_alpha"] <= 5.0
    assert 0.5 <= params_gen["reg_lambda"] <= 10.0


def test_load_feature_matrix_train_seasons_filter():
    """Verify load_feature_matrix_for_tuning applies train_seasons filter."""
    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchall.return_value = [("p1",)]
    
    mock_df = pd.DataFrame({
        "game_id": [1, 2, 3, 4, 5, 6, 7],
        "season": ["2023-24", "2023-24", "2023-24", "2023-24", "2023-24", "2024-25", "2024-25"],
        "start_time": pd.date_range("2023-10-01", periods=7, freq="D"),
        "minutes_played": [30.0]*7,
        "points": [20.0]*7,
        "rebounds": [5.0]*7,
        "assists": [5.0]*7,
        "turnovers": [2.0]*7,
        "potential_assists": [6.0]*7,
        "touches": [40.0]*7,
        "avg_defender_distance_ft": [4.0]*7,
        "days_rest": [1.0]*7,
        "is_back_to_back": [0.0]*7,
        "opp_def_rating": [110.0]*7,
        "opp_pace_factor": [100.0]*7,
        "opp_pos_pts_allowed": [22.0]*7,
        "vacated_usage": [0.0]*7,
        "passes_made": [25.0]*7,
        "time_of_possession_sec": [100.0]*7,
        "is_home_game": [1.0]*7,
        "pace_differential": [0.0]*7,
    })

    with patch("src.tune_models.sqlalchemy.create_engine") as mock_engine, \
         patch("src.tune_models.ContextualFeatureEngine") as mock_fe_class:
         
        mock_engine.return_value.connect.return_value.__enter__.return_value = mock_conn
        
        mock_fe = MagicMock()
        mock_fe.build_player_feature_matrix.return_value = mock_df
        mock_fe_class.return_value = mock_fe
        
        X, y_residual, roll_stat, y_actual = load_feature_matrix_for_tuning(
            target_key="pts", return_full=True, train_seasons=["2023-24"]
        )
        
        # 5 rows for 2023-24. 1 row lost to shift(1). ewma. Should be 4 rows.
        assert len(X) == 4

