import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from src.simple_baseline_evaluator import SimpleBaselineEvaluator


def test_evaluator_init():
    evaluator = SimpleBaselineEvaluator(alpha=0.3, league_pace=99.0, league_def_rating=115.0)
    assert evaluator.alpha == 0.3
    assert evaluator.league_pace == 99.0
    assert evaluator.league_def_rating == 115.0


def test_evaluate_point_in_time_insufficient_history():
    evaluator = SimpleBaselineEvaluator()
    # 4 games only (< 5)
    mock_history = pd.DataFrame([
        {"game_id": 1, "start_time": pd.to_datetime("2026-03-01", utc=True), "points": 20, "rebounds": 5, "assists": 4},
        {"game_id": 2, "start_time": pd.to_datetime("2026-03-03", utc=True), "points": 22, "rebounds": 6, "assists": 5},
        {"game_id": 3, "start_time": pd.to_datetime("2026-03-05", utc=True), "points": 18, "rebounds": 4, "assists": 3},
        {"game_id": 4, "start_time": pd.to_datetime("2026-03-07", utc=True), "points": 25, "rebounds": 7, "assists": 6},
    ])
    evaluator.player_history = {"player_1": mock_history}

    res = evaluator.evaluate_point_in_time(
        player_id="player_1",
        game_id=5,
        game_time=pd.to_datetime("2026-03-09", utc=True),
        stat_col="points",
        actual_value=30.0,
        player_team_id=1,
        home_team_id=1,
        away_team_id=2,
    )
    assert res is None


def test_evaluate_point_in_time_calculations():
    evaluator = SimpleBaselineEvaluator(alpha=0.3, league_pace=99.0, league_def_rating=115.0)
    
    # 6 historical games
    mock_history = pd.DataFrame([
        {"game_id": 1, "start_time": pd.to_datetime("2026-03-01", utc=True), "points": 10.0, "rebounds": 5.0, "assists": 2.0},
        {"game_id": 2, "start_time": pd.to_datetime("2026-03-03", utc=True), "points": 20.0, "rebounds": 6.0, "assists": 4.0},
        {"game_id": 3, "start_time": pd.to_datetime("2026-03-05", utc=True), "points": 15.0, "rebounds": 7.0, "assists": 3.0},
        {"game_id": 4, "start_time": pd.to_datetime("2026-03-07", utc=True), "points": 25.0, "rebounds": 8.0, "assists": 5.0},
        {"game_id": 5, "start_time": pd.to_datetime("2026-03-09", utc=True), "points": 30.0, "rebounds": 9.0, "assists": 6.0},
        {"game_id": 6, "start_time": pd.to_datetime("2026-03-11", utc=True), "points": 35.0, "rebounds": 10.0, "assists": 7.0},
    ])
    evaluator.player_history = {"player_1": mock_history}
    
    # Opponent matchup: away_team_id = 2, game_id = 7
    # Opponent pace = 108.9 (1.10x), Opponent Def Rating = 126.5 (1.10x)
    evaluator.matchup_map = {
        (2, 7): (126.5, 108.9)
    }

    game_time = pd.to_datetime("2026-03-13", utc=True)
    res = evaluator.evaluate_point_in_time(
        player_id="player_1",
        game_id=7,
        game_time=game_time,
        stat_col="points",
        actual_value=28.0,
        player_team_id=1,
        home_team_id=1,
        away_team_id=2,
    )

    assert res is not None
    assert res["actual_value"] == 28.0

    # 1. 5-Game SMA of last 5 games (games 2,3,4,5,6 -> [20, 15, 25, 30, 35]):
    # mean = 125 / 5 = 25.0
    expected_sma = (20.0 + 15.0 + 25.0 + 30.0 + 35.0) / 5.0
    assert abs(res["naive_sma"] - expected_sma) < 1e-4

    # 2. EWMA (alpha=0.3)
    expected_ewma = mock_history["points"].ewm(alpha=0.3, adjust=True).mean().iloc[-1]
    assert abs(res["ewma_raw"] - expected_ewma) < 1e-4

    # 3. Context Adjusted EWMA:
    # pace_scaler = 108.9 / 99.0 = 1.10
    # def_scaler = 126.5 / 115.0 = 1.10
    expected_context = expected_ewma * 1.10 * 1.10
    assert abs(res["ewma_context"] - expected_context) < 1e-4


def test_run_evaluation_mock():
    evaluator = SimpleBaselineEvaluator()

    mock_logs = pd.DataFrame([
        {"game_id": 1, "player_id": "p1", "player_name": "Player 1", "current_team_id": 1, "home_team_id": 1, "away_team_id": 2, "start_time": pd.to_datetime("2026-03-01", utc=True), "points": 20, "rebounds": 5, "assists": 4},
        {"game_id": 2, "player_id": "p1", "player_name": "Player 1", "current_team_id": 1, "home_team_id": 1, "away_team_id": 2, "start_time": pd.to_datetime("2026-03-03", utc=True), "points": 22, "rebounds": 6, "assists": 5},
        {"game_id": 3, "player_id": "p1", "player_name": "Player 1", "current_team_id": 1, "home_team_id": 1, "away_team_id": 2, "start_time": pd.to_datetime("2026-03-05", utc=True), "points": 18, "rebounds": 4, "assists": 3},
        {"game_id": 4, "player_id": "p1", "player_name": "Player 1", "current_team_id": 1, "home_team_id": 1, "away_team_id": 2, "start_time": pd.to_datetime("2026-03-07", utc=True), "points": 25, "rebounds": 7, "assists": 6},
        {"game_id": 5, "player_id": "p1", "player_name": "Player 1", "current_team_id": 1, "home_team_id": 1, "away_team_id": 2, "start_time": pd.to_datetime("2026-03-09", utc=True), "points": 30, "rebounds": 8, "assists": 7},
        {"game_id": 6, "player_id": "p1", "player_name": "Player 1", "current_team_id": 1, "home_team_id": 1, "away_team_id": 2, "start_time": pd.to_datetime("2026-03-11", utc=True), "points": 28, "rebounds": 6, "assists": 5},
    ])
    evaluator.logs_df = mock_logs
    evaluator.player_history = {"p1": mock_logs}
    evaluator.matchup_map = {}

    df_res = evaluator.run_evaluation(
        start_date="2026-03-10 00:00:00",
        end_date="2026-03-12 23:59:59",
        stat_categories=["points", "rebounds", "assists"],
    )

    assert len(df_res) == 3
    assert set(df_res["stat_category"]) == {"points", "rebounds", "assists"}
    assert "naive_sma" in df_res.columns
    assert "ewma_raw" in df_res.columns
    assert "ewma_context" in df_res.columns
