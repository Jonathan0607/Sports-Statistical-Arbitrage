import os
import sys
import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import ContextualFeatureEngine

@pytest.fixture
def mock_engine():
    with patch("sqlalchemy.create_engine") as mock_create:
        mock_engine_inst = MagicMock()
        mock_create.return_value = mock_engine_inst
        yield mock_engine_inst

def test_build_player_feature_matrix_pit_pandas_filtering(mock_engine):
    # Instantiate engine with mocked _load_player_info to avoid db call on init
    with patch.object(ContextualFeatureEngine, "_load_player_info", return_value=None):
        fe = ContextualFeatureEngine()
        fe.player_teams = {"player_1": 1}
        fe.player_positions = {"player_1": "SG"}

    # Mock DB query for bulk matchup metrics & teammate sidelined to return empty/mocked results
    # so we can test the target_date filtering logic on history_df.
    fe.get_co_stars = MagicMock(return_value=[])
    
    # We mock connection execute for team_matchup_metrics queries
    mock_conn = MagicMock()
    fe.engine.connect.return_value.__enter__.return_value = mock_conn
    mock_conn.execute.return_value.fetchall.return_value = []

    # Create dummy history_df
    history_df = pd.DataFrame([
        {"game_id": 101, "start_time": "2026-01-01 10:00:00", "home_team_id": 1, "away_team_id": 2, "minutes_played": 30.0},
        {"game_id": 102, "start_time": "2026-01-05 10:00:00", "home_team_id": 1, "away_team_id": 3, "minutes_played": 31.0},
        {"game_id": 103, "start_time": "2026-01-10 10:00:00", "home_team_id": 1, "away_team_id": 4, "minutes_played": 32.0},
    ])

    # Case 1: No target_date (entire history remains)
    res_df = fe.build_player_feature_matrix("player_1", history_df)
    assert len(res_df) == 3
    assert list(res_df["game_id"]) == [101, 102, 103]

    # Case 2: target_date = "2026-01-08" (only first two games remain)
    res_df_pit = fe.build_player_feature_matrix("player_1", history_df, target_date="2026-01-08")
    assert len(res_df_pit) == 2
    assert list(res_df_pit["game_id"]) == [101, 102]

    # Case 3: target_date = "2026-01-02" (only first game remains)
    res_df_pit2 = fe.build_player_feature_matrix("player_1", history_df, target_date="2026-01-02")
    assert len(res_df_pit2) == 1
    assert list(res_df_pit2["game_id"]) == [101]


def test_build_player_feature_matrix_pit_sql_fallback(mock_engine):
    with patch.object(ContextualFeatureEngine, "_load_player_info", return_value=None):
        fe = ContextualFeatureEngine()
        fe.player_teams = {"player_1": 1}
        fe.player_positions = {"player_1": "SG"}

    fe.get_co_stars = MagicMock(return_value=[])

    # We mock pd.read_sql to return dummy data representing games fetched from DB
    mock_sql_data = pd.DataFrame([
        {"game_id": 101, "start_time": "2026-01-01 10:00:00", "home_team_id": 1, "away_team_id": 2, "minutes_played": 30.0},
        {"game_id": 102, "start_time": "2026-01-05 10:00:00", "home_team_id": 1, "away_team_id": 3, "minutes_played": 31.0},
    ])

    with patch("pandas.read_sql") as mock_read_sql:
        mock_read_sql.return_value = mock_sql_data
        
        # When history_df is empty, it queries the database
        res_df = fe.build_player_feature_matrix("player_1", pd.DataFrame(), target_date="2026-01-08")
        
        # Check that pandas.read_sql was called
        assert mock_read_sql.called
        # Check the SQL params passed to read_sql
        args, kwargs = mock_read_sql.call_args
        params = kwargs.get("params", {})
        assert params["player_id"] == "player_1"
        assert params["target_date"] == "2026-01-08"

        # Check target_date filter condition was in the SQL query text
        query_text = str(args[0])
        assert ":target_date IS NULL OR gm.start_time < :target_date" in query_text


def test_build_active_features_pit(mock_engine):
    with patch.object(ContextualFeatureEngine, "_load_player_info", return_value=None):
        fe = ContextualFeatureEngine()
        fe.player_teams = {"player_1": 1}
        fe.player_positions = {"player_1": "SG"}

    fe.get_co_stars = MagicMock(return_value=[])

    mock_conn = MagicMock()
    fe.engine.connect.return_value.__enter__.return_value = mock_conn

    # Setup mock return values for:
    # 1. conn.execute(query_game) -> returns one upcoming game row: (game_id, home_team_id, away_team_id, start_time)
    # 2. conn.execute(query_last_game) -> returns last game start time: (start_time,)
    # 3. conn.execute(query_metrics) -> opponent matchup metrics: (rolling_def, rolling_pace, allowed)
    
    mock_game_row = (201, 1, 5, "2026-01-15 19:00:00")
    mock_last_game_row = ("2026-01-12 19:00:00",)
    mock_metrics_row = (110.0, 98.0, 20.0)
    mock_own_metrics_row = (115.0, 100.0, 22.0)

    mock_conn.execute.return_value.fetchone.side_effect = [
        mock_game_row,         # query_game
        mock_last_game_row,    # query_last_game
        mock_metrics_row,      # query_metrics (opponent)
        mock_own_metrics_row,  # query_metrics (own team for pace differential)
    ]
    
    mock_conn.execute.return_value.fetchall.return_value = []

    # Test execution with target_date
    features = fe.build_active_features("player_1", target_date="2026-01-14")

    # Assert correct parameters were passed to the queries
    calls = mock_conn.execute.call_args_list
    
    # 1st call (upcoming game query)
    query_game_text = str(calls[0][0][0])
    query_game_params = calls[0][0][1]
    assert ":target_date IS NOT NULL AND start_time >= :target_date" in query_game_text
    assert query_game_params["target_date"] == "2026-01-14"

    # 2nd call (last game query)
    query_last_game_text = str(calls[1][0][0])
    query_last_game_params = calls[1][0][1]
    assert ":target_date IS NULL OR gm.start_time < :target_date" in query_last_game_text
    assert query_last_game_params["target_date"] == "2026-01-14"

    # 3rd call (opponent metrics query)
    query_metrics_text = str(calls[2][0][0])
    query_metrics_params = calls[2][0][1]
    assert "g.start_time < :game_time" in query_metrics_text
    assert query_metrics_params["game_time"] == "2026-01-15 19:00:00"

    # Check calculated features
    assert features["days_rest"] == 3.0  # (Jan 15 - Jan 12) = 3 days
    assert features["is_back_to_back"] == 0.0
    assert features["opp_def_rating"] == 110.0
    assert features["opp_pace_factor"] == 98.0
    assert features["opp_pos_pts_allowed"] == 20.0
    assert features["pace_differential"] == 100.0 - 98.0


def test_season_boundary_reset(mock_engine):
    """Verify that crossing season boundaries resets days_rest to 3.0 and partitions EWMA features."""
    with patch.object(ContextualFeatureEngine, "_load_player_info", return_value=None):
        fe = ContextualFeatureEngine()
        fe.player_teams = {"player_1": 1}
        fe.player_positions = {"player_1": "SG"}

    fe.get_co_stars = MagicMock(return_value=[])

    # Player history spanning two distinct seasons: 2023-24 (April) and 2024-25 (October)
    multi_season_df = pd.DataFrame([
        {
            "game_id": 1,
            "start_time": "2024-04-14 15:30:00",
            "season": "2023-24",
            "home_team_id": 1,
            "away_team_id": 2,
            "minutes_played": 36.0,
            "points": 30,
            "assists": 10,
            "rebounds": 8,
            "potential_assists": 18.0,
            "touches": 85.0,
            "avg_defender_distance_ft": 4.2
        },
        {
            "game_id": 2,
            "start_time": "2024-10-22 19:30:00",  # Season opener of 2024-25 (6 months later)
            "season": "2024-25",
            "home_team_id": 1,
            "away_team_id": 3,
            "minutes_played": 32.0,
            "points": 24,
            "assists": 5,
            "rebounds": 6,
            "potential_assists": 10.0,
            "touches": 60.0,
            "avg_defender_distance_ft": 4.8
        },
        {
            "game_id": 3,
            "start_time": "2024-10-24 19:30:00",  # Game 2 of 2024-25 (2 days rest)
            "season": "2024-25",
            "home_team_id": 1,
            "away_team_id": 4,
            "minutes_played": 34.0,
            "points": 28,
            "assists": 7,
            "rebounds": 5,
            "potential_assists": 12.0,
            "touches": 65.0,
            "avg_defender_distance_ft": 4.5
        }
    ])

    res = fe.build_player_feature_matrix("player_1", multi_season_df)

    # 1. Season boundary reset on days_rest:
    # Game 1 (2023-24 opener): 3.0
    # Game 2 (2024-25 opener): must be 3.0 (NOT ~190 days from April!)
    # Game 3 (2024-25 game 2): 2.0 days
    assert res.iloc[0]["days_rest"] == 3.0
    assert res.iloc[1]["days_rest"] == 3.0
    assert res.iloc[2]["days_rest"] == 2.0

    # 2. EWMA rolling tracking features should reset at start of 2024-25 season:
    # For game 2 (opener of 2024-25), rolling tracking should be 0.0 or default, NOT the April 2024 values
    assert res.iloc[1]["roll_potential_assists"] == 0.0
    assert res.iloc[1]["roll_touches"] == 0.0
    assert res.iloc[1]["roll_defender_distance"] == 4.5

