import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, AsyncMock, patch

from src.config import LEAGUE_CONFIG, get_league_config
from src.models import BetMaskingEngine
from src.risk import RiskManager
from src.features import ContextualFeatureEngine
from src.execution import SlipQueueManager
from scrapers.hfp_poller import UnifiedRestPoller
import src.simulation_engine as sim_engine

def test_league_config_specs():
    assert "NBA" in LEAGUE_CONFIG
    assert "WNBA" in LEAGUE_CONFIG
    
    nba = LEAGUE_CONFIG["NBA"]
    assert nba["nba_api_id"] == "00"
    assert nba["regulation_minutes"] == 48.0
    assert nba["max_retail_bet_limit"] == 100.00
    assert nba["min_anchor_liquidity"] == 100.00
    assert nba["prizepicks_league_name"] == "NBA"
    assert "underdog_league_name" not in nba

    wnba = LEAGUE_CONFIG["WNBA"]
    assert wnba["nba_api_id"] == "10"
    assert wnba["regulation_minutes"] == 40.0
    assert wnba["max_retail_bet_limit"] == 25.00
    assert wnba["min_anchor_liquidity"] == 25.00
    assert wnba["prizepicks_league_name"] == "WNBA"
    assert "underdog_league_name" not in wnba

    assert get_league_config("nba")["regulation_minutes"] == 48.0
    assert get_league_config("wnba")["regulation_minutes"] == 40.0


def test_dynamic_bet_limits_masking():
    # NBA raw stake $150 should be capped at $100
    nba_stake = BetMaskingEngine.mask_kelly_stake(150.0, league="NBA")
    assert float(nba_stake) <= 100.00

    # WNBA raw stake $150 or $50 should be aggressively capped at $25
    wnba_stake = BetMaskingEngine.mask_kelly_stake(150.0, league="WNBA")
    assert float(wnba_stake) <= 25.00

    wnba_stake_50 = BetMaskingEngine.mask_kelly_stake(50.0, league="WNBA")
    assert float(wnba_stake_50) <= 25.00

    # RiskManager mask_bet_size
    assert RiskManager.mask_bet_size(80.0, league="WNBA") <= 25.00
    assert RiskManager.mask_bet_size(80.0, league="NBA") <= 100.00


def test_slip_queue_manager_wnba_cap():
    leg1 = {"player": "A'ja Wilson", "stat": "Points", "line": 22.5, "side": "Over", "edge": 0.05, "prob": 0.60}
    leg2 = {"player": "Breanna Stewart", "stat": "Rebounds", "line": 9.5, "side": "Over", "edge": 0.06, "prob": 0.58}
    qm = SlipQueueManager()
    payload = qm.formulate_execution_payload(leg1, leg2, risk_adjusted_stake=50.00, ev_edge=0.12, league="WNBA")
    assert payload["league"] == "WNBA"
    assert payload["risk_adjusted_stake"] <= 25.00


def test_contextual_feature_scaling():
    fe_nba = ContextualFeatureEngine(league="NBA")
    assert fe_nba.regulation_minutes == 48.0

    fe_wnba = ContextualFeatureEngine(league="WNBA")
    assert fe_wnba.regulation_minutes == 40.0

    df_sample = pd.DataFrame([{
        "game_id": 1,
        "start_time": pd.to_datetime("2024-10-22 19:30:00", utc=True),
        "season": "2024-25",
        "home_team_id": 1,
        "away_team_id": 2,
        "minutes_played": 20.0,
        "points": 10,
        "rebounds": 5,
        "assists": 4,
        "field_goals_attempted": 8,
        "free_throws_attempted": 2,
        "usage_rate": 0.22
    }])

    with patch.object(fe_nba, "_load_player_info", return_value=None), \
         patch.object(fe_nba, "get_co_stars", return_value=[]):
        res_nba = fe_nba.build_player_feature_matrix("p1", df_sample.copy())
        # reb_rate = (5 / 20) * 48 = 12.0
        assert abs(res_nba["reb_rate"].iloc[0] - 12.0) < 1e-4

    with patch.object(fe_wnba, "_load_player_info", return_value=None), \
         patch.object(fe_wnba, "get_co_stars", return_value=[]):
        res_wnba = fe_wnba.build_player_feature_matrix("p1", df_sample.copy())
        # reb_rate = (5 / 20) * 40 = 10.0
        assert abs(res_wnba["reb_rate"].iloc[0] - 10.0) < 1e-4


def test_simulation_engine_regulation_minutes():
    match_ctx = {
        "home_team": "NYL",
        "away_team": "LVA",
        "home_ortg": 110.0,
        "away_ortg": 108.0,
        "pace": 98.0
    }
    players = [
        {
            "player_id": "p1",
            "team": "home",
            "is_starter": True,
            "baseline_minutes": 30.0,
            "baseline_pts": 18.0,
            "efficiencies": {"pts": {"lambda": 20.0, "alpha": 0.15}}
        }
    ]
    # WNBA simulation uses 40 minutes regulation
    sim_obj = sim_engine.EndogenousGameSimulator(n_iterations=500)
    sim_res = sim_obj.simulate_game(match_ctx, players, league="WNBA")
    assert isinstance(sim_res, pd.DataFrame)
    assert len(sim_res) == 1
    assert "expected_minutes" in sim_res.columns



def test_hfp_poller_prizepicks_multi_league():
    poller = UnifiedRestPoller(leagues=["NBA", "WNBA"])
    sample_pp_data = {
        "data": [
            {
                "id": "prop_wnba_1",
                "attributes": {
                    "stat_display_name": "Points",
                    "line_score": 19.5,
                    "odds_type": "standard",
                    "league": "WNBA"
                },
                "relationships": {
                    "new_player": {"data": {"id": "player_w1", "type": "new_player"}}
                }
            }
        ],
        "included": [
            {
                "type": "new_player",
                "id": "player_w1",
                "attributes": {
                    "display_name": "Caitlin Clark",
                    "league": "WNBA"
                }
            }
        ]
    }
    parsed_wnba = poller._parse_pp(sample_pp_data, league="WNBA")
    assert len(parsed_wnba) == 1
    assert parsed_wnba["prop_wnba_1"]["player"] == "caitlinclark"

    parsed_nba = poller._parse_pp(sample_pp_data, league="NBA")
    assert len(parsed_nba) == 0
