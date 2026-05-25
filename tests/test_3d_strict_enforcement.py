import pytest
import os
import sys
import json
from unittest.mock import MagicMock, patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scrapers.unified_api import SharpLineProvider
from infrastructure.entity_resolver import clean_player_name
from scrapers.exchanges import KalshiClobClient, PolymarketClobClient
from scrapers.hfp_poller import UnifiedRestPoller

def test_clean_player_name_normalization():
    """Verify player names are correctly cleaned (accents stripped, lowercase, suffixes removed)."""
    assert clean_player_name("Luka Dončić") == "luka doncic"
    assert clean_player_name("Nikola Jokić") == "nikola jokic"
    assert clean_player_name("LeBron James Jr.") == "lebron james"
    assert clean_player_name("Jaren Jackson III") == "jaren jackson"
    assert clean_player_name("Kenyon Martin Jr") == "kenyon martin"
    assert clean_player_name("Robert Williams III") == "robert williams"
    assert clean_player_name("") == ""
    assert clean_player_name(None) == ""

def test_json_deserialization_trap():
    """Verify that JSON serialization float-key string conversion is reversed in unified_api."""
    provider = SharpLineProvider()
    raw_serialized_cache = {
        "lebron james": {
            "points": {
                "25.5": {
                    "pinnacle_true_prob": 0.542,
                    "pinnacle_line": 25.5,
                    "dk_line": 24.5
                }
            }
        }
    }
    deserialized = provider._deserialize_lines(raw_serialized_cache)
    
    # Assert third-dimension key is cast to float
    assert 25.5 in deserialized["lebron james"]["points"]
    assert "25.5" not in deserialized["lebron james"]["points"]
    assert deserialized["lebron james"]["points"][25.5]["pinnacle_true_prob"] == 0.542

def test_half_point_hook_translation():
    """Verify that X+ and X or more contract formats are translated to X - 0.5."""
    kalshi = KalshiClobClient()
    polymarket = PolymarketClobClient()

    # Kalshi style colon formats
    p1, s1, l1 = kalshi.parse_nba_contract("LeBron James: 20+ Points")
    assert p1 == "lebron james"
    assert s1 == "Points"
    assert l1 == 19.5

    p2, s2, l2 = kalshi.parse_nba_contract("Nikola Jokic: 10 or more Rebounds")
    assert p2 == "nikola jokic"
    assert s2 == "Rebounds"
    assert l2 == 9.5

    # Polymarket style direct questions
    p3, s3, l3 = polymarket.parse_nba_contract("Will LeBron James record 20+ points?")
    assert p3 == "lebron james"
    assert s3 == "Points"
    assert l3 == 19.5

    p4, s4, l4 = polymarket.parse_nba_contract("Will Nikola Jokic have 10 or more rebounds?")
    assert p4 == "nikola jokic"
    assert s4 == "Rebounds"
    assert l4 == 9.5

def test_strict_line_enforcement():
    """Verify that exact line matching skips props without matching lines in anchors."""
    import asyncio

    async def run():
        poller = UnifiedRestPoller()
        
        # Mock anchors: Jarrett Allen has a point line at 15.5 (prob=0.52)
        poller.anchors = {
            "jarrett allen": {
                "points": {
                    15.5: {"Pinnacle": {"prob": 0.52, "ask_liquidity_usd": None, "bid_liquidity_usd": None}}
                }
            }
        }

        # Seed state with PrizePicks-format props
        old_state_template = {
            'prop_1': {
                'player': 'Jarrett Allen',
                'prop': 'Points',
                'line': 10.5,  # different from any new states
                'odds_type': 'standard'
            }
        }

        # 1. Retail prop matching exactly (15.5) -> should be evaluated
        matching_state = {
            "prop_1": {
                "player": "Jarrett Allen",
                "prop": "Points",
                "line": 15.5,
                "odds_type": "standard"
            }
        }

        # 2. Retail prop slightly off (15.505) -> within 0.01 tolerance, should match
        matching_state_tolerance = {
            "prop_1": {
                "player": "Jarrett Allen",
                "prop": "Points",
                "line": 15.505,
                "odds_type": "standard"
            }
        }

        # 3. Retail prop mismatched (14.5) -> should be skipped
        mismatched_state = {
            "prop_1": {
                "player": "Jarrett Allen",
                "prop": "Points",
                "line": 14.5,
                "odds_type": "standard"
            }
        }

        with patch.object(poller, '_fire_alert_and_save') as mock_fire:
            # We don't care about DB logs in this test, mock them or let them skip
            poller.resolver.resolve_player = MagicMock(return_value="123")
            poller.get_player_history = MagicMock()
            zinb_mock = MagicMock()
            zinb_mock.predict_over_probability.return_value = 0.52
            zinb_mock.mu = 15.5
            zinb_mock.pi = 0.03
            zinb_mock.n = 15.0
            poller.zinb_cache = {("123", "points"): (zinb_mock, {"xMin": 25.0, "xUSG": 0.20})}

            # For mismatched_state, ZINB fallback is triggered, so it calls resolve_player
            poller.last_state = {"pp": dict(old_state_template)}
            poller.resolver.resolve_player.reset_mock()
            await poller._diff_engine("pp", mismatched_state)
            poller.resolver.resolve_player.assert_called_once()

            # For matching_state, it must call resolve_player
            poller.last_state = {"pp": dict(old_state_template)}
            poller.resolver.resolve_player.reset_mock()
            await poller._diff_engine("pp", matching_state)
            poller.resolver.resolve_player.assert_called_once()

            # For matching_state_tolerance, it must call resolve_player
            poller.last_state = {"pp": dict(old_state_template)}
            poller.resolver.resolve_player.reset_mock()
            await poller._diff_engine("pp", matching_state_tolerance)
            poller.resolver.resolve_player.assert_called_once()

    asyncio.run(run())

def test_graceful_degradation_missing_key():
    """Verify that provider does not crash and returns empty data on missing API key."""
    # Temporarily remove API key from env
    with patch.dict(os.environ, {}, clear=True):
        # We need to recreate SharpLineProvider to trigger env reload
        provider = SharpLineProvider()
        # Force setting api_key to empty to simulate missing
        provider.api_key = ""
        
        with patch.object(provider, 'load_cache', return_value=None):
            # Test force_sync_seed() returns empty dict
            seed_result = provider.force_sync_seed()
            assert seed_result == {}

            # Test get_sharp_lines() returns empty dict
            import asyncio
            async def check():
                lines = await provider.get_sharp_lines()
                assert lines == {}
            asyncio.run(check())
