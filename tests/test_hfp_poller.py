import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scrapers.hfp_poller import UnifiedRestPoller

def test_map_stat_type():
    poller = UnifiedRestPoller()
    assert poller._map_stat_type("Points") == "points"
    assert poller._map_stat_type("Rebounds") == "rebounds"
    assert poller._map_stat_type("Assists") == "assists"
    assert poller._map_stat_type("Turnovers") == "turnovers"
    assert poller._map_stat_type("Invalid") is None

def test_project_player_baseline_fallback():
    poller = UnifiedRestPoller()
    # If history df has less than 5 rows, it should fall back to baseline defaults
    history_df = pd.DataFrame({
        'minutes_played': [20.0, 22.0],
        'points': [10, 15],
        'rebounds': [5, 4],
        'assists': [2, 3],
        'turnovers': [1, 2]
    })
    pred = poller.project_player_baseline(history_df, 'points')
    assert pred['xMin'] == 25.0
    assert pred['xUSG'] == 0.20

def test_project_player_baseline_fit():
    poller = UnifiedRestPoller()
    # 6 rows of history to allow training (needs >= 5, yields train_df >= 2)
    history_df = pd.DataFrame({
        'minutes_played': [30.0, 32.0, 28.0, 35.0, 33.0, 31.0],
        'points': [20, 25, 18, 28, 22, 21],
        'rebounds': [5, 6, 4, 7, 5, 5],
        'assists': [5, 4, 6, 5, 4, 5],
        'turnovers': [2, 3, 1, 4, 2, 2]
    })
    pred = poller.project_player_baseline(history_df, 'points')
    assert isinstance(pred, dict)
    assert 'xMin' in pred
    assert 'xUSG' in pred
    assert isinstance(pred['xMin'], float)
    assert isinstance(pred['xUSG'], float)

def test_diff_engine_integration():
    """
    Tests the diff engine using PrizePicks-style retail prop format.
    Verifies that line movements are detected and the model pipeline runs.
    """
    import asyncio

    async def run_test():
        poller = UnifiedRestPoller()
        poller.min_ev_threshold = 1.0

        # Mock database call to return a known history
        history_df = pd.DataFrame({
            'minutes_played': [30.0, 32.0, 28.0, 35.0, 33.0, 31.0],
            'points': [20, 25, 18, 28, 22, 21],
            'rebounds': [5, 6, 4, 7, 5, 5],
            'assists': [5, 4, 6, 5, 4, 5],
            'turnovers': [2, 3, 1, 4, 2, 2]
        })

        # Seed state with PrizePicks-format props
        old_state = {
            'pp': {
                'prop_1': {
                    'player': 'Jarrett Allen',
                    'prop': 'Points',
                    'line': 15.5,
                    'odds_type': 'standard'
                }
            }
        }
        poller.last_state = old_state
        
        # Seed anchors to allow exact line matching (14.5) to pass
        poller.anchors = {
            'jarrett allen': {
                'points': {
                    14.5: {'Pinnacle': {'prob': 0.52, 'ask_liquidity_usd': None, 'bid_liquidity_usd': None}}
                }
            }
        }

        # New state: line movement (15.5 → 14.5)
        new_state = {
            'prop_1': {
                'player': 'Jarrett Allen',
                'prop': 'Points',
                'line': 14.5,
                'odds_type': 'standard'
            }
        }

        with patch.object(poller, 'get_player_history', return_value=history_df) as mock_get_history, \
             patch.object(poller.resolver, 'resolve_player', return_value='1628983') as mock_resolve, \
             patch.object(poller, '_fire_alert_and_save') as mock_fire:

            ticks = await poller._diff_engine('pp', new_state)

            assert len(ticks) == 1
            assert ticks[0]['player'] == 'Jarrett Allen'
            mock_get_history.assert_called_once()
            mock_resolve.assert_called_once_with('PrizePicks', 'Jarrett Allen')

    asyncio.run(run_test())


def test_liquidity_validation_and_fallback():
    """
    Tests the independent Bid/Ask liquidity check and fallback cascade in _diff_engine.
    Scenario 1: Exchange is sharp anchor but has thin liquidity -> fallback to Pinnacle.
    Scenario 2: Exchange is sharp anchor, passes min(bid, ask) >= 100 -> used as anchor.
    Scenario 3: Execution target is exchange, YES side has ask_liquidity < 100 -> skip YES execution.
    """
    import asyncio

    async def run_test():
        poller = UnifiedRestPoller()
        poller.min_ev_threshold = 1.0

        history_df = pd.DataFrame({
            'minutes_played': [30.0, 32.0, 28.0, 35.0, 33.0, 31.0],
            'points': [20, 25, 18, 28, 22, 21],
            'rebounds': [5, 6, 4, 7, 5, 5],
            'assists': [5, 4, 6, 5, 4, 5],
            'turnovers': [2, 3, 1, 4, 2, 2]
        })

        # Set up old and new state on PrizePicks
        poller.last_state = {
            'pp': {
                'prop_1': {
                    'player': 'Jarrett Allen',
                    'prop': 'Points',
                    'line': 15.5,
                    'odds_type': 'standard'
                }
            }
        }
        new_state = {
            'prop_1': {
                'player': 'Jarrett Allen',
                'prop': 'Points',
                'line': 14.5,
                'odds_type': 'standard'
            }
        }

        # Scenario 1: Kalshi is thin (< 100 USD liquidity). It should fallback to Pinnacle.
        poller.anchors = {
            'jarrett allen': {
                'points': {
                    14.5: {
                        'Kalshi': {
                            'prob': 0.60,
                            'ask_liquidity_usd': 50.0,  # thin
                            'bid_liquidity_usd': 150.0
                        },
                        'Pinnacle': {
                            'prob': 0.52,
                            'ask_liquidity_usd': None,
                            'bid_liquidity_usd': None
                        }
                    }
                }
            }
        }

        with patch.object(poller, 'get_player_history', return_value=history_df), \
             patch.object(poller.resolver, 'resolve_player', return_value='123'), \
             patch.object(poller, '_fire_alert_and_save') as mock_fire:

            await poller._diff_engine('pp', new_state)
            
            # Since Kalshi is thin, it falls back to Pinnacle (prob=0.52).
            # Over EV = (0.52 / 0.5434) - 1.0 = -4.3% (< 1% threshold). Under EV = (0.48 / 0.5434) - 1.0 = -11.6%.
            # Neither exceeds min_ev_threshold = 1.0. So no alert is fired.
            mock_fire.assert_not_called()

        # Scenario 2: Kalshi has sufficient liquidity (>= 100 USD liquidity on both sides).
        # Should use Kalshi (prob=0.60).
        # Over EV = (0.60 / 0.5434) - 1.0 = 10.4% (> 1% threshold). Alert should fire with Kalshi.
        poller.last_state = {
            'pp': {
                'prop_1': {
                    'player': 'Jarrett Allen',
                    'prop': 'Points',
                    'line': 15.5,
                    'odds_type': 'standard'
                }
            }
        }
        poller.anchors = {
            'jarrett allen': {
                'points': {
                    14.5: {
                        'Kalshi': {
                            'prob': 0.60,
                            'ask_liquidity_usd': 120.0,  # thick
                            'bid_liquidity_usd': 150.0   # thick
                        },
                        'Pinnacle': {
                            'prob': 0.52,
                            'ask_liquidity_usd': None,
                            'bid_liquidity_usd': None
                        }
                    }
                }
            }
        }

        with patch.object(poller, 'get_player_history', return_value=history_df), \
             patch.object(poller.resolver, 'resolve_player', return_value='123'), \
             patch.object(poller, '_fire_alert_and_save') as mock_fire:

            await poller._diff_engine('pp', new_state)
            mock_fire.assert_called_once()
            # Verify it used Kalshi as the anchor
            called_args = mock_fire.call_args[1]
            assert called_args['sharp_book'] == 'Kalshi'
            assert called_args['sharp_prob'] == 0.60
            assert called_args['anchor_liquidity_usd'] == 120.0

        # Scenario 3: Execution target is Kalshi, and YES side has thin liquidity (< 100).
        # Over/YES execution should be skipped.
        poller.last_state = {
            'kalshi': {
                'prop_1': {
                    'player': 'Jarrett Allen',
                    'prop': 'Points',
                    'line': 15.5,
                    'yes_ask': 0.54,
                    'yes_bid': 0.50,
                    'ask_liquidity_usd': 50.0,   # thin execution side
                    'bid_liquidity_usd': 150.0,  # thick execution side
                    'odds_type': 'implied_prob'
                }
            }
        }
        new_state_kalshi = {
            'prop_1': {
                'player': 'Jarrett Allen',
                'prop': 'Points',
                'line': 14.5,
                'yes_ask': 0.50,
                'yes_bid': 0.48,
                'ask_liquidity_usd': 50.0,   # thin execution side
                'bid_liquidity_usd': 150.0,  # thick execution side
                'odds_type': 'implied_prob'
            }
        }
        # Pinnacle anchor has prob 0.65.
        # For YES: EV = (0.65 / 0.50) - 1 = 30%. But target YES ask liquidity is 50.0 (< 100), so YES is skipped.
        # For NO: target is 1 - yes_bid = 0.52. EV = (0.35 / 0.52) - 1 = -32.7%.
        # So no alert should fire.
        poller.anchors = {
            'jarrett allen': {
                'points': {
                    14.5: {
                        'Pinnacle': {
                            'prob': 0.65,
                            'ask_liquidity_usd': None,
                            'bid_liquidity_usd': None
                        }
                    }
                }
            }
        }

        with patch.object(poller, 'get_player_history', return_value=history_df), \
             patch.object(poller.resolver, 'resolve_player', return_value='123'), \
             patch.object(poller, '_fire_alert_and_save') as mock_fire:

            await poller._diff_engine('kalshi', new_state_kalshi)
            mock_fire.assert_not_called()

    asyncio.run(run_test())
