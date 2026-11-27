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
    assert poller._map_stat_type("3-Pointers Made") == "three_pointers_made"
    assert poller._map_stat_type("3-PT Made") == "three_pointers_made"
    assert poller._map_stat_type("3pt Made") == "three_pointers_made"
    assert poller._map_stat_type("3-Pointers") == "three_pointers_made"
    assert poller._map_stat_type("Steals") == "steals"
    assert poller._map_stat_type("Blocked Shots") == "blocks"
    assert poller._map_stat_type("Invalid") is None

def test_project_player_baseline_fallback():
    poller = UnifiedRestPoller()
    # If history df has less than 5 rows, it should fall back to a simple historical average
    history_df = pd.DataFrame({
        'minutes_played': [20.0, 22.0],
        'points': [10, 15],
        'rebounds': [5, 4],
        'assists': [2, 3],
        'turnovers': [1, 2]
    })
    pred = poller.project_player_baseline(history_df, 'points')
    assert isinstance(pred, float)
    assert abs(pred - 12.5) < 1e-5

def test_project_player_baseline_fit():
    poller = UnifiedRestPoller()
    # 6 rows of history
    history_df = pd.DataFrame({
        'minutes_played': [30.0, 32.0, 28.0, 35.0, 33.0, 31.0],
        'points': [20, 25, 18, 28, 22, 21],
        'rebounds': [5, 6, 4, 7, 5, 5],
        'assists': [5, 4, 6, 5, 4, 5],
        'turnovers': [2, 3, 1, 4, 2, 2]
    })
    pred = poller.project_player_baseline(history_df, 'points')
    assert isinstance(pred, float)
    assert pred > 0.0

def test_diff_engine_integration():
    """
    Tests the diff engine using PrizePicks-style retail prop format.
    Verifies that line movements are detected and the model pipeline runs.
    """
    import asyncio

    async def run_test():
        poller = UnifiedRestPoller()
        from unittest.mock import MagicMock, AsyncMock
        poller.redis = MagicMock()
        poller.redis.json = MagicMock()
        poller.redis.json().get = AsyncMock(return_value=None)
        poller.redis.json().set = AsyncMock()
        poller.redis.get = AsyncMock(return_value=None)
        poller.redis.set = AsyncMock(return_value=True)
        poller.redis.hset = AsyncMock()
        poller.redis.expire = AsyncMock()
        poller.redis.rpush = AsyncMock()
        
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
            'jarrettallen': {
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

        with patch.object(poller, 'get_player_history_async', return_value=history_df) as mock_get_history, \
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
        poller.blender.calculate_blended_probability = lambda sharp_prior_prob, internal_model_prob, market_limit_usd: sharp_prior_prob
        from unittest.mock import MagicMock, AsyncMock
        poller.redis = MagicMock()
        poller.redis.json = MagicMock()
        poller.redis.json().get = AsyncMock(return_value=None)
        poller.redis.json().set = AsyncMock()
        poller.redis.get = AsyncMock(return_value=None)
        poller.redis.set = AsyncMock(return_value=True)
        poller.redis.hset = AsyncMock()
        poller.redis.expire = AsyncMock()
        poller.redis.rpush = AsyncMock()
        
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
            'jarrettallen': {
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

        with patch.object(poller, 'get_player_history_async', return_value=history_df), \
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
            'jarrettallen': {
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

        with patch.object(poller, 'get_player_history_async', return_value=history_df), \
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
        # Under new dynamic liquidity scaling, it should scale stake to 47.50 USD instead of skipping.
        # Over EV = (0.65 / 0.50) - 1 = 30%. With target YES ask liquidity = 50.0,
        # recommended stake = 0.05 * 50000.0 = 2500.00.
        # scaled stake = min(2500.00, 50.0 * 0.95) = 47.50 USD.
        # scaled kelly_fraction = 47.5 / 50000.0 = 0.00095.
        # Since 47.5 >= 5.0, it should fire the alert with the scaled kelly fraction of 0.00095.
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
        poller.anchors = {
            'jarrettallen': {
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

        with patch.object(poller, 'get_player_history_async', return_value=history_df), \
             patch.object(poller.resolver, 'resolve_player', return_value='123'), \
             patch.object(poller, '_fire_alert_and_save') as mock_fire:

            await poller._diff_engine('kalshi', new_state_kalshi)
            mock_fire.assert_called_once()
            called_args = mock_fire.call_args[1]
            assert called_args['book'] == 'kalshi'
            assert abs(called_args['kelly_fraction'] - 0.00095) < 1e-6

        # Scenario 4: Deep tail protection check (implied target cost < 0.15)
        # Target YES ask price = 0.10 (10 cents), true prob = 0.14.
        # Absolute edge = 0.14 - 0.10 = 0.04 (4.0%). This is less than 5.0% absolute edge floor.
        # It should be SKIPPED (not fire), even though relative edge = (0.14/0.10) - 1 = 40% (which is > 5% min_ev_threshold).
        poller.last_state = {
            'kalshi': {
                'prop_2': {
                    'player': 'Jalen Brunson',
                    'prop': 'Points',
                    'line': 3.5,
                    'yes_ask': 0.12,
                    'yes_bid': 0.08,
                    'ask_liquidity_usd': 200.0,
                    'bid_liquidity_usd': 200.0,
                    'odds_type': 'implied_prob'
                }
            }
        }
        new_state_tail = {
            'prop_2': {
                'player': 'Jalen Brunson',
                'prop': 'Points',
                'line': 3.5,
                'yes_ask': 0.10,
                'yes_bid': 0.08,
                'ask_liquidity_usd': 200.0,
                'bid_liquidity_usd': 200.0,
                'odds_type': 'implied_prob'
            }
        }
        poller.anchors = {
            'jalenbrunson': {
                'points': {
                    3.5: {
                        'Pinnacle': {
                            'prob': 0.14,
                            'ask_liquidity_usd': None,
                            'bid_liquidity_usd': None
                        }
                    }
                }
            }
        }

        with patch.object(poller, 'get_player_history_async', return_value=history_df), \
             patch.object(poller.resolver, 'resolve_player', return_value='123'), \
             patch.object(poller, '_fire_alert_and_save') as mock_fire:

            await poller._diff_engine('kalshi', new_state_tail)
            mock_fire.assert_not_called()

        # Reset last_state so that the diff engine detects the tick again
        poller.last_state['kalshi'] = {
            'prop_2': {
                'player': 'Jalen Brunson',
                'prop': 'Points',
                'line': 3.5,
                'yes_ask': 0.12,
                'yes_bid': 0.08,
                'ask_liquidity_usd': 200.0,
                'bid_liquidity_usd': 200.0,
                'odds_type': 'implied_prob'
            }
        }

        # If true prob = 0.16 (16%), absolute edge = 0.16 - 0.10 = 0.06 (6.0% >= 5.0% floor).
        # It should FIRE, reporting the absolute edge of 6.0% (0.06).
        poller.anchors = {
            'jalenbrunson': {
                'points': {
                    3.5: {
                        'Pinnacle': {
                            'prob': 0.16,
                            'ask_liquidity_usd': None,
                            'bid_liquidity_usd': None
                        }
                    }
                }
            }
        }

        with patch.object(poller, 'get_player_history_async', return_value=history_df), \
             patch.object(poller.resolver, 'resolve_player', return_value='123'), \
             patch.object(poller, '_fire_alert_and_save') as mock_fire:

            await poller._diff_engine('kalshi', new_state_tail)
            mock_fire.assert_called_once()
            called_args = mock_fire.call_args[1]
            assert called_args['ev_edge'] == 0.06


    asyncio.run(run_test())


def test_coalesced_alert_consumer():
    """
    Tests that the coalesced alert consumer pulls multiple alerts from the queue,
    consolidates them into a single multi-embed payload, and dispatches them.
    """
    import asyncio
    from unittest.mock import AsyncMock, patch

    async def run_test():
        poller = UnifiedRestPoller()
        
        # Populate the queue with 12 mock alert payloads (should yield 2 batches: 10, then 2)
        payloads = [
            {
                "player": f"Player {i}",
                "prop": "Points",
                "side": "Over",
                "line": 15.5,
                "edge": 0.08,
                "book": "Kalshi",
                "stake": 100.0,
                "sharp_book": "ZINB",
                "sharp_line": "15.5",
                "anchor_liquidity_usd": 250.0,
                "timestamp": "2026-05-31T14:00:00Z"
            }
            for i in range(12)
        ]
        
        # Drain any auto-started task queue/consumer
        while not poller.alert_queue.empty():
            poller.alert_queue.get_nowait()
            poller.alert_queue.task_done()
            
        for p in payloads:
            poller.alert_queue.put_nowait(p)
            
        assert poller.alert_queue.qsize() == 12
        
        dispatched_payloads = []
        def mock_dispatch(url, payload):
            dispatched_payloads.append(payload)
            return 204
            
        poller._dispatch_webhook_sync = mock_dispatch
        
        sleep_count = 0
        original_sleep = asyncio.sleep
        
        async def mock_sleep(seconds):
            nonlocal sleep_count
            sleep_count += 1
            if sleep_count >= 3:
                raise asyncio.CancelledError()
            await original_sleep(0.001)
            
        with patch('asyncio.sleep', mock_sleep):
            try:
                await poller._coalesced_alert_consumer()
            except asyncio.CancelledError:
                pass
                
        assert len(dispatched_payloads) == 2
        assert len(dispatched_payloads[0]["embeds"]) == 10
        assert len(dispatched_payloads[1]["embeds"]) == 2
        
        first_embed = dispatched_payloads[0]["embeds"][0]
        assert first_embed["title"] == "🚨 +EV Edge Detected | KALSHI"
        assert first_embed["description"] == "**Player**: `Player 0`\n**Prop**: `Points`\n**Selection**: `Over 15.5`"
        assert first_embed["fields"][0] == {"name": "Model Edge", "value": "`8.00%`", "inline": True}
        assert first_embed["fields"][1] == {"name": "Recommended Stake", "value": "`$100.00`", "inline": True}
        assert first_embed["fields"][2] == {"name": "Sharp Book (Anchor)", "value": "`ZINB` (Line: `15.5`)", "inline": True}
        assert first_embed["fields"][3] == {"name": "Anchor Liquidity", "value": "`$250.00`", "inline": True}
        
    asyncio.run(run_test())


def test_tail_risk_volatility_scaling():
    """
    Tests the Tail-Risk Volatility Scaler inside _diff_engine.
    Case 1: YES/Over contract with sharp_prob = 0.15 (15%) gets scaled by 0.50x.
    Case 2: NO/Under contract with event_prob (1.0 - sharp_prob) = 0.15 (15%) gets scaled by 0.50x.
    """
    import asyncio

    async def run_test():
        poller = UnifiedRestPoller()
        poller.blender.calculate_blended_probability = lambda sharp_prior_prob, internal_model_prob, market_limit_usd: sharp_prior_prob
        from unittest.mock import MagicMock, AsyncMock
        poller.redis = MagicMock()
        poller.redis.json = MagicMock()
        poller.redis.json().get = AsyncMock(return_value=None)
        poller.redis.json().set = AsyncMock()
        poller.redis.get = AsyncMock(return_value=None)
        poller.redis.set = AsyncMock(return_value=True)
        poller.redis.hset = AsyncMock()
        poller.redis.expire = AsyncMock()
        poller.redis.rpush = AsyncMock()
        
        poller.min_ev_threshold = 1.0

        history_df = pd.DataFrame({
            'minutes_played': [30.0, 32.0, 28.0, 35.0, 33.0, 31.0],
            'points': [20, 25, 18, 28, 22, 21],
            'rebounds': [5, 6, 4, 7, 5, 5],
            'assists': [5, 4, 6, 5, 4, 5],
            'turnovers': [2, 3, 1, 4, 2, 2]
        })

        # --- CASE 1: YES/Over contract tail-risk (sharp_prob = 0.15 < 0.30) ---
        poller.last_state = {
            'kalshi': {
                'prop_yes_tail': {
                    'player': 'Jarrett Allen',
                    'prop': 'Points',
                    'line': 14.5,
                    'yes_ask': 0.10,  # Old state is higher price
                    'yes_bid': 0.06,
                    'ask_liquidity_usd': 1000.0,
                    'bid_liquidity_usd': 1000.0,
                    'odds_type': 'implied_prob'
                }
            }
        }
        new_state_yes = {
            'prop_yes_tail': {
                'player': 'Jarrett Allen',
                'prop': 'Points',
                'line': 14.5,
                'yes_ask': 0.08,  # New state triggers diff (price drop to 0.08)
                'yes_bid': 0.06,
                'ask_liquidity_usd': 1000.0,
                'bid_liquidity_usd': 1000.0,
                'odds_type': 'implied_prob'
            }
        }
        # Set anchor for YES/Over tail event with sharp probability 0.15
        poller.anchors = {
            'jarrettallen': {
                'points': {
                    14.5: {
                        'Pinnacle': {
                            'prob': 0.15,
                            'ask_liquidity_usd': None,
                            'bid_liquidity_usd': None
                        }
                    }
                }
            }
        }

        with patch.object(poller, 'get_player_history_async', return_value=history_df), \
             patch.object(poller.resolver, 'resolve_player', return_value='123'), \
             patch.object(poller, '_fire_alert_and_save') as mock_fire:

            await poller._diff_engine('kalshi', new_state_yes)
            
            mock_fire.assert_called_once()
            called_args = mock_fire.call_args[1]
            
            # Net odds: b = (1.0 / 0.08) - 1.0 = 11.5
            # Raw Kelly YES = (0.15 - 0.85/11.5) * 0.25 = 0.0190217
            # Penalty multiplier = 0.15 / 0.30 = 0.50
            # Expected Kelly fraction = 0.0190217 * 0.50 = 0.0095108
            expected_kelly = ((0.15 - 0.85 / 11.5) * 0.25) * 0.50
            assert abs(called_args['kelly_fraction'] - expected_kelly) < 1e-6

        # --- CASE 2: NO/Under contract tail-risk (event_prob = 1.0 - sharp_prob = 1.0 - 0.85 = 0.15 < 0.30) ---
        poller.last_state = {
            'kalshi': {
                'prop_no_tail': {
                    'player': 'Jarrett Allen',
                    'prop': 'Points',
                    'line': 14.5,
                    'yes_ask': 0.94,
                    'yes_bid': 0.90,  # Old state is lower yes_bid (implied NO is higher: 1.0-0.90=0.10)
                    'ask_liquidity_usd': 1000.0,
                    'bid_liquidity_usd': 1000.0,
                    'odds_type': 'implied_prob'
                }
            }
        }
        new_state_no = {
            'prop_no_tail': {
                'player': 'Jarrett Allen',
                'prop': 'Points',
                'line': 14.5,
                'yes_ask': 0.94,
                'yes_bid': 0.92,  # New state has higher yes_bid (implied NO is 1.0-0.92=0.08, triggers diff)
                'ask_liquidity_usd': 1000.0,
                'bid_liquidity_usd': 1000.0,
                'odds_type': 'implied_prob'
            }
        }
        # Set anchor for YES/Over to 0.85, meaning NO/Under has probability 0.15
        poller.anchors = {
            'jarrettallen': {
                'points': {
                    14.5: {
                        'Pinnacle': {
                            'prob': 0.85,
                            'ask_liquidity_usd': None,
                            'bid_liquidity_usd': None
                        }
                    }
                }
            }
        }

        with patch.object(poller, 'get_player_history_async', return_value=history_df), \
             patch.object(poller.resolver, 'resolve_player', return_value='123'), \
             patch.object(poller, '_fire_alert_and_save') as mock_fire:

            await poller._diff_engine('kalshi', new_state_no)
            
            mock_fire.assert_called_once()
            called_args = mock_fire.call_args[1]
            
            # Net odds: b = (1.0 / 0.08) - 1.0 = 11.5
            # Raw Kelly NO = (0.15 - 0.85/11.5) * 0.25 = 0.0190217
            # Penalty multiplier = 0.15 / 0.30 = 0.50
            # Expected Kelly fraction = 0.0190217 * 0.50 = 0.0095108
            expected_kelly = ((0.15 - 0.85 / 11.5) * 0.25) * 0.50
            assert abs(called_args['kelly_fraction'] - expected_kelly) < 1e-6

    asyncio.run(run_test())


def test_bayesian_market_blender():
    """
    Unit test for BayesianMarketBlender.
    Verifies that the heuristic limits are fetched correctly based on market type,
    and that blended probabilities are computed using the correct Bayesian shrinkage formula.
    """
    from src.models import BayesianMarketBlender

    blender = BayesianMarketBlender(confidence_threshold_usd=2000.0)

    # 1. Test heuristic limits
    assert blender.get_heuristic_limit("Points Spread") == 5000.0
    assert blender.get_heuristic_limit("Player Points") == 350.0
    assert blender.get_heuristic_limit("Player Steals") == 150.0
    assert blender.get_heuristic_limit("Something Else") == 250.0

    # 2. Test blended probability math
    # Confidence threshold = 2000.0, limit = 500.0 => alpha = 500 / 2000 = 0.25
    # sharp_prior = 0.60, internal_model = 0.40
    # Expected blended = 0.25 * 0.60 + 0.75 * 0.40 = 0.15 + 0.30 = 0.45
    p1 = blender.calculate_blended_probability(
        sharp_prior_prob=0.60,
        internal_model_prob=0.40,
        market_limit_usd=500.0
    )
    assert abs(p1 - 0.45) < 1e-6

    # Test clamping when liquidity exceeds threshold (alpha = 1.0)
    p2 = blender.calculate_blended_probability(
        sharp_prior_prob=0.60,
        internal_model_prob=0.40,
        market_limit_usd=3000.0
    )
    assert abs(p2 - 0.60) < 1e-6

    # Test clamping for negative liquidity
    p3 = blender.calculate_blended_probability(
        sharp_prior_prob=0.60,
        internal_model_prob=0.40,
        market_limit_usd=-100.0
    )
    assert abs(p3 - 0.40) < 1e-6


