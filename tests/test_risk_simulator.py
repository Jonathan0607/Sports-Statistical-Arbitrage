import pytest
import time
from src.risk import simulate_bankroll, american_to_decimal

def test_american_to_decimal():
    # Standard values
    assert american_to_decimal(100) == 2.0
    assert american_to_decimal(200) == 3.0
    assert american_to_decimal(-100) == 2.0
    assert american_to_decimal(-200) == 1.5
    assert american_to_decimal(0) == 1.0

def test_simulate_bankroll_metrics():
    starting_bankroll = 10000.0
    win_rate = 0.55
    average_odds = -110
    bet_size_pct = 0.02
    
    # Run smaller simulation size for unit test speed
    results = simulate_bankroll(
        starting_bankroll=starting_bankroll,
        win_rate=win_rate,
        average_odds=average_odds,
        bet_size_pct=bet_size_pct,
        num_bets=100,
        num_simulations=1000
    )
    
    assert isinstance(results, dict)
    keys = [
        "probability_of_ruin",
        "average_max_drawdown",
        "p99_max_drawdown",
        "median_ending_bankroll",
        "p5_ending_bankroll"
    ]
    for key in keys:
        assert key in results
        assert isinstance(results[key], float)
        
    assert 0.0 <= results["probability_of_ruin"] <= 1.0
    assert 0.0 <= results["average_max_drawdown"] <= 1.0
    assert 0.0 <= results["p99_max_drawdown"] <= 1.0
    assert results["median_ending_bankroll"] > 0.0

def test_simulate_bankroll_extreme_ruin():
    # If win rate is extremely low (e.g. 10%) and bet size is high (e.g. 20%),
    # probability of ruin should be extremely high
    results = simulate_bankroll(
        starting_bankroll=1000.0,
        win_rate=0.10,
        average_odds=-110,
        bet_size_pct=0.20,
        num_bets=200,
        num_simulations=500
    )
    assert results["probability_of_ruin"] > 0.80

def test_simulate_bankroll_latency():
    # Performance benchmark: 100,000 simulations & 1,000 bets should run quickly (in seconds)
    t0 = time.time()
    results = simulate_bankroll(
        starting_bankroll=10000.0,
        win_rate=0.55,
        average_odds=-110,
        bet_size_pct=0.02,
        num_bets=1000,
        num_simulations=100000
    )
    duration = time.time() - t0
    
    print(f"\n[Bench] 100k simulations of 1k bets took: {duration*1000:.2f} ms")
    
    # Assert it takes less than 120.0 seconds (normally takes ~600-900ms, but buffer is added for virtualized/limited environments using float64)
    assert duration < 120.0
