import pytest
import numpy as np
from src.risk import RiskManager
from src.risk import Backtester

def test_multivariate_kelly():
    """Asserts that highly correlated positive-EV bets scale down their individual Kelly fractions to avoid over-exposure."""
    # Imagine two bets with positive EV edge
    edge_vector = np.array([0.05, 0.05])
    
    # 1. Independent case: diagonal covariance matrix
    cov_independent = np.array([
        [0.25, 0.0],
        [0.0, 0.25]
    ])
    f_independent = RiskManager.multivariate_kelly(edge_vector, cov_independent, max_bankroll_fraction=0.5)
    
    # 2. Correlated case: high positive covariance (correlation = 0.8)
    cov_correlated = np.array([
        [0.25, 0.20],
        [0.20, 0.25]
    ])
    f_correlated = RiskManager.multivariate_kelly(edge_vector, cov_correlated, max_bankroll_fraction=0.5)
    
    # Check that allocations are positive
    assert np.all(f_independent > 0.0)
    assert np.all(f_correlated > 0.0)
    
    # Highly correlated bets should scale down the individual Kelly sizes to avoid joint risk
    assert f_correlated[0] < f_independent[0]
    assert f_correlated[1] < f_independent[1]
    
    # Assert that the exposure cap is respected (e.g. max_bankroll_fraction = 0.05)
    f_capped = RiskManager.multivariate_kelly(edge_vector, cov_independent, max_bankroll_fraction=0.05)
    assert np.sum(f_capped) <= 0.05 + 1e-12

def test_mask_bet_size():
    """Asserts that bet sizes round to recreational increments and respect floor."""
    # Under $5 floor checks
    assert RiskManager.mask_bet_size(3.10) == 5
    assert RiskManager.mask_bet_size(4.99) == 5
    assert RiskManager.mask_bet_size(0.0) == 5
    
    # Recreational rounding checks (nearest $5 increment)
    assert RiskManager.mask_bet_size(27.42) in (25, 30) # Nearest is 25
    assert RiskManager.mask_bet_size(27.42) == 25
    assert RiskManager.mask_bet_size(28.75) == 30
    assert RiskManager.mask_bet_size(52.51) == 55 # 10.502 rounds to 11
    assert RiskManager.mask_bet_size(52.49) == 50 # 10.498 rounds to 10

def test_calculate_clv():
    """Asserts that placing a bet at 50% implied probability that closes at 55% yields +10% CLV."""
    clv = Backtester.calculate_clv(placed_implied_prob=0.50, closing_implied_prob=0.55)
    assert pytest.approx(clv, abs=1e-6) == 0.10 # +10% CLV
    
    # Check zero values safeguard
    assert Backtester.calculate_clv(0.0, 0.55) == 0.0
