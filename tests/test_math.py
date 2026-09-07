import pytest
import numpy as np
from src.models import MarketMicrostructure, PlayerProjectionSimulator

def test_shins_method_symmetric():
    """Asserts that shins_method(-110, -110) returns exactly 0.5 for both sides."""
    res = MarketMicrostructure.shins_method(-110, -110)
    
    assert "true_prob_over" in res
    assert "true_prob_under" in res
    assert "z_factor" in res
    
    # Tolerances are strict but account for floating point precision
    assert pytest.approx(res["true_prob_over"], abs=1e-6) == 0.5
    assert pytest.approx(res["true_prob_under"], abs=1e-6) == 0.5
    
    # z_factor should represent estimated insider trading volume (z > 0 because of overround)
    assert res["z_factor"] > 0.0
    assert pytest.approx(res["true_prob_over"] + res["true_prob_under"], abs=1e-12) == 1.0

def test_shins_method_asymmetric():
    """Asserts that shins_method(-150, +130) correctly skews true probability toward the favored side."""
    res = MarketMicrostructure.shins_method(-150, +130)
    
    assert pytest.approx(res["true_prob_over"] + res["true_prob_under"], abs=1e-12) == 1.0
    
    # -150 odds are favored, so true probability should be > 0.5
    assert res["true_prob_over"] > 0.5
    assert res["true_prob_under"] < 0.5
    
    # Check that de-vigged probability is less than raw implied probability (strip margin)
    # Raw implied of -150 is 150/250 = 0.60
    assert res["true_prob_over"] < 0.60
    assert res["z_factor"] > 0.0

def test_shins_method_no_vig():
    """Asserts that shins_method with no vig returns original normalized probabilities."""
    res = MarketMicrostructure.shins_method(+100, +100) # Implied = 0.5, 0.5
    assert pytest.approx(res["true_prob_over"], abs=1e-6) == 0.5
    assert pytest.approx(res["true_prob_under"], abs=1e-6) == 0.5
    assert res["z_factor"] == 0.0

def test_probability_calibration():
    """Asserts that calibrate_probabilities via IsotonicRegression works as expected."""
    raw_probs = np.array([0.1, 0.4, 0.35, 0.8, 0.9])
    outcomes = np.array([0, 0, 1, 1, 1])
    
    calibrated = PlayerProjectionSimulator.calibrate_probabilities(raw_probs, outcomes)
    
    assert len(calibrated) == len(raw_probs)
    # Calibrated probabilities should be monotonically increasing relative to sorted raw_probs
    # Let's verify that isotonic constraint holds: if raw_i <= raw_j, then calibrated_i <= calibrated_j
    sorted_indices = np.argsort(raw_probs)
    for idx in range(len(sorted_indices) - 1):
        i = sorted_indices[idx]
        j = sorted_indices[idx + 1]
        assert calibrated[i] <= calibrated[j]

def test_frank_copula():
    """Asserts that for u=0.5, v=0.5, positive dependence (theta=5) yields higher joint probability than independence (theta=0)."""
    # Independent copula (theta=0) yields u * v = 0.25
    prob_ind = MarketMicrostructure.frank_copula_joint_prob(0.5, 0.5, 0.0)
    assert pytest.approx(prob_ind, abs=1e-6) == 0.25
    
    # Positive dependence (theta=5) yields higher joint probability
    prob_dep = MarketMicrostructure.frank_copula_joint_prob(0.5, 0.5, 5.0)
    assert prob_dep > 0.25
    
    # Negative dependence (theta=-5) yields lower joint probability
    prob_neg = MarketMicrostructure.frank_copula_joint_prob(0.5, 0.5, -5.0)
    assert prob_neg < 0.25

def test_garch():
    """Passes high-variance returns and asserts that GARCH conditional volatility is higher than simple standard deviation."""
    # Returns starts low variance then bursts into high variance
    returns = [0.01, -0.01, 0.02, -0.01, 0.01, -0.02] + [0.5, -0.6, 0.7, -0.5, 0.6, -0.6]
    
    # Unconditional simple standard deviation
    simple_std = float(np.std(returns, ddof=1))
    
    # GARCH conditional volatility at the end should capture the high volatility clustering
    # Parameter inputs: omega=0.01, alpha=0.25, beta=0.70
    garch_vol = MarketMicrostructure.garch_volatility(returns, omega=0.01, alpha=0.25, beta=0.70)
    
    assert garch_vol > simple_std

def test_ou_process():
    """Asserts that a current_price of 0.8 smoothly reverts toward a long_term_mean of 0.5 over a large dt."""
    # Small dt -> very little reversion
    price_t_small = MarketMicrostructure.ou_mean_reversion_prob(
        current_price=0.8,
        long_term_mean=0.5,
        reversion_rate=2.0,
        volatility=0.1,
        dt=0.05
    )
    assert price_t_small < 0.8
    assert price_t_small > 0.7
    
    # Large dt -> almost complete reversion to mean
    price_t_large = MarketMicrostructure.ou_mean_reversion_prob(
        current_price=0.8,
        long_term_mean=0.5,
        reversion_rate=2.0,
        volatility=0.1,
        dt=5.0
    )
    assert pytest.approx(price_t_large, abs=1e-4) == 0.5

