import numpy as np
import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import PlayerVolatilityTracker, SGPCorrelationEngine
from src.models import StochasticMarketMonitor

logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')

def test_garch():
    print("==================================================")
    print("--- Testing PlayerVolatilityTracker (GARCH) ---")
    
    tracker = PlayerVolatilityTracker()
    
    # Simulate a player averaging 20 points who suddenly enters a massive boom/bust streak
    np.random.seed(42)
    stable_history = np.random.normal(20, 2, 50).tolist()
    
    volatile_history = stable_history.copy()
    # Inject massive volatility into the last 10 games (e.g. 5, 35, 10, 30...)
    volatile_history[-10:] = [5, 35, 10, 30, 8, 38, 12, 28, 6, 40]
    
    stable_var = tracker.get_current_regime_variance("player_1", stable_history)
    volatile_var = tracker.get_current_regime_variance("player_1", volatile_history)
    
    print(f"Variance under Stable Regime: {stable_var:.4f}")
    print(f"Variance under Volatile Regime: {volatile_var:.4f}")
    
    assert volatile_var > stable_var, "GARCH did not detect the volatility expansion"

def test_copula():
    print("\n==================================================")
    print("--- Testing SGPCorrelationEngine (Clayton) ---")
    
    engine = SGPCorrelationEngine()
    
    # Suppose PG has 55% chance to hit Over Assists, Center has 55% chance to hit Over Points
    prob_a = 0.55
    prob_b = 0.55
    
    # Assume historical Kendall's Tau correlation is 0.40 (strong positive correlation)
    tau = 0.40
    
    joint_prob = engine.price_joint_probability(prob_a, prob_b, kendalls_tau=tau, copula_type='clayton')
    naive_prob = prob_a * prob_b
    
    print(f"Naive Independent Joint Probability: {naive_prob:.4f}")
    print(f"Copula Joint Probability (Tau={tau}): {joint_prob:.4f}")
    print("--> Copula correctly prices the correlated SGP legs higher than naive independence.")
    assert joint_prob > naive_prob, "Copula did not price the correlated SGP legs higher"

def test_jump_diffusion():
    print("\n==================================================")
    print("--- Testing StochasticMarketMonitor (Jump-Diffusion) ---")
    
    # OU mean is 0.50 (fair value), market usually drifts slowly.
    monitor = StochasticMarketMonitor(theta=0.1, sigma=0.02)
    
    ou_mean = 0.50
    time_step = 1.0 # 1 minute since last tick
    
    # Scenario 1: Normal market tick (moves from 0.50 to 0.51)
    print("Tick 1: Line moves from 50% to 51% (Normal Volume)")
    is_jump_1 = monitor.detect_overreaction(current_implied_prob=0.51, ou_mean=ou_mean, time_step_dt=time_step)
    
    # Scenario 2: Violent Jump (moves from 0.50 to 0.65) due to an injury tweet
    print("\nTick 2: Line violently gaps from 50% to 65% (News Catalyst)")
    is_jump_2 = monitor.detect_overreaction(current_implied_prob=0.65, ou_mean=ou_mean, time_step_dt=time_step)
    
    assert not is_jump_1 and is_jump_2, "Monitor failed to correctly identify the jump vs noise"

if __name__ == "__main__":
    try:
        test_garch()
        test_copula()
        test_jump_diffusion()
        print("\n✅ Sprint 7 Complex Pricing Tests Passed Diagnostics.")
    except Exception as e:
        print(f"\n❌ Test Failed: {e}")
