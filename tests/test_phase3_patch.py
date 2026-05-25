import numpy as np
import pandas as pd
import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import PlayerProjectionModel, ProbabilityCalibrator, ZINBModel, EVCalculator

logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')

def test_xgboost():
    print("==================================================")
    print("--- Testing XGBoost PlayerProjectionModel ---")
    
    # Synthetic dataset
    # Features: [historical_pts, opp_def_rating, starting_center_out (0 or 1)]
    X_train = np.array([
        [15.0, 110.0, 0],
        [16.0, 112.0, 0],
        [14.5, 108.0, 0],
        [15.5, 115.0, 1], # Big spike in minutes/usage expected
        [15.0, 111.0, 1]
    ])
    y_min = np.array([28.0, 30.0, 26.0, 36.0, 34.0])
    y_usg = np.array([0.18, 0.19, 0.17, 0.25, 0.24])
    
    model = PlayerProjectionModel()
    model.fit(X_train, y_min, y_usg)
    
    # Test prediction on a game where the starting center is OUT (flag=1)
    X_eval = np.array([[15.0, 112.0, 1]])
    pred = model.predict(X_eval)
    print(f"Projected Baseline -> Minutes: {pred['xMin']:.2f}, Usage: {pred['xUSG']:.2f}")
    assert pred['xMin'] > 30.0, "XGBoost failed to isolate the categorical bump in minutes."

def test_isotonic_regression():
    print("\n==================================================")
    print("--- Testing ProbabilityCalibrator (Isotonic) ---")
    
    # Synthetic raw ZINB outputs vs True Hit Rates
    raw_probs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    # Assume the model is overconfident in the middle ranges
    true_outcomes = np.array([0, 0, 0, 1, 0, 1, 1, 1, 1])
    
    calibrator = ProbabilityCalibrator()
    calibrator.fit(raw_probs, true_outcomes)
    
    raw_50 = 0.50
    calibrated_50 = calibrator.calibrate(raw_50)
    print(f"Raw Prob: {raw_50:.2f} -> Calibrated Prob: {calibrated_50:.4f}")
    assert 0.0 <= calibrated_50 <= 1.0, f"Calibrated probability {calibrated_50} is out of bounds"

def test_zinb_conditioning():
    print("\n==================================================")
    print("--- Testing ZINB Categorical Conditioning ---")
    
    np.random.seed(42)
    # Generate some standard data
    counts = np.random.negative_binomial(n=10, p=0.33, size=100) # mean roughly 20
    
    model = ZINBModel()
    model.fit(pd.Series(counts))
    initial_mu = model.mu
    initial_pi = model.pi
    
    # Condition: Huge blowout risk
    model.condition_parameters(blowout_risk=0.90, pace_factor=1.0)
    print(f"Initial Pi: {initial_pi:.4f} -> Conditioned Pi: {model.pi:.4f}")
    assert model.pi > initial_pi, "ZINB failed to widen zero-inflation on blowout risk."

def test_dfs_ev():
    print("\n==================================================")
    print("--- Testing DFS Fixed Payout EV Calculator ---")
    
    # Suppose we find two great edges (60% true probability each)
    leg_probs = [0.60, 0.60]
    
    # Evaluate against PrizePicks 2-Pick Power (3X payout)
    ev = EVCalculator.calculate_dfs_edge(leg_probs, platform='PrizePicks_2_Power')
    print(f"Legs: {leg_probs}")
    print(f"Platform: PrizePicks 2-Pick Power Play (3X)")
    print(f"Expected Value: {ev:.4%} (+EV)" if ev > 0 else f"Expected Value: {ev:.4%} (-EV)")
    assert ev > 0, "DFS EV Calculator failed to flag a +EV spot."

if __name__ == "__main__":
    import pandas as pd
    try:
        test_xgboost()
        test_isotonic_regression()
        test_zinb_conditioning()
        test_dfs_ev()
        print("\n✅ Phase 3 Mathematical Patch Diagnostics Passed.")
    except Exception as e:
        print(f"\n❌ Test Failed: {e}")
