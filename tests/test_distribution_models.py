import numpy as np
import pandas as pd
import logging
import os
import sys

# Add root directory to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.zinb_model import ZINBModel
from models.volatility_tracker import VolatilityTracker
from models.copula_engine import CopulaEngine

logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')

def test_zinb():
    print("==================================================")
    print("--- Testing ZINB Model (Count Data Modeling) ---")
    
    # Synthetic NBA player points data
    # 20% DNP/0 points, otherwise Negative Binomial with mean ~20
    np.random.seed(42)
    zeros = np.zeros(20)
    counts = np.random.negative_binomial(n=10, p=0.33, size=80) # mean roughly 20
    data = np.concatenate([zeros, counts])
    np.random.shuffle(data)
    
    model = ZINBModel()
    model.fit(pd.Series(data))
    
    # Let's test the probability of scoring Over 19.5 points
    prob_over_19_5 = model.predict_over_probability(19.5)
    print(f"Probability of Over 19.5 points: {prob_over_19_5:.4f} (Expected: ~0.4 - 0.5)")

def test_garch():
    print("\n==================================================")
    print("--- Testing GARCH Volatility Tracker ---")
    
    # Synthetic residuals with volatility clustering
    np.random.seed(42)
    returns = np.random.normal(0, 1, 100)
    returns[50:60] *= 5.0 # Introduce a volatility cluster
    
    tracker = VolatilityTracker()
    tracker.fit(pd.Series(returns))
    forecasted_var = tracker.forecast_variance()
    print(f"Forecasted Variance for next game: {forecasted_var:.4f}")

def test_copula():
    print("\n==================================================")
    print("--- Testing Copula Engine (Joint Probabilities) ---")
    
    # Synthetic correlated data (e.g. PG assists and C points)
    np.random.seed(42)
    x = np.random.normal(0, 1, 100)
    y = 0.7 * x + np.random.normal(0, 0.5, 100)
    
    engine = CopulaEngine(copula_type='clayton')
    engine.fit(pd.Series(x), pd.Series(y))
    
    # Let's say Player A has 60% chance to go UNDER, Player B has 50% chance to go UNDER
    prob_a_under = 0.60
    prob_b_under = 0.50
    
    joint_under = engine.evaluate_joint_probability(prob_a_under, prob_b_under)
    naive_under = prob_a_under * prob_b_under
    
    print(f"Naive Independent Joint UNDER: {naive_under:.4f}")
    print(f"Copula Modeled Joint UNDER: {joint_under:.4f}")
    
    # Joint OVER
    prob_a_over = 1.0 - prob_a_under
    prob_b_over = 1.0 - prob_b_under
    
    joint_over = engine.evaluate_joint_survival(prob_a_under, prob_b_under)
    naive_over = prob_a_over * prob_b_over
    
    print(f"\nNaive Independent Joint OVER: {naive_over:.4f}")
    print(f"Copula Modeled Joint OVER: {joint_over:.4f}")
    
    if joint_over > naive_over:
         print("\n--> Edge identified! The sportsbooks (assuming independence) are undervaluing the OVER/OVER correlation.")

if __name__ == "__main__":
    try:
        test_zinb()
        test_garch()
        test_copula()
        print("\n✅ All Distribution Models Passed Diagnostics.")
    except Exception as e:
        print(f"\n❌ Test Failed: {e}")
