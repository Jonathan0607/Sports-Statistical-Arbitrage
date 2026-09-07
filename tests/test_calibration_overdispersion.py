import os
import sys
import pytest
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import ZINBModel


def test_zinb_dynamic_overdispersion():
    """Test ZINBModel condition_parameters for spread and pace adjustments."""
    model = ZINBModel()
    
    # Fit dummy data to get baseline parameters
    data = [10, 12, 8, 15, 11, 9, 14, 10, 13, 11]
    model.fit(data)
    
    pi_baseline = model.pi
    n_baseline = model.n

    # High blowout risk: abs(spread) >= 12.0
    model.condition_parameters(spread=14.0, pace_factor=100.0)
    assert model.pi == min(pi_baseline + 0.05, 0.99)
    assert abs(model.n - n_baseline * 0.9) < 1e-4

    # Reset and test high pace: pace_factor > 102.0
    model.fit(data)
    model.condition_parameters(spread=0.0, pace_factor=105.0)
    assert abs(model.n - n_baseline * 0.85) < 1e-4

    # Reset and test slow pace: pace_factor < 96.0
    model.fit(data)
    model.condition_parameters(spread=0.0, pace_factor=90.0)
    assert abs(model.n - n_baseline * 1.15) < 1e-4
