import pytest
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from dashboard import (
    generate_mock_iv_surface,
    calculate_atm_greeks,
    normal_cdf
)

def test_normal_cdf():
    assert abs(normal_cdf(0.0) - 0.5) < 1e-6
    assert normal_cdf(10.0) > 0.9999
    assert normal_cdf(-10.0) < 0.0001

def test_mock_iv_surface_dimensions():
    base_iv = 0.45
    skew = 0.35
    base_line = 22.5
    
    strikes, times, Z = generate_mock_iv_surface(base_iv, skew, base_line)
    
    # Assert dimension lengths
    assert len(strikes) == 21
    assert len(times) == 13
    assert Z.shape == (13, 21)
    
    # Assert boundaries and limits
    # Z should be >= 0.10 always
    assert np.all(Z >= 0.10)
    
    # Strike boundaries check
    assert strikes[0] == base_line - 10.0
    assert strikes[-1] == base_line + 10.0
    
    # Time boundaries check
    assert times[0] == 0.0
    assert times[-1] == 120.0

def test_atm_greeks_math():
    base_line = 22.5
    time_minutes = 60.0
    iv = 0.45
    
    delta, vega = calculate_atm_greeks(base_line, time_minutes, iv)
    
    assert isinstance(delta, float)
    assert isinstance(vega, float)
    
    # Delta should be between 0.5 and 1.0 because d1 = 0.5 * sigma * sqrt(t) > 0.0, so N(d1) > 0.5
    assert 0.5 < delta < 1.0
    assert vega > 0.0
