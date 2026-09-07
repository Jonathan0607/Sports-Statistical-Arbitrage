import pytest
import time
import numpy as np
from src.models import MarketMicrostructure
from src import quant_core

def test_shins_method_accuracy():
    # Test case 1: Implied probabilities representing standard NBA lines
    # Odds: -110, -110 -> 0.523809, 0.523809
    odds_over = -110
    odds_under = -110
    
    # Implied probs
    def to_implied(odds):
        if odds > 0:
            return 100.0 / (odds + 100.0)
        else:
            return abs(odds) / (abs(odds) + 100.0)
            
    pi_over = to_implied(odds_over)
    pi_under = to_implied(odds_under)
    
    # Python version
    py_results = MarketMicrostructure.shins_method(odds_over, odds_under)
    
    # C++ version accepts implied probs or decimal odds
    # Implied probs
    cpp_results_probs = quant_core.calculate_shins_probs([pi_over, pi_under])
    # Decimal odds (e.g. 1.909, 1.909)
    dec_over = 1.0 / pi_over
    dec_under = 1.0 / pi_under
    cpp_results_odds = quant_core.calculate_shins_probs([dec_over, dec_under])
    
    # Verify outputs match
    assert abs(cpp_results_probs[0] - py_results["true_prob_over"]) < 1e-6
    assert abs(cpp_results_probs[1] - py_results["true_prob_under"]) < 1e-6
    
    assert abs(cpp_results_odds[0] - py_results["true_prob_over"]) < 1e-6
    assert abs(cpp_results_odds[1] - py_results["true_prob_under"]) < 1e-6

def test_shins_method_multi_way():
    # Multi-way market test: 3 outcomes (e.g. Soccer 1X2)
    # Probs: 0.45, 0.30, 0.35 (sum = 1.10)
    pi = [0.45, 0.30, 0.35]
    cpp_results = quant_core.calculate_shins_probs(pi)
    
    assert len(cpp_results) == 3
    assert abs(sum(cpp_results) - 1.0) < 1e-7

def test_shins_method_invalid_inputs():
    with pytest.raises(ValueError):
        # Empty input
        quant_core.calculate_shins_probs([])
        
    with pytest.raises(ValueError):
        # Negative value
        quant_core.calculate_shins_probs([1.5, -1.1])

def test_shins_method_performance_benchmark():
    # Generate random test cases
    num_runs = 50000
    
    # Pre-generate random odds values
    odds_over_list = np.random.randint(-150, -105, size=num_runs)
    odds_under_list = np.random.randint(-150, -105, size=num_runs)
    
    def to_implied(odds):
        if odds > 0:
            return 100.0 / (odds + 100.0)
        else:
            return abs(odds) / (abs(odds) + 100.0)
            
    # Pre-generate inputs for C++
    inputs_list = [[to_implied(o1), to_implied(o2)] for o1, o2 in zip(odds_over_list, odds_under_list)]

    # 1. Benchmark Python Implementation
    t0_py = time.time()
    for o1, o2 in zip(odds_over_list, odds_under_list):
        _ = MarketMicrostructure.shins_method(int(o1), int(o2))
    duration_py = time.time() - t0_py

    # 2. Benchmark C++ Native Implementation
    t0_cpp = time.time()
    for inputs in inputs_list:
        _ = quant_core.calculate_shins_probs(inputs)
    duration_cpp = time.time() - t0_cpp

    py_avg_us = (duration_py / num_runs) * 1e6
    cpp_avg_us = (duration_cpp / num_runs) * 1e6
    speedup = duration_py / max(1e-9, duration_cpp)

    print("\n" + "=" * 60)
    print(" SHIN'S METHOD PERFORMANCE BENCHMARK")
    print("=" * 60)
    print(f"Total Iterations:      {num_runs:,}")
    print(f"Python Solver Total:   {duration_py:.4f} s  ({py_avg_us:.2f} μs / call)")
    print(f"C++ Bare-Metal Total:  {duration_cpp:.4f} s  ({cpp_avg_us:.2f} μs / call)")
    print(f"C++ Speedup Factor:    {speedup:.2f}x faster")
    print("=" * 60 + "\n")
    
    # Assert C++ is substantially faster (at least 5x speedup)
    assert speedup > 5.0
