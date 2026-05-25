import pytest
import numpy as np
from unittest.mock import MagicMock
from src.slip_builder import CorrelatedSlipEvaluator

def test_determine_relationship_team_key():
    """Verify that team abbreviations are resolved directly from signal dictionaries if present."""
    evaluator = CorrelatedSlipEvaluator()
    
    sig1 = {"player": "LeBron James", "team": "LAL"}
    sig2 = {"player": "Anthony Davis", "team": "LAL"}
    sig3 = {"player": "Nikola Jokic", "Team": "DEN"}
    
    # Same team
    assert evaluator.determine_relationship(sig1, sig2) == "teammate"
    # Different team
    assert evaluator.determine_relationship(sig1, sig3) == "opponent"

def test_determine_relationship_cached_game_logs():
    """Verify that player relationships are resolved via game log overlap when team keys are missing."""
    player_to_games = {
        1: {101, 102, 103, 104},  # Player 1
        2: {101, 102, 103, 105},  # Player 2 (teammate: 3/4 overlap = 0.75)
        3: {201, 202, 203, 204},  # Player 3 (opponent: 0 overlap)
    }
    player_name_to_id = {
        "lebron james": 1,
        "anthony davis": 2,
        "nikola jokic": 3
    }
    
    evaluator = CorrelatedSlipEvaluator(
        player_to_games_cache=player_to_games,
        player_name_to_id_cache=player_name_to_id
    )
    
    # Case 1: Teammates (overlap ratio 0.75 >= 0.5)
    sig1 = {"player": "LeBron James"}
    sig2 = {"player": "Anthony Davis"}
    assert evaluator.determine_relationship(sig1, sig2) == "teammate"
    
    # Case 2: Opponents (overlap ratio 0.0 < 0.5)
    sig3 = {"player": "Nikola Jokic"}
    assert evaluator.determine_relationship(sig1, sig3) == "opponent"

def test_calculate_pit_adjusted_prob_raw_fallback():
    """Verify that PIT fallback uses raw probability when ZINB parameters are missing."""
    evaluator = CorrelatedSlipEvaluator()
    sig = {"player": "LeBron James", "prob": 0.55}
    # No ZINB parameters, should return raw prob (clipped)
    assert evaluator.calculate_pit_adjusted_prob(sig) == 0.55

def test_calculate_pit_adjusted_prob_over_under():
    """Verify that PIT adjusted probability is computed correctly using ZINB parameters."""
    evaluator = CorrelatedSlipEvaluator()
    # Let's seed ZINB parameters for LeBronJames: mu=25.0, pi=0.03, n=15.0, line=24.5
    # For Over: (1.0 - cdf_k) + 0.5 * pmf_k
    # For Under: cdf_k - 0.5 * pmf_k
    sig_over = {
        "player": "LeBron James",
        "side": "Over",
        "line": 24.5,
        "prob": 0.55,
        "zinb_mu": 25.0,
        "zinb_pi": 0.03,
        "zinb_n": 15.0
    }
    
    sig_under = {
        "player": "LeBron James",
        "side": "Under",
        "line": 24.5,
        "prob": 0.45,
        "zinb_mu": 25.0,
        "zinb_pi": 0.03,
        "zinb_n": 15.0
    }
    
    u_pit = evaluator.calculate_pit_adjusted_prob(sig_over)
    v_pit = evaluator.calculate_pit_adjusted_prob(sig_under)
    
    # Check that probabilities are valid
    assert 0.0 < u_pit < 1.0
    assert 0.0 < v_pit < 1.0
    # The sum of Over and Under midpoint PIT should equal 1.0 since:
    # ((1.0 - cdf_k) + 0.5 * pmf_k) + (cdf_k - 0.5 * pmf_k) = 1.0
    assert pytest.approx(u_pit + v_pit) == 1.0

def test_evaluate_slip_math_and_scaling():
    """Verify the copula joint probability, correlation sign inversion, and payout scaling penalty."""
    evaluator = CorrelatedSlipEvaluator()
    
    # 1. Test standard teammate points-assists positive correlation (baseline = 0.15)
    # LeBron: Over 24.5 Points, raw_prob = 0.55
    # Davis: Over 10.5 Rebounds, raw_prob = 0.58
    # Teammate points-rebounds baseline correlation is -0.05
    # If side1 == side2 ("Over" and "Over"), adjusted_rho = baseline_rho = -0.05
    # Since adjusted_rho < 0, joint_prob should be < independent_prob, and NO payout multiplier penalty
    sig1 = {
        "player": "LeBron James",
        "team": "LAL",
        "stat": "points",
        "side": "Over",
        "line": 24.5,
        "prob": 0.55,
        "zinb_mu": 25.0,
        "zinb_pi": 0.03,
        "zinb_n": 15.0
    }
    
    sig2 = {
        "player": "Anthony Davis",
        "team": "LAL",
        "stat": "rebounds",
        "side": "Over",
        "line": 10.5,
        "prob": 0.58,
        "zinb_mu": 11.0,
        "zinb_pi": 0.02,
        "zinb_n": 12.0
    }
    
    res = evaluator.evaluate_slip(sig1, sig2)
    assert res["relationship"] == "teammate"
    assert res["baseline_rho"] == -0.05
    assert res["adjusted_rho"] == -0.05
    assert res["joint_prob"] < res["independent_prob"]
    assert res["payout_multiplier"] == 3.0  # No penalty for negative correlation
    
    # 2. Test high positive correlation teammate points-assists (baseline = 0.15)
    # LeBron: Over Points (0.55)
    # Davis: Over Assists (0.58)
    sig3 = {
        "player": "LeBron James",
        "team": "LAL",
        "stat": "points",
        "side": "Over",
        "line": 24.5,
        "prob": 0.55,
        "zinb_mu": 25.0,
        "zinb_pi": 0.03,
        "zinb_n": 15.0
    }
    sig4 = {
        "player": "Anthony Davis",
        "team": "LAL",
        "stat": "assists",
        "side": "Over",
        "line": 5.5,
        "prob": 0.58,
        "zinb_mu": 6.0,
        "zinb_pi": 0.02,
        "zinb_n": 10.0
    }
    
    res_pos = evaluator.evaluate_slip(sig3, sig4)
    assert res_pos["baseline_rho"] == 0.15
    assert res_pos["adjusted_rho"] == 0.15
    assert res_pos["joint_prob"] > res_pos["independent_prob"]
    # Payout should be penalized down from 3.0x: 3.0 - 2 * 0.15 = 2.7
    assert pytest.approx(res_pos["payout_multiplier"]) == 2.7
    # EV check: (joint_prob * 2.7) - 1.0
    expected_ev = (res_pos["joint_prob"] * 2.7) - 1.0
    assert pytest.approx(res_pos["ev"]) == expected_ev

def test_evaluate_slip_side_sign_inversion():
    """Verify that different sides (Over vs Under) invert the correlation coefficient sign."""
    evaluator = CorrelatedSlipEvaluator()
    
    # Teammate points-assists: baseline = 0.15
    # LeBron: Over Points (0.55)
    # Davis: Under Assists (0.58)
    # Since sides are different (Over vs Under), adjusted_rho should be -0.15
    sig1 = {
        "player": "LeBron James",
        "team": "LAL",
        "stat": "points",
        "side": "Over",
        "line": 24.5,
        "prob": 0.55,
        "zinb_mu": 25.0,
        "zinb_pi": 0.03,
        "zinb_n": 15.0
    }
    sig2 = {
        "player": "Anthony Davis",
        "team": "LAL",
        "stat": "assists",
        "side": "Under",
        "line": 5.5,
        "prob": 0.58,
        "zinb_mu": 6.0,
        "zinb_pi": 0.02,
        "zinb_n": 10.0
    }
    
    res = evaluator.evaluate_slip(sig1, sig2)
    assert res["baseline_rho"] == 0.15
    assert res["adjusted_rho"] == -0.15
    assert res["payout_multiplier"] == 3.0  # No penalty for negative adjusted correlation
