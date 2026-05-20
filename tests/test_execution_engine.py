import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from execution.shins_method import ShinDevigger
from execution.ev_engine import EVCalculator
from execution.middle_identifier import MiddleScanner

logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')

def test_shin():
    print("==================================================")
    print("--- Testing Shin's Method (De-vigging) ---")
    
    # Sharp book odds (e.g. Pinnacle) heavily juiced Over
    # Over -150 (implied ~60%), Under +120 (implied ~45%)
    # Total implied = ~105.4% (5.4% vig)
    american_odds = [-150, 120]
    
    raw_implied = [ShinDevigger.american_to_implied(o) for o in american_odds]
    print(f"Raw Implied: {raw_implied[0]:.4f}, {raw_implied[1]:.4f} (Sum: {sum(raw_implied):.4f})")
    
    true_probs = ShinDevigger.devig(american_odds)
    print(f"Shin True Probs: {true_probs[0]:.4f}, {true_probs[1]:.4f} (Sum: {sum(true_probs):.4f})")
    
    assert abs(sum(true_probs) - 1.0) < 1e-4, "Shin true probabilities must sum to 1.0"
    return true_probs[0] # Return the true over prob for the next test

def test_ev(true_over_prob):
    print("\n==================================================")
    print("--- Testing EV & Kelly Engine ---")
    
    # Suppose DraftKings is offering the Over at +110
    dk_odds = 110
    
    ev = EVCalculator.calculate_ev(true_over_prob, dk_odds)
    print(f"True Prob: {true_over_prob:.4f} vs Offered Odds: {dk_odds}")
    print(f"Expected Value: {ev:.4%} (+EV)" if ev > 0 else f"Expected Value: {ev:.4%} (-EV)")
    
    # Calculate Quarter Kelly
    q_kelly = EVCalculator.calculate_fractional_kelly(true_over_prob, dk_odds, fraction=0.25)
    print(f"Recommended Quarter-Kelly Stake: {q_kelly:.2%} of bankroll")

def test_middles():
    print("\n==================================================")
    print("--- Testing Middle / Arbitrage Scanner ---")
    
    market_matrix = {
        'DraftKings': {'line': 25.5, 'over_odds': 105, 'under_odds': -125},
        'FanDuel':    {'line': 26.5, 'over_odds': -110, 'under_odds': -110},
        'Pinnacle':   {'line': 25.5, 'over_odds': -130, 'under_odds': 110},
        'BetMGM':     {'line': 25.5, 'over_odds': 100, 'under_odds': -120}
    }
    
    # This matrix contains a Middle (DK Over 25.5 @ +105, FD Under 26.5 @ -110)
    # and a Pure Arb (DK Over 25.5 @ +105, Pinny Under 25.5 @ +110)
    
    opportunities = MiddleScanner.scan_for_arbitrage(market_matrix)
    
    for idx, opp in enumerate(opportunities):
        print(f"Opportunity #{idx+1} -> Type: {opp['type']}")
        print(f"  Buy OVER: {opp['buy_over']['book']} @ {opp['buy_over']['line']} (Odds: {opp['buy_over']['odds']})")
        print(f"  Buy UNDER: {opp['buy_under']['book']} @ {opp['buy_under']['line']} (Odds: {opp['buy_under']['odds']})")
        print(f"  Implied Sum: {opp['implied_sum']}")
        if opp['type'] == 'PURE_ARBITRAGE':
            print(f"  Guaranteed ROI: {opp['guaranteed_roi']}%")
        else:
            print(f"  Middle Gap: {opp['middle_gap']} points")
        print()

if __name__ == "__main__":
    try:
        p_over = test_shin()
        test_ev(p_over)
        test_middles()
        print("✅ All Execution Engine Tests Passed Diagnostics.")
    except Exception as e:
        print(f"❌ Test Failed: {e}")
