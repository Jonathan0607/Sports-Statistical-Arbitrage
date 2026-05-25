import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from the newly consolidated src/ module
from src.models import PlayerProjectionModel, ZINBModel, ShinDevigger, EVCalculator

class BetMaskingEngine:
    @staticmethod
    def mask_bet_size(raw_kelly_stake_pct, bankroll=1000):
        """
        Rounds a mathematically exact Kelly stake (e.g., $44.32) to a 
        'recreational' number (e.g., $45.00) to prevent algorithmic flagging.
        """
        raw_dollar_amount = raw_kelly_stake_pct * bankroll
        # Round to nearest 5 dollars
        masked_amount = round(raw_dollar_amount / 5) * 5
        return float(masked_amount)

def run_e2e_simulation():
    print("\n=================================================================")
    print("         INITIATING STAT-ARB PIPELINE: END-TO-END SIMULATION         ")
    print("=================================================================\n")
    
    # ---------------------------------------------------------
    # STEP 1: INGESTION & CATALYST
    # ---------------------------------------------------------
    print("[STEP 1] Ingestion & Catalyst")
    mock_tweet = "Dallas PR: Luka Doncic is OUT for tonight's game."
    print(f"  > Incoming WebSocket Stream: '{mock_tweet}'")
    
    # Mocking the LLM parser output
    parsed_news = {"player": "Luka Doncic", "status": "OUT"}
    time.sleep(0.5)
    print(f"  > FastNewsParser (GPT-4o-mini) Extracted: {parsed_news}\n")
    
    # ---------------------------------------------------------
    # STEP 2: FEATURE MATRIX & REDISTRIBUTION
    # ---------------------------------------------------------
    print("[STEP 2] Feature Matrix & Redistribution")
    print(f"  > Catalyst Detected: {parsed_news['player']} ({parsed_news['status']})")
    print("  > Triggering UsageRecalculator...")
    time.sleep(0.5)
    # Mock redistribution
    active_roster = {"P.J. Washington": 0.15, "Kyrie Irving": 0.28, "Dereck Lively": 0.12}
    print("  > Base Usage -> P.J. Washington: 15.0%")
    print("  > Re-distributing 35% usage from Luka Doncic to active roster...")
    time.sleep(0.5)
    print("  > Adjusted Usage -> P.J. Washington: 24.5%")
    print("  > Environmental Context: feature_altitude_shock_flag = 1 (Denver)\n")
    
    # ---------------------------------------------------------
    # STEP 3: ZINB MODELING
    # ---------------------------------------------------------
    print("[STEP 3] Quantitative Projections & ZINB Modeling")
    print("  > Injecting modified feature matrix into XGBoost PlayerProjectionModel...")
    time.sleep(0.5)
    # Mock tree projections
    x_min = 36.2
    x_usg = 0.245
    print(f"  > xMin Projected: {x_min}")
    print(f"  > xUSG Projected: {x_usg}")
    print(f"  > Running ZINBModel for: P.J. Washington Over 15.5 Points")
    time.sleep(0.5)
    # Mock ZINB calibrated probability
    calibrated_prob = 0.625
    print(f"  > Calibrated True Probability (Over 15.5): {calibrated_prob*100:.1f}%\n")
    
    # ---------------------------------------------------------
    # STEP 4: MARKET PRICING & DE-VIGGING
    # ---------------------------------------------------------
    print("[STEP 4] Market Microstructure (Sharp Book Tracking)")
    print("  > Pulling Pinnacle sharp odds: Over 15.5 (-130) / Under 15.5 (+110)")
    time.sleep(0.5)
    pinnacle_odds = [-130, 110]
    true_probs = ShinDevigger.devig(pinnacle_odds)
    pinnacle_true_prob = true_probs[0]
    print(f"  > Shin's Method Stripped Vig.")
    print(f"  > Sharp Market True Probability: {pinnacle_true_prob*100:.2f}%\n")
    
    # ---------------------------------------------------------
    # STEP 5: EXECUTION EV & CAMOUFLAGE
    # ---------------------------------------------------------
    print("[STEP 5] Execution Engine (EV & Sizing)")
    print("  > Target Platform: PrizePicks 2-Pick Power Play (Implied 57.74% per leg)")
    time.sleep(0.5)
    
    # Assuming the other leg is exactly at the implied threshold just to test this single leg's EV impact
    leg_probs = [calibrated_prob, 0.5774] 
    ev = EVCalculator.calculate_dfs_edge(leg_probs, platform='PrizePicks_2_Power')
    
    print(f"  > Expected Value against Platform: {ev*100:.2f}% (+EV)")
    
    # Fractional Kelly sizing
    # Since Kelly against fixed parlays gets complex, we mock the raw output based on the single leg for simulation
    raw_kelly_fraction = EVCalculator.calculate_fractional_kelly(calibrated_prob, 100, fraction=0.25)
    # If the true prob is 62.5% vs an implied 57.74% (-137 odds equivalent)
    raw_kelly_fraction = EVCalculator.calculate_fractional_kelly(calibrated_prob, -137, fraction=0.25)
    
    bankroll = 2500
    print(f"  > Bankroll: ${bankroll}")
    print(f"  > Raw Quarter-Kelly Output: {raw_kelly_fraction*100:.2f}% (${raw_kelly_fraction * bankroll:.2f})")
    
    time.sleep(0.5)
    masked_bet = BetMaskingEngine.mask_bet_size(raw_kelly_fraction, bankroll=bankroll)
    print(f"  > BetMaskingEngine -> Smoothing stake to prevent algorithmic flagging.")
    print(f"  > Final Execution Stake: ${masked_bet:.2f}\n")
    
    print("=================================================================")
    print("                      PIPELINE COMPLETED                         ")
    print("=================================================================\n")

if __name__ == "__main__":
    run_e2e_simulation()
