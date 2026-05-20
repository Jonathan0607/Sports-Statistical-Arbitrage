import logging

logger = logging.getLogger("EVEngine")

class EVCalculator:
    @staticmethod
    def calculate_ev(true_prob: float, american_odds: int) -> float:
        """
        Calculates Expected Value (EV) percentage.
        true_prob: The true probability of the event occurring (e.g., 0.55 for 55%)
        american_odds: The odds offered by the retail sportsbook.
        
        Formula: EV = (True_Prob * Decimal_Payout) - 1.0
        """
        # Convert American odds to Decimal Payout (including original stake)
        if american_odds > 0:
            decimal_payout = (american_odds / 100.0) + 1.0
        else:
            decimal_payout = (100.0 / abs(american_odds)) + 1.0
            
        # EV is expressed as a percentage of the stake
        # e.g., 0.05 means 5% expected return on investment
        expected_value = (true_prob * decimal_payout) - 1.0
        return float(expected_value)

    @staticmethod
    def calculate_fractional_kelly(true_prob: float, american_odds: int, fraction: float = 0.25) -> float:
        """
        Calculates the optimal bet size as a percentage of the bankroll using the Kelly Criterion.
        Scaled by a fraction (e.g., 0.25 for Quarter-Kelly) to manage variance.
        
        Kelly Formula: f* = p - (q / b)
        where:
        p = true probability of winning
        q = true probability of losing (1 - p)
        b = net fractional odds received on the bet (decimal odds - 1)
        """
        if true_prob <= 0 or true_prob >= 1:
            return 0.0
            
        q = 1.0 - true_prob
        
        # Calculate 'b' (net decimal odds)
        if american_odds > 0:
            b = american_odds / 100.0
        else:
            b = 100.0 / abs(american_odds)
            
        # Calculate full Kelly fraction
        kelly_f = true_prob - (q / b)
        
        # If EV is negative, Kelly says do not bet
        if kelly_f <= 0:
            return 0.0
            
        # Apply fractional scaling (e.g., Quarter-Kelly)
        scaled_kelly = kelly_f * fraction
        
        # Cap max bet size at 5% of bankroll to protect against extreme edge overconfidence
        return min(scaled_kelly, 0.05)

    @staticmethod
    def calculate_dfs_edge(leg_probabilities: list, platform: str) -> float:
        """
        Calculates Expected Value against Fixed-Payout DFS platform multipliers.
        
        leg_probabilities: List of true probabilities for each leg.
        platform: String identifier (e.g., 'PrizePicks_2_Power')
        """
        if not leg_probabilities:
            return 0.0
            
        # Assuming independent legs unless a copula pre-calculated the joint probability
        joint_prob = 1.0
        for p in leg_probabilities:
            joint_prob *= p
            
        if platform == 'PrizePicks_2_Power':
            # Pays 3X (which means risking 1 to win 3 total, +200 odds equivalent)
            # Implied probability of the slip: 1 / 3 = 0.3333
            payout = 3.0
        elif platform == 'PrizePicks_3_Power':
            # Pays 5X
            payout = 5.0
        else:
            logger.warning(f"Unknown DFS platform/slip type: {platform}. Defaulting to 1X.")
            payout = 1.0
            
        ev = (joint_prob * payout) - 1.0
        return float(ev)
