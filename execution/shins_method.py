import numpy as np
from scipy.optimize import minimize_scalar
import logging

logger = logging.getLogger("ShinDevigger")

class ShinDevigger:
    @staticmethod
    def american_to_decimal(american_odds: int) -> float:
        """Converts American odds (e.g., -110, +105) to Decimal odds."""
        if american_odds > 0:
            return (american_odds / 100.0) + 1.0
        elif american_odds < 0:
            return (100.0 / abs(american_odds)) + 1.0
        else:
            raise ValueError("Odds cannot be exactly zero.")

    @staticmethod
    def american_to_implied(american_odds: int) -> float:
        """Converts American odds to raw implied probability (includes vig)."""
        dec = ShinDevigger.american_to_decimal(american_odds)
        return 1.0 / dec

    @staticmethod
    def _shin_equation(z: float, implied_probs: list) -> float:
        """
        The objective function to solve for z (the proportion of sharp/insider money).
        The sum of the true probabilities must equal 1.0.
        True_Prob_i = (sqrt(z^2 + 4 * (1-z) * (Implied_Prob_i^2 / implied_prob_i... wait))
        Actually, the mathematically correct formula for true probability pi_i under Shin is:
        pi_i = (sqrt(z^2 + 4 * (1-z) * (Pi^implied)) - z) / (2 * (1-z))
        We want sum(pi_i) - 1.0 = 0
        """
        total = 0.0
        for p_implied in implied_probs:
            # Avoid division by zero when z -> 1
            if z >= 1.0:
                return float('inf')
                
            numerator = np.sqrt(z**2 + 4 * (1 - z) * p_implied) - z
            denominator = 2 * (1 - z)
            total += (numerator / denominator)
            
        return total - 1.0

    @staticmethod
    def devig(american_odds_list: list) -> list:
        """
        Takes a list of American odds (e.g., [-115, -105]) for an exhaustive mutually exclusive market.
        Returns a list of 'true' implied probabilities using Shin's Method.
        """
        implied_probs = [ShinDevigger.american_to_implied(odds) for odds in american_odds_list]
        
        sum_implied = sum(implied_probs)
        if sum_implied <= 1.0:
            logger.warning("Market has no vig or negative vig. Returning raw normalized probabilities.")
            return [p / sum_implied for p in implied_probs]

        # Use root finding to solve for z in the interval [0, 1)
        # z represents the proportion of insider bettors
        try:
            # We minimize the squared error instead of looking for a root bracket
            def objective(z):
                return ShinDevigger._shin_equation(z, implied_probs) ** 2
                
            res = minimize_scalar(
                objective,
                bounds=(0.0, 0.999),
                method='bounded'
            )
            
            if not res.success or res.fun > 1e-4:
                logger.error("Shin's optimization failed to converge. Falling back to multiplicative de-vigging.")
                return [p / sum_implied for p in implied_probs]
                
            z = res.x
            
            # Calculate true probabilities using the optimized z
            true_probs = []
            for p_implied in implied_probs:
                numerator = np.sqrt(z**2 + 4 * (1 - z) * p_implied) - z
                denominator = 2 * (1 - z)
                true_probs.append(numerator / denominator)
                
            return true_probs
            
        except Exception as e:
            logger.error(f"Error during Shin De-vigging: {e}. Falling back to multiplicative method.")
            return [p / sum_implied for p in implied_probs]
