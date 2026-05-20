import numpy as np
import logging

logger = logging.getLogger("SGPCorrelationEngine")

class SGPCorrelationEngine:
    def __init__(self):
        """
        Archimedean Copula Engine for pricing Same Game Parlays (SGPs).
        """
        pass

    @staticmethod
    def _clayton_theta_from_tau(kendalls_tau: float) -> float:
        """
        Converts Kendall's Tau rank correlation coefficient into the Clayton Copula parameter (theta).
        Tau = theta / (theta + 2) => theta = (2 * Tau) / (1 - Tau)
        """
        if kendalls_tau >= 1.0:
            return 1e6 # High dependence limit
        if kendalls_tau <= 0.0:
            return 1e-6 # Clayton only models positive dependence
            
        return (2.0 * kendalls_tau) / (1.0 - kendalls_tau)

    def price_joint_probability(self, prob_a: float, prob_b: float, kendalls_tau: float, copula_type='clayton') -> float:
        """
        Calculates the mathematically fair probability of both legs hitting simultaneously.
        
        prob_a: Marginal probability of Leg A (e.g., PG Over 8.5 Assists)
        prob_b: Marginal probability of Leg B (e.g., Center Over 15.5 Points)
        kendalls_tau: The historical rank correlation between these two specific stats.
        """
        # Constrain probabilities
        u = np.clip(prob_a, 1e-6, 1.0 - 1e-6)
        v = np.clip(prob_b, 1e-6, 1.0 - 1e-6)
        
        if copula_type.lower() == 'clayton':
            # Clayton Copula Formula: C(u,v) = max(u^(-theta) + v^(-theta) - 1, 0)^(-1/theta)
            theta = self._clayton_theta_from_tau(kendalls_tau)
            
            try:
                joint_prob = max((u ** -theta) + (v ** -theta) - 1.0, 0.0) ** (-1.0 / theta)
                return float(joint_prob)
            except OverflowError:
                # If theta is massive, implies perfect dependence: C(u,v) = min(u,v)
                return float(min(u, v))
                
        elif copula_type.lower() == 'frank':
            # Frank Copula can handle both positive and negative dependence
            # Tau to Theta for Frank requires numerical inversion, but for this engine,
            # we will approximate or use the standard Clayton which dominates SGP lower-tail pricing.
            logger.warning("Frank Copula approximation not implemented, falling back to independent probability.")
            return u * v
            
        else:
            raise ValueError(f"Unsupported copula type: {copula_type}")
