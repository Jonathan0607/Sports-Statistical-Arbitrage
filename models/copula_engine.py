import numpy as np
import pandas as pd
from copulas.bivariate import Clayton, Gumbel, Frank
import logging

logger = logging.getLogger("CopulaEngine")

class CopulaEngine:
    def __init__(self, copula_type='clayton'):
        """
        Initializes an Archimedean copula.
        Clayton is good for lower-tail dependence (e.g. if the PG struggles, the Center likely struggles).
        Gumbel is good for upper-tail dependence (e.g. if the QB goes off, the WR goes off).
        """
        if copula_type.lower() == 'clayton':
            self.copula = Clayton()
        elif copula_type.lower() == 'gumbel':
            self.copula = Gumbel()
        elif copula_type.lower() == 'frank':
            self.copula = Frank()
        else:
            raise ValueError(f"Unsupported copula type: {copula_type}")
            
    def fit(self, data_a: pd.Series, data_b: pd.Series):
        """
        Fits the copula using pseudo-observations from two correlated time series.
        """
        # Transform margins to uniform [0, 1] using empirical CDF (pseudo-observations)
        u = data_a.rank() / (len(data_a) + 1)
        v = data_b.rank() / (len(data_b) + 1)
        
        # Use a 2D numpy array instead of a DataFrame
        X = np.column_stack((u, v))
        
        try:
            self.copula.fit(X)
            logger.info(f"{self.copula.__class__.__name__} Copula fitted successfully. Theta: {getattr(self.copula, 'theta', 'N/A')}")
        except Exception as e:
            logger.error(f"Failed to fit Copula: {e}")
            raise
            
    def evaluate_joint_probability(self, prob_a: float, prob_b: float) -> float:
        """
        Calculates the joint cumulative probability P(U <= u, V <= v) using the fitted copula.
        For sports betting, if prob_a is the probability of Player A hitting their UNDER,
        and prob_b is the probability of Player B hitting their UNDER,
        this returns the joint probability of BOTH hitting the UNDER.
        """
        X_eval = np.array([[prob_a, prob_b]])
        
        # The CDF evaluates P(U <= prob_a, V <= prob_b)
        joint_cdf = self.copula.cumulative_distribution(X_eval)[0]
        return float(joint_cdf)
        
    def evaluate_joint_survival(self, prob_u: float, prob_v: float) -> float:
        """
        Calculates the joint survival probability P(U > prob_u, V > prob_v).
        Used for pricing an OVER / OVER Same Game Parlay.
        """
        joint_cdf = self.evaluate_joint_probability(prob_u, prob_v)
        # Survival copula relation
        return 1.0 - prob_u - prob_v + joint_cdf
