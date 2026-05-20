import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP
import logging
from scipy import stats

logger = logging.getLogger("ZINBModel")

class ZINBModel:
    def __init__(self):
        self.pi = 0.0
        self.mu = 0.0
        self.n = 1.0 # dispersion
    
    def fit(self, endog: pd.Series):
        """
        Fits a Zero-Inflated Negative Binomial distribution.
        Uses a robust Method of Moments approach suitable for high-frequency sports pipelines
        to avoid the convergence failures common in statsmodels GLMs.
        """
        data = np.array(endog)
        
        # 1. Estimate pi (Zero-inflation probability)
        # In sports (like NBA points), a true 0 is often a DNP or blowout scratch.
        zeros_count = np.sum(data == 0)
        total_count = len(data)
        
        # 2. Isolate the "played" data (non-zero)
        non_zero_data = data[data > 0]
        
        if len(non_zero_data) == 0:
            self.pi = 1.0
            self.mu = 0.0
            self.n = 1.0
            logger.info("ZINB Model fitted (all zeros).")
            return
            
        sample_mu = np.mean(non_zero_data)
        sample_var = np.var(non_zero_data)
        
        # The zero count consists of structural zeros (pi) + NB zeros (1-pi)*P(NB=0)
        # For a robust first-pass, we assume most zeros are structural if sample_mu > 5 (like points).
        self.pi = zeros_count / total_count if total_count > 0 else 0
        
        self.mu = sample_mu
        
        # Negative Binomial variance: var = mu + mu^2 / n  =>  n = mu^2 / (var - mu)
        if sample_var > sample_mu:
            self.n = (sample_mu ** 2) / (sample_var - sample_mu)
        else:
            # Poisson limit (variance == mean). n -> infinity
            self.n = 1e6
            
    def condition_parameters(self, blowout_risk: float, pace_factor: float):
        """
        Dynamically shifts the fitted distribution parameters based on categorical game context.
        blowout_risk: 0.0 to 1.0 probability of a blowout.
        pace_factor: Scaling multiplier for game pace (e.g., 1.05 for fast, 0.95 for slow).
        """
        if blowout_risk > 0.8:
            logger.info("High blowout risk detected. Widening tails and increasing zero-inflation.")
            # Increase probability of exact zero (DNP / early benching)
            self.pi = min(self.pi + 0.15, 0.99)
            # Widen the dispersion (lower n means higher variance in Negative Binomial)
            self.n = max(self.n * 0.7, 1.0)
            
        if pace_factor > 1.05:
            logger.info(f"High pace detected ({pace_factor}). Scaling mean up.")
            self.mu = self.mu * pace_factor
            # Maintain variance ratio by slightly widening dispersion
            self.n = max(self.n * 0.9, 1.0)
        elif pace_factor < 0.95:
            logger.info(f"Slow pace detected ({pace_factor}). Scaling mean down.")
            self.mu = self.mu * pace_factor

    def predict_over_probability(self, line: float) -> float:
        """
        Calculates the probability of observing a value > 'line' based on the fitted distribution.
        """
        if self.mu == 0 and self.pi == 1.0:
            return 0.0
            
        domain = np.arange(0, 150)
        
        # Scipy nbinom parametrization: n, p
        # where p = n / (n + mu)
        p = self.n / (self.n + self.mu)
        
        nb_pmf = stats.nbinom.pmf(domain, self.n, p)
        
        pmf = np.zeros(len(domain))
        pmf[0] = self.pi + (1 - self.pi) * nb_pmf[0]
        pmf[1:] = (1 - self.pi) * nb_pmf[1:]
        
        # strictly greater than line
        cutoff = int(np.floor(line)) + 1
        
        if cutoff >= len(domain):
            return 0.0
            
        prob_over = np.sum(pmf[cutoff:])
        return float(prob_over)
