import numpy as np
import pandas as pd
from arch import arch_model
import logging
from copulas.bivariate import Clayton, Gumbel, Frank
import statsmodels.api as sm
from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP
from scipy import stats
from scipy.optimize import minimize_scalar
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Tuple
import os
import sys
import lifelines
try:
    from src import quant_core
except ImportError:
    try:
        import quant_core
    except ImportError:
        quant_core = None
import math
import sqlite3
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend to save figure without window GUI
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════════════════════
# MODELS.PY — Consolidated Quantitative Modeling Layer
# Merges: models/models.py, bayesian_prior.py, pca_factor_model.py, robust_kelly.py
# ═══════════════════════════════════════════════════════════════════════════════

logger = logging.getLogger("MonteCarloEngine")


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1: Player Projection Simulator (Monte Carlo)
# ═══════════════════════════════════════════════════════════════════════════════

class PlayerProjectionSimulator:
    """
    Simulates individual player performances using Monte Carlo methods
    to derive true probabilities for Over/Under prop bets.
    """
    @staticmethod
    def calculate_dynamic_overdispersion(historical_stats: list, default_factor=1.25) -> float:
        """
        Calculates the player-specific overdispersion factor based on their game logs.
        Formula: Variance / Mean. 
        """
        if historical_stats is None or len(historical_stats) < 5:
            return default_factor
            
        mean = np.mean(historical_stats)
        if mean <= 0.5: 
            return default_factor
            
        # Use sample variance (ddof=1)
        variance = np.var(historical_stats, ddof=1)
        factor = variance / mean
        
        # Cap the factor between 1.0 (Poisson) and 3.0 (Extreme Variance) 
        # to prevent small-sample-size outliers from breaking the model.
        return float(max(1.0, min(factor, 3.0)))
        
    @staticmethod
    def simulate_stat_distribution(projected_mean: float, sportsbook_line: float,
                                   overdispersion_factor: float = 1.2,
                                   n_simulations: int = 10000) -> dict:
        """
        Simulates a player's stat (Points, Rebounds, etc.) using a Negative Binomial distribution.
        Basketball stats are count data with high variance (overdispersion).

        projected_mean: The player's true or seasonal average (e.g., 26.5 points)
        sportsbook_line: The line offered by the book (e.g., 27.5)
        overdispersion_factor: Ratio of variance to mean (usually 1.1 to 1.5 for NBA scoring)
        """
        variance = projected_mean * overdispersion_factor

        # If variance is artificially low, fallback to Poisson (variance == mean)
        if variance <= projected_mean:
            simulated_stats = np.random.poisson(lam=projected_mean, size=n_simulations)
        else:
            # Calculate Negative Binomial parameters (r, p)
            p = projected_mean / variance
            r = (projected_mean ** 2) / (variance - projected_mean)
            simulated_stats = np.random.negative_binomial(n=r, p=p, size=n_simulations)

        # Calculate Over/Under probabilities
        # E.g., Line is 27.5. Over means >= 28. Under means <= 27.
        over_hits = np.sum(simulated_stats > sportsbook_line)
        under_hits = np.sum(simulated_stats < sportsbook_line)
        pushes = np.sum(simulated_stats == sportsbook_line) # Only triggers on whole-number lines

        prob_over = over_hits / n_simulations
        prob_under = under_hits / n_simulations

        # Helper to convert probability to American Odds
        def prob_to_american(prob):
            if prob <= 0: return "+99900"
            if prob >= 1: return "-99900"
            if prob > 0.5:
                return int((prob / (1 - prob)) * -100)
            else:
                return int((100 / prob) - 100)
                
        return {
            'projected_mean': projected_mean,
            'sportsbook_line': sportsbook_line,
            'simulations_run': n_simulations,
            'prob_over': prob_over,
            'prob_under': prob_under,
            'fair_odds_over': prob_to_american(prob_over),
            'fair_odds_under': prob_to_american(prob_under),
            '95th_percentile_ceiling': int(np.percentile(simulated_stats, 95)),
            '5th_percentile_floor': int(np.percentile(simulated_stats, 5))
        }

    @staticmethod
    def simulate_board_matrix(projected_means: np.ndarray, sportsbook_lines: np.ndarray, overdispersion_factors: np.ndarray, n_simulations: int = 10000) -> dict:
        """
        Simulates the entire active board in a single vectorized NumPy operation.
        Eliminates Python-level iteration overhead.
        """
        projected_means = np.asarray(projected_means, dtype=float)
        sportsbook_lines = np.asarray(sportsbook_lines, dtype=float)
        overdispersion_factors = np.asarray(overdispersion_factors, dtype=float)
        
        # Ensure that if variances == projected_means exactly, the overdispersion factor is exactly 1.0
        variances = projected_means * overdispersion_factors
        exact_match_mask = np.isclose(variances, projected_means, rtol=1e-9, atol=1e-9)
        overdispersion_factors = np.where(exact_match_mask, 1.0, overdispersion_factors)
        variances = projected_means * overdispersion_factors
        
        # Separate Poisson (low variance) from Negative Binomial (high variance)
        poisson_mask = variances <= projected_means
        nb_mask = ~poisson_mask
        
        num_players = len(projected_means)
        simulated_stats = np.zeros((num_players, n_simulations))
        
        if np.any(poisson_mask):
            lams = projected_means[poisson_mask]
            # Broadcast to matrix (M, n_sims)
            simulated_stats[poisson_mask, :] = np.random.poisson(
                lam=lams[:, None], size=(np.sum(poisson_mask), n_simulations)
            )
            
        if np.any(nb_mask):
            means_nb = projected_means[nb_mask]
            vars_nb = variances[nb_mask]
            p = means_nb / vars_nb
            r = (means_nb ** 2) / ((vars_nb - means_nb) + 1e-6)
            # Broadcast to matrix (M, n_sims)
            simulated_stats[nb_mask, :] = np.random.negative_binomial(
                n=r[:, None], p=p[:, None], size=(np.sum(nb_mask), n_simulations)
            )
            
        # Vectorized logical operations against the target retail lines
        over_hits = np.sum(simulated_stats > sportsbook_lines[:, None], axis=1)
        under_hits = np.sum(simulated_stats < sportsbook_lines[:, None], axis=1)
        
        return {
            'prob_over': over_hits / n_simulations,
            'prob_under': under_hits / n_simulations,
            'simulated_matrix': simulated_stats
        }

    @staticmethod
    def calibrate_probabilities(raw_probs: np.ndarray, empirical_outcomes: np.ndarray) -> np.ndarray:
        """
        Utilizes sklearn.isotonic.IsotonicRegression(out_of_bounds='clip') to fit the raw
        ZINB Monte Carlo probabilities against a binary array of historical outcomes
        (0 for Under, 1 for Over). Returns the calibrated probability array.
        """
        ir = IsotonicRegression(out_of_bounds='clip')
        # Ensure correct type/dimension
        raw_probs_arr = np.asarray(raw_probs, dtype=float)
        empirical_outcomes_arr = np.asarray(empirical_outcomes, dtype=float)
        
        # Fit and transform
        calibrated = ir.fit_transform(raw_probs_arr, empirical_outcomes_arr)
        return np.asarray(calibrated, dtype=float)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2: Market Microstructure (Shin's Method, GARCH, OU, Frank Copula)
# ═══════════════════════════════════════════════════════════════════════════════

class MarketMicrostructure:
    """
    Implements advanced market microstructure algorithms, including Shin's Method
    to de-vig sportsbook props and calculate z-factors (insider trading proportion).
    """
    @staticmethod
    def shins_method(odds_over: int, odds_under: int) -> dict:
        """
        Calculates de-vigged true probabilities and the insider parameter z (z_factor)
        using Shin's iterative solver.
        """
        # Convert American odds to implied probabilities
        def to_implied(odds: int) -> float:
            if odds > 0:
                return 100.0 / (odds + 100.0)
            else:
                return abs(odds) / (abs(odds) + 100.0)
                
        pi_over = to_implied(odds_over)
        pi_under = to_implied(odds_under)
        s = pi_over + pi_under
        
        # Safeguard if sum is less than or equal to 1.0 (no vig / negative margin)
        if s <= 1.0:
            return {
                "true_prob_over": pi_over / s if s > 0 else 0.5,
                "true_prob_under": pi_under / s if s > 0 else 0.5,
                "z_factor": 0.0
            }
            
        # Iterative solver (Bisection method) for z in [0, 1)
        low = 0.0
        high = 1.0 - 1e-9
        z = 0.0
        
        for _ in range(100):
            mid = (low + high) / 2.0
            
            # Shin's formula for true probabilities
            val_over = mid**2 + 4 * (1.0 - mid) * (pi_over**2) / s
            val_under = mid**2 + 4 * (1.0 - mid) * (pi_under**2) / s
            
            p_over = (np.sqrt(max(0.0, val_over)) - mid) / (2.0 * (1.0 - mid))
            p_under = (np.sqrt(max(0.0, val_under)) - mid) / (2.0 * (1.0 - mid))
            
            p_sum = p_over + p_under
            
            if abs(p_sum - 1.0) < 1e-12:
                z = mid
                break
            elif p_sum > 1.0:
                low = mid
            else:
                high = mid
        else:
            z = (low + high) / 2.0
            
        # Compute final probabilities with converged z
        val_over = z**2 + 4 * (1.0 - z) * (pi_over**2) / s
        val_under = z**2 + 4 * (1.0 - z) * (pi_under**2) / s
        
        true_prob_over = (np.sqrt(max(0.0, val_over)) - z) / (2.0 * (1.0 - z))
        true_prob_under = (np.sqrt(max(0.0, val_under)) - z) / (2.0 * (1.0 - z))
        
        # Normalize strictly to ensure exact sum is 1.0
        total = true_prob_over + true_prob_under
        true_prob_over /= total
        true_prob_under /= total
        
        return {
            "true_prob_over": float(true_prob_over),
            "true_prob_under": float(true_prob_under),
            "z_factor": float(z)
        }

    @staticmethod
    def frank_copula_joint_prob(u: float, v: float, theta: float) -> float:
        """
        Calculates the joint cumulative probability of two uniform marginal variables
        using a bivariate Frank Copula.
        Formula: -1/theta * ln(1 + ((exp(-theta*u) - 1) * (exp(-theta*v) - 1)) / (exp(-theta) - 1))
        """
        # Handle theta -> 0 (independence copula)
        if abs(theta) < 1e-9:
            return u * v
            
        num = (math.exp(-theta * u) - 1.0) * (math.exp(-theta * v) - 1.0)
        den = math.exp(-theta) - 1.0
        
        inner_val = 1.0 + (num / den)
        # Handle boundary cases or domain constraints
        inner_val = max(1e-15, inner_val)
        
        joint_prob = -1.0 / theta * math.log(inner_val)
        return float(max(0.0, min(1.0, joint_prob)))

    @staticmethod
    def garch_volatility(returns: list, omega: float, alpha: float, beta: float) -> float:
        """
        Calculates the conditional volatility sigma_t of a GARCH(1,1) process.
        Formula: sigma_t^2 = omega + alpha * r_{t-1}^2 + beta * sigma_{t-1}^2
        """
        returns_arr = np.asarray(returns, dtype=float)
        n = len(returns_arr)
        if n == 0:
            return 0.0
            
        # Initialize variance to standard variance of the returns
        var_0 = np.var(returns_arr, ddof=1) if n > 1 else np.var(returns_arr)
        if var_0 <= 0:
            # Fallback to GARCH long-term variance (unconditional variance)
            if (1.0 - alpha - beta) > 0:
                var_0 = omega / (1.0 - alpha - beta)
            else:
                var_0 = omega
                
        current_var = var_0
        for r in returns_arr:
            current_var = omega + alpha * (r**2) + beta * current_var
            
        return float(math.sqrt(max(0.0, current_var)))

    @staticmethod
    def ou_mean_reversion_prob(current_price: float, long_term_mean: float, reversion_rate: float, volatility: float, dt: float) -> float:
        """
        Calculates the expected path value of an Ornstein-Uhlenbeck process after time dt.
        Formula: E[X_t] = X_0 * exp(-r*t) + mu * (1 - exp(-r*t))
        """
        exp_factor = math.exp(-reversion_rate * dt)
        expected_val = current_price * exp_factor + long_term_mean * (1.0 - exp_factor)
        return float(expected_val)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3: Volatility Tracker (GARCH)
# ═══════════════════════════════════════════════════════════════════════════════

logger_vol = logging.getLogger('VolatilityTracker')

class VolatilityTracker:

    def __init__(self):
        self.model = None
        self.result = None
        self.scale = 1.0

    def fit(self, returns_series: pd.Series):
        """
        Fits a GARCH(1,1) model to the given series.
        In sports context, this is typically the residual (actual points - expected points),
        or percentage deviations from the EWMA baseline.
        """
        if returns_series.std() < 1:
            self.scale = 100.0
        scaled_series = returns_series * self.scale
        try:
            self.model = arch_model(scaled_series, vol='Garch', p=1, q=1, rescale=False)
            self.result = self.model.fit(disp='off')
            logger_vol.info('GARCH(1,1) model fitted successfully.')
        except Exception as e:
            logger_vol.error(f'Failed to fit GARCH model: {e}')
            raise

    def forecast_variance(self, horizon: int=1) -> float:
        """
        Forecasts the expected variance for the next 'horizon' periods.
        """
        if self.result is None:
            raise ValueError('Model must be fitted before forecasting.')
        forecast = self.result.forecast(horizon=horizon)
        forecasted_var = forecast.variance.iloc[-1, 0] / self.scale ** 2
        return float(forecasted_var)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4: Copula Engine (Archimedean)
# ═══════════════════════════════════════════════════════════════════════════════

logger_copula = logging.getLogger('CopulaEngine')

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
            raise ValueError(f'Unsupported copula type: {copula_type}')

    def fit(self, data_a: pd.Series, data_b: pd.Series):
        """
        Fits the copula using pseudo-observations from two correlated time series.
        """
        u = data_a.rank() / (len(data_a) + 1)
        v = data_b.rank() / (len(data_b) + 1)
        X = np.column_stack((u, v))
        try:
            self.copula.fit(X)
            logger_copula.info(f"{self.copula.__class__.__name__} Copula fitted successfully. Theta: {getattr(self.copula, 'theta', 'N/A')}")
        except Exception as e:
            logger_copula.error(f'Failed to fit Copula: {e}')
            raise

    def evaluate_joint_probability(self, prob_a: float, prob_b: float) -> float:
        """
        Calculates the joint cumulative probability P(U <= u, V <= v) using the fitted copula.
        For sports betting, if prob_a is the probability of Player A hitting their UNDER,
        and prob_b is the probability of Player B hitting their UNDER,
        this returns the joint probability of BOTH hitting the UNDER.
        """
        X_eval = np.array([[prob_a, prob_b]])
        joint_cdf = self.copula.cumulative_distribution(X_eval)[0]
        return float(joint_cdf)

    def evaluate_joint_survival(self, prob_u: float, prob_v: float) -> float:
        """
        Calculates the joint survival probability P(U > prob_u, V > prob_v).
        Used for pricing an OVER / OVER Same Game Parlay.
        """
        joint_cdf = self.evaluate_joint_probability(prob_u, prob_v)
        return 1.0 - prob_u - prob_v + joint_cdf


# ═══════════════════════════════════════════════════════════════════════════════
# Section 5: ZINB Model (Zero-Inflated Negative Binomial)
# ═══════════════════════════════════════════════════════════════════════════════

logger_zinb = logging.getLogger('ZINBModel')

class ZINBModel:

    def __init__(self):
        self.pi = 0.0
        self.mu = 0.0
        self.n = 1.0

    def fit(self, endog: pd.Series):
        """
        Fits a Zero-Inflated Negative Binomial distribution.
        Uses a robust Method of Moments approach suitable for high-frequency sports pipelines
        to avoid the convergence failures common in statsmodels GLMs.
        """
        data = np.array(endog)
        zeros_count = np.sum(data == 0)
        total_count = len(data)
        non_zero_data = data[data > 0]
        if len(non_zero_data) == 0:
            self.pi = 1.0
            self.mu = 0.0
            self.n = 1.0
            logger_zinb.info('ZINB Model fitted (all zeros).')
            return
        sample_mu = np.mean(non_zero_data)
        sample_var = np.var(non_zero_data)
        pi_raw = zeros_count / total_count if total_count > 0 else 0
        self.mu = sample_mu
        if sample_var > sample_mu:
            self.n = sample_mu ** 2 / (sample_var - sample_mu)
        else:
            self.n = 1000000.0
        
        p = self.n / (self.n + self.mu)
        p_n = p ** self.n
        # Corrected zero-inflation parameter to account for NB-generated zeros
        self.pi = max(0.0, (pi_raw - p_n) / max(1e-9, 1.0 - p_n))

    def condition_parameters(self, spread: float = 0.0, pace_factor: float = 100.0, blowout_risk: float = None):
        """
        Dynamically shifts the ZINB distribution parameters based on game spread and pace context.
        Handles legacy blowout_risk/pace_factor scaling for backward compatibility.
        """
        # Backwards compatibility handler
        if blowout_risk is not None:
            if blowout_risk > 0.8:
                spread = 15.0
            else:
                spread = 0.0
            if pace_factor <= 2.0:
                pace_factor = pace_factor * 100.0

        # Spread Condition (High Blowout Risk)
        if abs(spread) >= 12.0:
            logger_zinb.info(f"High blowout risk (spread={spread:.2f}) detected. Widening tails.")
            self.pi = min(self.pi + 0.05, 0.99)
            self.n = max(self.n * 0.9, 1.0)

        # Pace Condition
        if pace_factor > 102.0:
            logger_zinb.info(f"High pace ({pace_factor:.2f}) detected. Widening variance tails.")
            self.n = max(self.n * 0.85, 1.0)
        elif pace_factor < 96.0:
            logger_zinb.info(f"Slow pace ({pace_factor:.2f}) detected. Tightening variance.")
            self.n = self.n * 1.15

    def predict_over_probability(self, line: float) -> float:
        """
        Calculates the probability of observing a value > 'line' based on the fitted distribution.
        """
        if self.mu == 0 and self.pi == 1.0:
            return 0.0
        p = self.n / (self.n + self.mu)
        cutoff = int(np.floor(line))
        cdf = self.pi + (1 - self.pi) * stats.nbinom.cdf(cutoff, self.n, p)
        prob_over = 1.0 - cdf
        return float(max(0.0, min(1.0, prob_over)))


# ═══════════════════════════════════════════════════════════════════════════════
# Section 6: SGP Correlation Engine (Same Game Parlay)
# ═══════════════════════════════════════════════════════════════════════════════

logger_sgp = logging.getLogger('SGPCorrelationEngine')

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
            return 1000000.0
        if kendalls_tau <= 0.0:
            return 1e-06
        return 2.0 * kendalls_tau / (1.0 - kendalls_tau)

    def price_joint_probability(self, prob_a: float, prob_b: float, kendalls_tau: float, copula_type='clayton') -> float:
        """
        Calculates the mathematically fair probability of both legs hitting simultaneously.
        
        prob_a: Marginal probability of Leg A (e.g., PG Over 8.5 Assists)
        prob_b: Marginal probability of Leg B (e.g., Center Over 15.5 Points)
        kendalls_tau: The historical rank correlation between these two specific stats.
        """
        u = np.clip(prob_a, 1e-06, 1.0 - 1e-06)
        v = np.clip(prob_b, 1e-06, 1.0 - 1e-06)
        if copula_type.lower() == 'clayton':
            theta = self._clayton_theta_from_tau(kendalls_tau)
            try:
                joint_prob = max(u ** (-theta) + v ** (-theta) - 1.0, 0.0) ** (-1.0 / theta)
                return float(joint_prob)
            except OverflowError:
                return float(min(u, v))
        elif copula_type.lower() == 'frank':
            logger_sgp.warning('Frank Copula approximation not implemented, falling back to independent probability.')
            return u * v
        else:
            raise ValueError(f'Unsupported copula type: {copula_type}')


# ═══════════════════════════════════════════════════════════════════════════════
# Section 7: Player Projection Model (XGBoost)
# ═══════════════════════════════════════════════════════════════════════════════

logger_proj = logging.getLogger('PlayerProjectionModel')

class PlayerProjectionModel:

    def __init__(self):
        """
        Gradient Boosted Tree model for forecasting baseline player expected metrics
        based on tabular game logs and categorical interactions.
        """
        # Tightened XGBoost Regularization Profile for Contextual Data
        self.xgb_params = {
            "n_estimators": 150,
            "max_depth": 4,                # Lowered from default 6 to prevent over-fitting B2B noise
            "learning_rate": 0.05,
            "colsample_bytree": 0.7,       # Forces tree to drop features and ignore collinearity
            "subsample": 0.8,              # Row-level variance
            "reg_lambda": 1.5,             # L2 regularization (Smooths extreme weights)
            "reg_alpha": 0.5,              # L1 regularization (Forces useless stats to exact zero)
            "objective": "count:poisson"   # Standard objective for discrete sports metrics
        }
        self.xmin_model = xgb.XGBRegressor(**self.xgb_params)
        self.xusg_model = xgb.XGBRegressor(**self.xgb_params)
        self.is_fitted = False

    def fit(self, X_train: np.ndarray, y_min: np.ndarray, y_usg: np.ndarray):
        """
        Trains the XGBoost models.
        X_train: Matrix of features (e.g., historical stats, matchup strength, categorical flags like 'starting_center_out')
        y_min: Target expected minutes
        y_usg: Target expected usage rate
        """
        try:
            logger_proj.info('Training xMin XGBoost sub-model...')
            self.xmin_model.fit(X_train, y_min)
            logger_proj.info('Training xUSG XGBoost sub-model...')
            self.xusg_model.fit(X_train, y_usg)
            self.is_fitted = True
            logger_proj.info('XGBoost training complete.')
        except Exception as e:
            logger_proj.error(f'Failed to fit XGBoost models: {e}')
            raise

    def predict(self, X_eval: np.ndarray) -> dict:
        """
        Generates point projections for minutes and usage based on the active state matrix.
        """
        if not self.is_fitted:
            logger_proj.warning('Model not fitted. Returning baseline defaults.')
            return {'xMin': 25.0, 'xUSG': 0.2}
        x_min = self.xmin_model.predict(X_eval)
        x_usg = self.xusg_model.predict(X_eval)
        return {'xMin': float(x_min[0]), 'xUSG': float(x_usg[0])}


# ═══════════════════════════════════════════════════════════════════════════════
# Section 8: Player Volatility Tracker (GARCH per Player)
# ═══════════════════════════════════════════════════════════════════════════════

logger_pvol = logging.getLogger('PlayerVolatilityTracker')

class PlayerVolatilityTracker:

    def __init__(self):
        """
        GARCH(1,1) Volatility Tracker for Player Performance.
        Models conditional variance: sigma_t^2 = omega + alpha * epsilon_{t-1}^2 + beta * sigma_{t-1}^2
        """
        self.fitted_models = {}

    def get_current_regime_variance(self, player_id: str, history_array: list) -> float:
        """
        Analyzes a player's recent performance history to determine their current volatility regime.
        If they are in a highly volatile streak (boom-or-bust), this will output a high variance,
        which can be injected into the ZINB distribution to widen the tails.
        """
        if len(history_array) < 10:
            logger_pvol.warning(f'Insufficient history for {player_id}. Falling back to standard sample variance.')
            return np.var(history_array) if len(history_array) > 1 else 1.0
        series = pd.Series(history_array)
        rolling_mean = series.ewm(span=5, min_periods=1).mean()
        residuals = series - rolling_mean
        scale = 100.0 if residuals.std() < 1 else 1.0
        scaled_residuals = residuals * scale
        try:
            model = arch_model(scaled_residuals, vol='Garch', p=1, q=1, rescale=False)
            res = model.fit(disp='off')
            forecast = res.forecast(horizon=1)
            forecasted_var = forecast.variance.iloc[-1, 0] / scale ** 2
            self.fitted_models[player_id] = res
            logger_pvol.info(f'[{player_id}] Volatility Regime Variance: {forecasted_var:.4f}')
            return float(forecasted_var)
        except Exception as e:
            logger_pvol.error(f'GARCH fitting failed for {player_id}: {e}. Returning sample variance.')
            return float(np.var(history_array))


# ═══════════════════════════════════════════════════════════════════════════════
# Section 9: Probability Calibrator (Isotonic Regression)
# ═══════════════════════════════════════════════════════════════════════════════

logger_cal = logging.getLogger('ProbabilityCalibrator')

class ProbabilityCalibrator:

    def __init__(self):
        """
        Uses Isotonic Regression to map raw, mathematically derived probabilities 
        to perfectly calibrated empirical frequencies.
        """
        self.ir = IsotonicRegression(out_of_bounds='clip')
        self.is_fitted = False

    def fit(self, raw_probs: np.ndarray, true_outcomes: np.ndarray):
        """
        Fits a piecewise constant non-decreasing function to smooth prediction bias.
        
        raw_probs: 1D array of predicted probabilities (e.g., from the ZINB model)
        true_outcomes: 1D array of binary outcomes (1 for OVER hit, 0 for miss)
        """
        try:
            self.ir.fit(raw_probs, true_outcomes)
            self.is_fitted = True
            logger_cal.info('Isotonic Regression curve fitted successfully.')
        except Exception as e:
            logger_cal.error(f'Failed to fit Probability Calibrator: {e}')

    def calibrate(self, raw_prob: float) -> float:
        """
        Returns the calibrated probability.
        """
        if not self.is_fitted:
            logger_cal.warning('Calibrator not fitted. Returning uncalibrated probability.')
            return float(raw_prob)
        calibrated = self.ir.predict([raw_prob])[0]
        return float(calibrated)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 10: Shin Devigger (Centralized De-Vigging Engine)
# ═══════════════════════════════════════════════════════════════════════════════

logger_shin = logging.getLogger('ShinDevigger')

class ShinDevigger:

    @staticmethod
    def american_to_decimal(american_odds: int) -> float:
        """Converts American odds (e.g., -110, +105) to Decimal odds."""
        if american_odds > 0:
            return american_odds / 100.0 + 1.0
        elif american_odds < 0:
            return 100.0 / abs(american_odds) + 1.0
        else:
            raise ValueError('Odds cannot be exactly zero.')

    @staticmethod
    def american_to_implied(american_odds: int) -> float:
        """Converts American odds to raw implied probability (includes vig)."""
        dec = ShinDevigger.american_to_decimal(american_odds)
        return 1.0 / dec

    @staticmethod
    def _shin_equation(z: float, implied_probs: list) -> float:
        total = 0.0
        for p_implied in implied_probs:
            if z >= 1.0:
                return float('inf')
            numerator = np.sqrt(z ** 2 + 4 * (1 - z) * p_implied) - z
            denominator = 2 * (1 - z)
            total += numerator / denominator
        return total - 1.0

    @staticmethod
    def devig(american_odds_list: list) -> list:
        implied_probs = [ShinDevigger.american_to_implied(odds) for odds in american_odds_list]
        sum_implied = sum(implied_probs)
        if sum_implied <= 1.0:
            logger_shin.warning('Market has no vig or negative vig. Returning raw normalized probabilities.')
            return [p / sum_implied for p in implied_probs]
        try:
            def objective(z):
                return ShinDevigger._shin_equation(z, implied_probs) ** 2
            res = minimize_scalar(objective, bounds=(0.0, 0.999), method='bounded')
            if not res.success or res.fun > 0.0001:
                logger_shin.error("Shin's optimization failed to converge. Falling back to multiplicative de-vigging.")
                return [p / sum_implied for p in implied_probs]
            z = res.x
            true_probs = []
            for p_implied in implied_probs:
                numerator = np.sqrt(z ** 2 + 4 * (1 - z) * p_implied) - z
                denominator = 2 * (1 - z)
                true_probs.append(numerator / denominator)
            return true_probs
        except Exception as e:
            logger_shin.error(f'Error during Shin De-vigging: {e}. Falling back to multiplicative method.')
            return [p / sum_implied for p in implied_probs]


# ═══════════════════════════════════════════════════════════════════════════════
# Section 11: Middle Scanner (Cross-Book Arbitrage)
# ═══════════════════════════════════════════════════════════════════════════════

logger_middle = logging.getLogger('MiddleScanner')

class MiddleScanner:

    @staticmethod
    def scan_for_arbitrage(market_data: Dict[str, Dict]) -> List[Dict]:
        opportunities = []
        books = list(market_data.keys())
        for i in range(len(books)):
            for j in range(i + 1, len(books)):
                book_a = books[i]
                book_b = books[j]
                data_a = market_data[book_a]
                data_b = market_data[book_b]
                line_a = data_a.get('line')
                line_b = data_b.get('line')
                if line_a is None or line_b is None:
                    continue
                if line_a < line_b:
                    opp = MiddleScanner._evaluate_cross_book(book_a, data_a, book_b, data_b, 'over_odds', 'under_odds', line_a, line_b)
                    if opp:
                        opportunities.append(opp)
                elif line_b < line_a:
                    opp = MiddleScanner._evaluate_cross_book(book_b, data_b, book_a, data_a, 'over_odds', 'under_odds', line_b, line_a)
                    if opp:
                        opportunities.append(opp)
                if line_a == line_b:
                    opp1 = MiddleScanner._evaluate_cross_book(book_a, data_a, book_b, data_b, 'over_odds', 'under_odds', line_a, line_b, is_arb=True)
                    opp2 = MiddleScanner._evaluate_cross_book(book_b, data_b, book_a, data_a, 'over_odds', 'under_odds', line_b, line_a, is_arb=True)
                    if opp1:
                        opportunities.append(opp1)
                    if opp2:
                        opportunities.append(opp2)
        return opportunities

    @staticmethod
    def _evaluate_cross_book(book_over: str, data_over: dict, book_under: str, data_under: dict, over_key: str, under_key: str, line_over: float, line_under: float, is_arb=False) -> Optional[Dict]:
        odds_over = data_over.get(over_key)
        odds_under = data_under.get(under_key)
        if odds_over is None or odds_under is None:
            return None

        def _to_implied(american: int) -> float:
            dec = american / 100.0 + 1.0 if american > 0 else 100.0 / abs(american) + 1.0
            return 1.0 / dec
        implied_over = _to_implied(odds_over)
        implied_under = _to_implied(odds_under)
        sum_implied = implied_over + implied_under
        is_profitable_middle = False
        if not is_arb and line_over < line_under:
            if line_under - line_over >= 1.0 and sum_implied < 1.08:
                is_profitable_middle = True
        is_profitable_arb = is_arb and sum_implied < 1.0
        if is_profitable_arb or is_profitable_middle:
            return {'type': 'PURE_ARBITRAGE' if is_profitable_arb else 'MIDDLE', 'buy_over': {'book': book_over, 'line': line_over, 'odds': odds_over}, 'buy_under': {'book': book_under, 'line': line_under, 'odds': odds_under}, 'implied_sum': round(sum_implied, 4), 'guaranteed_roi': round((1.0 - sum_implied) * 100, 2) if is_profitable_arb else None, 'middle_gap': line_under - line_over if not is_arb else 0.0}
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Section 12: Stochastic Market Monitor (OU Process)
# ═══════════════════════════════════════════════════════════════════════════════

logger_ou = logging.getLogger('StochasticMarketMonitor')

class StochasticMarketMonitor:

    def __init__(self, theta: float=0.1, sigma: float=0.02):
        self.theta = theta
        self.sigma = sigma

    def detect_overreaction(self, current_implied_prob: float, ou_mean: float, time_step_dt: float=1.0) -> bool:
        expected_drift = self.theta * (ou_mean - current_implied_prob) * time_step_dt
        volatility_bound = self.sigma * np.sqrt(time_step_dt)
        actual_delta = current_implied_prob - ou_mean
        jump_threshold = abs(expected_drift) + 3 * volatility_bound
        if abs(actual_delta) > jump_threshold:
            logger_ou.warning(f'[JUMP DETECTED] Line gapped by {actual_delta * 100:.2f}%. Exceeds OU noise threshold of {jump_threshold * 100:.2f}%. Flagging for mean-reversion fade.')
            return True
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# Section 13: Sharp Money Tracker (Kalman Filter)
# ═══════════════════════════════════════════════════════════════════════════════

logger_kalman = logging.getLogger('KalmanTracker')

class SharpMoneyTracker:

    def __init__(self, initial_prob: float, process_variance=1e-05, measurement_variance=0.001):
        self.x = initial_prob
        self.P = 1.0
        self.Q = process_variance
        self.R = measurement_variance

    def update(self, measurement: float) -> float:
        x_pred = self.x
        P_pred = self.P + self.Q
        K = P_pred / (P_pred + self.R)
        self.x = x_pred + K * (measurement - x_pred)
        self.P = (1 - K) * P_pred
        logger_kalman.debug(f'Filtered Prob: {self.x:.4f} (Measurement: {measurement:.4f}, Gain: {K:.4f})')
        return self.x

    def detect_stale_line(self, retail_prob: float, threshold: float=0.02) -> bool:
        discrepancy = abs(self.x - retail_prob)
        if discrepancy > threshold:
            logger_kalman.info(f'🚨 STALE LINE DETECTED: Filtered Sharp ({self.x:.4f}) vs Retail ({retail_prob:.4f})')
            return True
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# Section 14: EV Calculator & Kelly Sizing
# ═══════════════════════════════════════════════════════════════════════════════

logger_ev = logging.getLogger('EVEngine')

class EVCalculator:

    @staticmethod
    def calculate_ev(true_prob: float, american_odds: int) -> float:
        if american_odds > 0:
            decimal_payout = american_odds / 100.0 + 1.0
        else:
            decimal_payout = 100.0 / abs(american_odds) + 1.0
        expected_value = true_prob * decimal_payout - 1.0
        return float(expected_value)

    @staticmethod
    def calculate_fractional_kelly(true_prob: float, american_odds: int, fraction: float=0.25) -> float:
        if true_prob <= 0 or true_prob >= 1:
            return 0.0
        q = 1.0 - true_prob
        if american_odds > 0:
            b = american_odds / 100.0
        else:
            b = 100.0 / abs(american_odds)
        kelly_f = true_prob - q / b
        if kelly_f <= 0:
            return 0.0
        scaled_kelly = kelly_f * fraction
        return min(scaled_kelly, 0.05)

    @staticmethod
    def calculate_dfs_edge(leg_probabilities: list, platform: str) -> float:
        if not leg_probabilities:
            return 0.0
        joint_prob = 1.0
        for p in leg_probabilities:
            joint_prob *= p
        if platform == 'PrizePicks_2_Power':
            payout = 3.0
        elif platform == 'PrizePicks_3_Power':
            payout = 5.0
        else:
            logger_ev.warning(f'Unknown DFS platform/slip type: {platform}. Defaulting to 1X.')
            payout = 1.0
        ev = joint_prob * payout - 1.0
        return float(ev)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 15: Multivariate Kelly Optimizer
# ═══════════════════════════════════════════════════════════════════════════════

logger_kelly = logging.getLogger('MultivariateKelly')

class MultivariateKellyOptimizer:

    @staticmethod
    def optimize_portfolio(expected_values: np.ndarray, covariance_matrix: np.ndarray,
                           max_account_limit: float, kelly_fraction: float = 0.25) -> np.ndarray:
        n_bets = len(expected_values)
        try:
            cov_inv = np.linalg.inv(covariance_matrix)
            full_kelly = cov_inv @ expected_values
        except np.linalg.LinAlgError:
            logger_kelly.error("Covariance matrix is singular. Falling back to diagonal approximation.")
            variances = np.diag(covariance_matrix)
            variances[variances == 0] = 1e-6
            full_kelly = expected_values / variances
        fractional_kelly = full_kelly * kelly_fraction
        fractional_kelly = np.clip(fractional_kelly, 0.0, None)
        fractional_kelly = np.clip(fractional_kelly, 0.0, max_account_limit)
        logger_kelly.info(f"Optimized {n_bets}-bet portfolio. Allocations: {np.round(fractional_kelly, 4)}")
        return fractional_kelly


# ═══════════════════════════════════════════════════════════════════════════════
# Section 16: Bet Masking Engine (Camouflage)
# ═══════════════════════════════════════════════════════════════════════════════

logger_mask = logging.getLogger('BetMaskingEngine')

class BetMaskingEngine:

    @staticmethod
    def mask_kelly_stake(raw_stake: float, league: str = "NBA") -> int:
        if raw_stake <= 0:
            return 0
        from src.config import LEAGUE_CONFIG
        league_key = league.upper() if league else "NBA"
        max_limit = LEAGUE_CONFIG.get(league_key, LEAGUE_CONFIG["NBA"])["max_retail_bet_limit"]
        
        # Aggressively cap the raw stake at the league's max retail bet limit
        capped_stake = min(float(raw_stake), float(max_limit))
        
        if capped_stake < 50:
            masked = round(capped_stake / 5) * 5
        elif capped_stake <= 200:
            masked = round(capped_stake / 10) * 10
        else:
            masked = round(capped_stake / 25) * 25
            
        masked = min(int(masked), int(max_limit))
        return max(masked, 5) if max_limit >= 5 else int(max_limit)

    @staticmethod
    def generate_decoy_bet(ev_edge: float) -> dict:
        if ev_edge <= 0.08:
            return {}
        decoy = {
            'type': 'DECOY_PARLAY',
            'description': 'Popular team ML parlay (2-leg)',
            'implied_ev': -0.04,
            'stake_ratio': 0.15,
            'purpose': 'Account camouflage — dilute sharp signal with recreational noise'
        }
        logger_mask.info(
            f"Sharp edge {ev_edge*100:.1f}% exceeds camouflage threshold. "
            f"Generating decoy bet: {decoy['description']}"
        )
        return decoy


# ═══════════════════════════════════════════════════════════════════════════════
# Section 17: Bayesian Prior (Positional Shrinkage)
# (Merged from bayesian_prior.py)
# ═══════════════════════════════════════════════════════════════════════════════

logger_bayes = logging.getLogger("BayesianPrior")

def get_default_priors(league: str = "NBA") -> dict:
    """Returns reasonable fallback positional priors (per 36 minutes)."""
    priors = {
        'points': {'PG': 18.0, 'SG': 17.0, 'SF': 16.0, 'PF': 15.5, 'C': 16.0},
        'rebounds': {'PG': 4.0, 'SG': 4.5, 'SF': 6.0, 'PF': 8.5, 'C': 10.5},
        'assists': {'PG': 6.5, 'SG': 4.0, 'SF': 3.5, 'PF': 2.5, 'C': 2.0},
        'turnovers': {'PG': 2.5, 'SG': 2.0, 'SF': 1.8, 'PF': 1.5, 'C': 1.6},
        'steals': {'PG': 1.5, 'SG': 1.4, 'SF': 1.3, 'PF': 1.2, 'C': 1.0},
        'blocks': {'PG': 0.3, 'SG': 0.5, 'SF': 0.8, 'PF': 1.5, 'C': 2.2},
        'three_pointers_made': {'PG': 2.5, 'SG': 2.8, 'SF': 2.2, 'PF': 1.5, 'C': 0.5}
    }
    
    if league == "WNBA":
        # WNBA pace and scoring is roughly 72% of NBA. We scale the per-36 priors accordingly.
        for stat, pos_dict in priors.items():
            for pos in pos_dict:
                priors[stat][pos] = round(priors[stat][pos] * 0.72, 2)
                
    return priors

def calculate_positional_priors(box_scores_df: pd.DataFrame) -> dict:
    """
    Calculates the league-average stats per 36 minutes for each position.
    Supports both standard box score CSV headers and database-style column names.
    """
    df = box_scores_df.copy()
    
    # Normalize column names to standard internal names
    col_map = {
        'Minutes': 'minutes', 'minutes_played': 'minutes',
        'Points': 'points',
        'Rebounds': 'rebounds',
        'Assists': 'assists',
        'Turnovers': 'turnovers',
        'Steals': 'steals', 'steals': 'steals',
        'Blocks': 'blocks', 'blocks': 'blocks',
        'ThreePointersMade': 'three_pointers_made', 'three_pointers_made': 'three_pointers_made',
        'Position': 'position', 'position_name': 'position'
    }
    
    # Rename matching columns
    df = df.rename(columns={c: col_map[c] for c in df.columns if c in col_map})
    
    required = ['minutes', 'points', 'rebounds', 'assists', 'turnovers', 'steals', 'blocks', 'three_pointers_made', 'position']
    for req in required:
        if req not in df.columns:
            df[req] = 0.0 if req != 'position' else 'PG'

    # Filter out garbage time (minutes >= 15) to keep priors clean
    core_players = df[df['minutes'] >= 15.0].copy()
    if core_players.empty:
        # Fallback to whole dataset if no players played >= 15 mins to avoid empty groupby
        core_players = df.copy()

    priors = {}
    stats_list = ['points', 'rebounds', 'assists', 'turnovers', 'steals', 'blocks', 'three_pointers_made']
    
    for stat in stats_list:
        # Normalize to Per-36 to standardize volume
        col_name = f"{stat}_per_36"
        core_players[col_name] = (core_players[stat] / (core_players['minutes'] + 1e-5)) * 36
        
        # Group by position and compute the average
        grouped = core_players.groupby('position')[col_name].mean().to_dict()
        priors[stat] = {str(k).upper(): float(v) for k, v in grouped.items()}
        
    return priors

def apply_bayesian_shrinkage(player_stat_per_36: float, player_total_minutes: float, positional_prior: float, K: float = 500.0) -> float:
    """
    Shrinks a player's stat toward the positional prior based on sample size.
    K is the hyperparameter determining how quickly we trust the player's own data.
    Only applies shrinkage if player_total_minutes < 300.0.
    If player_total_minutes >= 300.0, returns the actual player_stat_per_36.
    """
    if player_total_minutes >= 300.0:
        return float(player_stat_per_36)

    # Calculate weight (w approaches 1 as total_minutes grows)
    w = player_total_minutes / (player_total_minutes + K)
    
    # Calculate Posterior
    posterior_stat = (w * player_stat_per_36) + ((1 - w) * positional_prior)
    
    return float(posterior_stat)

def load_all_positional_priors(db_uri: str = None) -> dict:
    """
    Loads the complete prior dictionary from PostgreSQL if db_uri is provided,
    otherwise returns the standard default positional priors dictionary.
    """
    priors = None
    if db_uri:
        try:
            import sqlalchemy
            adjusted_uri = db_uri
            if adjusted_uri.startswith("postgres://"):
                adjusted_uri = adjusted_uri.replace("postgres://", "postgresql+psycopg2://", 1)
            engine = sqlalchemy.create_engine(adjusted_uri)
            query = """
                SELECT g.minutes_played, g.points, g.rebounds, g.assists, g.turnovers,
                       g.steals, g.blocks, g.three_pointers_made, p.position
                FROM player_game_logs g
                JOIN players p ON g.player_id = p.master_player_id
                WHERE g.minutes_played IS NOT NULL AND p.position IS NOT NULL;
            """
            with engine.connect() as conn:
                df = pd.read_sql(query, conn)
            if not df.empty:
                priors = calculate_positional_priors(df)
        except Exception as e:
            logger_bayes.warning(f"Failed to pre-load positional priors from database: {e}. Using defaults.")

    if not priors:
        priors = get_default_priors()
        
    return priors

def get_positional_prior(stat_col: str, position: str, db_uri: str = None, cached_priors: dict = None) -> float:
    """
    Retrieves the positional prior for a specific stat and position.
    Uses cached_priors if provided to run in O(1) memory complexity.
    Otherwise, loads from db_uri (if provided) or uses default priors.
    """
    stat_clean = stat_col.lower().strip()
    # Map synonyms if needed
    if 'point' in stat_clean: stat_clean = 'points'
    elif 'rebound' in stat_clean: stat_clean = 'rebounds'
    elif 'assist' in stat_clean: stat_clean = 'assists'
    elif 'turnover' in stat_clean: stat_clean = 'turnovers'
    
    pos_clean = (position or 'SG').upper().strip()
    
    if cached_priors:
        priors = cached_priors
    elif db_uri:
        priors = load_all_positional_priors(db_uri)
    else:
        priors = get_default_priors()
        
    stat_priors = priors.get(stat_clean, {})
    # Fallback to general average or standard position if not found
    return stat_priors.get(pos_clean, stat_priors.get('SG', 15.0))


# ═══════════════════════════════════════════════════════════════════════════════
# Section 18: PCA Factor Model
# (Merged from pca_factor_model.py)
# ═══════════════════════════════════════════════════════════════════════════════

logger_pca = logging.getLogger("PCAFactorModel")

def load_box_scores_from_db(db_uri: str) -> pd.DataFrame:
    """Queries PostgreSQL database to get player game logs joined with team abbreviations."""
    try:
        import sqlalchemy
        adjusted_uri = db_uri
        if adjusted_uri.startswith("postgres://"):
            adjusted_uri = adjusted_uri.replace("postgres://", "postgresql+psycopg2://", 1)
        engine = sqlalchemy.create_engine(adjusted_uri)
        query = """
            SELECT g.game_id, p.team_abbrev, g.player_id, 
                   COALESCE(g.minutes_played, 0.0) as minutes_played, 
                   COALESCE(g.points, 0) as points, 
                   COALESCE(g.rebounds, 0) as rebounds, 
                   COALESCE(g.assists, 0) as assists
            FROM player_game_logs g
            JOIN players p ON g.player_id = p.master_player_id
            WHERE g.game_id IS NOT NULL AND p.team_abbrev IS NOT NULL;
        """
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
        return df
    except Exception as e:
        logger_pca.warning(f"Failed to load from PostgreSQL database: {e}. Generating mock data.")
        return pd.DataFrame()

def generate_mock_box_scores(num_games: int = 150) -> pd.DataFrame:
    """Generates a realistic mock dataset of box scores for PCA training."""
    np.random.seed(42)
    records = []
    
    positions = ['PG', 'SG', 'SF', 'PF', 'C', 'Bench1', 'Bench2', 'Bench3', 'Bench4', 'Bench5']
    # Mean stats per position [pts, reb, ast, min]
    pos_stats = {
        'PG': [20.0, 4.0, 7.0, 34.0],
        'SG': [18.0, 4.0, 4.0, 32.0],
        'SF': [16.0, 6.0, 3.0, 30.0],
        'PF': [15.0, 8.0, 2.0, 30.0],
        'C':  [14.0, 11.0, 2.0, 32.0],
        'Bench1': [10.0, 3.0, 2.0, 22.0],
        'Bench2': [8.0, 4.0, 1.0, 20.0],
        'Bench3': [6.0, 2.0, 2.0, 18.0],
        'Bench4': [4.0, 2.0, 1.0, 12.0],
        'Bench5': [2.0, 1.0, 0.5, 8.0]
    }
    
    for game_id in range(1, num_games + 1):
        # We simulate 2 teams playing
        team_1 = "T1"
        team_2 = "T2"
        
        # Stochastic game pace multiplier
        pace_multiplier = np.random.normal(1.0, 0.08)
        
        for team in [team_1, team_2]:
            for i, pos in enumerate(positions):
                player_id = f"player_{team}_{pos}"
                base_pts, base_reb, base_ast, base_min = pos_stats[pos]
                
                # Add random noise
                minutes = max(1.0, np.random.normal(base_min, 3.0))
                points = max(0, int(np.random.normal(base_pts * pace_multiplier, 4.0)))
                rebounds = max(0, int(np.random.normal(base_reb * pace_multiplier, 3.0)))
                assists = max(0, int(np.random.normal(base_ast * pace_multiplier, 2.0)))
                
                records.append({
                    "game_id": game_id,
                    "team_abbrev": team,
                    "player_id": player_id,
                    "minutes_played": minutes,
                    "points": points,
                    "rebounds": rebounds,
                    "assists": assists
                })
                
    return pd.DataFrame(records)

def build_fixed_width_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes the feature matrix:
    - Sorts players per team per game by Minutes descending.
    - Picks the Top 8 rotation players per team.
    - Flattens this into 16 players (64 features) per game.
    """
    if df.empty:
        return pd.DataFrame()
        
    game_records = []
    
    # Group by game_id
    grouped_game = df.groupby('game_id')
    for game_id, game_df in grouped_game:
        teams = game_df['team_abbrev'].unique()
        if len(teams) < 2:
            continue # We need exactly 2 teams
            
        t1, t2 = sorted(teams)[:2]
        t1_df = game_df[game_df['team_abbrev'] == t1].sort_values(by='minutes_played', ascending=False).head(8)
        t2_df = game_df[game_df['team_abbrev'] == t2].sort_values(by='minutes_played', ascending=False).head(8)
        
        row_dict = {"game_id": game_id}
        
        # Populate Team 1 players
        for idx in range(8):
            prefix = f"T1_P{idx+1}"
            if idx < len(t1_df):
                p_row = t1_df.iloc[idx]
                row_dict[f"{prefix}_MIN"] = float(p_row['minutes_played'])
                row_dict[f"{prefix}_PTS"] = float(p_row['points'])
                row_dict[f"{prefix}_REB"] = float(p_row['rebounds'])
                row_dict[f"{prefix}_AST"] = float(p_row['assists'])
            else:
                row_dict[f"{prefix}_MIN"] = 0.0
                row_dict[f"{prefix}_PTS"] = 0.0
                row_dict[f"{prefix}_REB"] = 0.0
                row_dict[f"{prefix}_AST"] = 0.0
                
        # Populate Team 2 players
        for idx in range(8):
            prefix = f"T2_P{idx+1}"
            if idx < len(t2_df):
                p_row = t2_df.iloc[idx]
                row_dict[f"{prefix}_MIN"] = float(p_row['minutes_played'])
                row_dict[f"{prefix}_PTS"] = float(p_row['points'])
                row_dict[f"{prefix}_REB"] = float(p_row['rebounds'])
                row_dict[f"{prefix}_AST"] = float(p_row['assists'])
            else:
                row_dict[f"{prefix}_MIN"] = 0.0
                row_dict[f"{prefix}_PTS"] = 0.0
                row_dict[f"{prefix}_REB"] = 0.0
                row_dict[f"{prefix}_AST"] = 0.0
                
        game_records.append(row_dict)
        
    return pd.DataFrame(game_records).set_index("game_id")

def fit_pca_factor_model(matrix_df: pd.DataFrame, save_plot_path: str = "logs/pca_variance_plot.png") -> tuple[StandardScaler, PCA, int]:
    """Fits StandardScaler and PCA on the games matrix, plots scree curve, and returns transformers."""
    if matrix_df.empty:
        raise ValueError("Cannot fit PCA model on an empty feature matrix.")
        
    # Scale features
    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(matrix_df)
    
    # Fit PCA
    pca = PCA()
    pca.fit(scaled_matrix)
    
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    
    # Find number of components to cross >= 80% explained variance
    n_components_80 = int(np.argmax(cumulative_variance >= 0.80) + 1)
    
    # Save Scree Plot
    os.makedirs(os.path.dirname(os.path.abspath(save_plot_path)), exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--', color='blue', label='Cumulative Explained Variance')
    plt.axhline(y=0.80, color='red', linestyle=':', label='80% Variance Threshold')
    plt.axvline(x=n_components_80, color='green', linestyle=':', label=f'{n_components_80} Components (80%)')
    
    plt.title("Scree Plot: PCA Cumulative Explained Variance Ratio", fontsize=14, fontweight='bold')
    plt.xlabel("Number of Principal Components", fontsize=12)
    plt.ylabel("Cumulative Explained Variance Ratio", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower right')
    
    plt.savefig(save_plot_path, dpi=300)
    plt.close()
    
    logger_pca.info(f"Scree plot successfully saved to {save_plot_path}")
    logger_pca.info(f"PCA fit complete. {n_components_80} components required to cross 80% variance threshold.")
    
    return scaler, pca, n_components_80

def extract_game_factors(game_vector: np.ndarray, scaler: StandardScaler, pca: PCA, n_components: int = 3) -> np.ndarray:
    """
    Transforms a raw 64-dimensional game features vector
    into its lower-dimensional PCA factor scores.
    """
    # Reshape if 1D array
    if len(game_vector.shape) == 1:
        game_vector = game_vector.reshape(1, -1)
        
    scaled = scaler.transform(game_vector)
    # Project to lower dimensions
    projected = pca.transform(scaled)
    # Return subset of components
    return projected[:, :n_components]


# ═══════════════════════════════════════════════════════════════════════════════
# Section 19: Robust Kelly (Uncertainty-Adjusted)
# (Merged from robust_kelly.py)
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_robust_kelly(prob_win: float, decimal_odds: float, prob_variance: float, kelly_multiplier: float = 0.25) -> float:
    """
    Calculates the uncertainty-adjusted (Robust) Kelly fraction.
    
    :param prob_win: The expected win probability from the ML model (e.g., 0.58)
    :param decimal_odds: The exchange odds (e.g., 1.91)
    :param prob_variance: The variance/uncertainty of the probability estimate (sigma^2)
    :param kelly_multiplier: Fractional Kelly adjustment (e.g., Quarter Kelly = 0.25)
    :return: The suggested percentage of bankroll to wager.
    """
    b = decimal_odds - 1.0
    
    # 1. Standard Kelly Calculation
    expected_value = (prob_win * b) - (1.0 - prob_win)
    
    if expected_value <= 0:
        return 0.0  # No mathematical edge
        
    f_standard = expected_value / b
    
    # 2. The Uncertainty Penalty (Taylor Expansion)
    # Penalizes the allocation squarely based on model variance and payout structure
    penalty = prob_variance / (b ** 2)
    
    # 3. Robust Kelly Allocation
    f_robust = f_standard - penalty
    
    # Clamp to 0 to prevent shorting, and apply fractional Kelly multiplier (e.g., Quarter Kelly)
    final_allocation = max(0.0, f_robust) * kelly_multiplier
    
    return final_allocation


# ═══════════════════════════════════════════════════════════════════════════════
# Section 20: Bayesian Market Blender (Liquidity-Driven Shrinkage)
# ═══════════════════════════════════════════════════════════════════════════════

class BayesianMarketBlender:
    def __init__(self, confidence_threshold_usd: float = 2000.0):
        """
        confidence_threshold_usd: The dollar limit at which we assume the market 
        is fully efficient and we trust the sharp prior 100%.
        """
        self.confidence_threshold = confidence_threshold_usd

    def get_heuristic_limit(self, prop_name: str) -> float:
        """
        Returns an estimated market limit based on the prop type.
        Mainlines (Moneyline/Spreads) are highly liquid.
        Player derivatives are thin and constrained.
        """
        prop_lower = str(prop_name).lower()
        if any(x in prop_lower for x in ["spread", "moneyline", "total points"]):
            return 5000.0  # High confidence, highly liquid
        elif any(x in prop_lower for x in ["points", "rebounds", "assists"]):
            return 350.0   # Medium confidence, standard player prop
        elif any(x in prop_lower for x in ["steals", "blocks", "turnovers"]):
            return 150.0   # Low confidence, highly volatile tail prop
        else:
            return 250.0   # Default derivative limit

    def calculate_blended_probability(self, 
                                      sharp_prior_prob: float, 
                                      internal_model_prob: float, 
                                      market_limit_usd: float) -> float:
        """
        Dynamically weights the sharp market probability (Prior) against the 
        internal model (Likelihood) based on the liquidity backing the market line.
        """
        market_limit_usd = max(0.0, float(market_limit_usd or 0.0))
        
        # Calculate alpha (market weight). Clamped between 0.0 and 1.0.
        alpha = min(1.0, market_limit_usd / self.confidence_threshold)
        
        # Bayesian Shrinkage: (Prior * Weight) + (Likelihood * (1 - Weight))
        blended_prob = (alpha * sharp_prior_prob) + ((1.0 - alpha) * internal_model_prob)
        
        return float(blended_prob)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 12: Hierarchical Predictive Engine
# ═══════════════════════════════════════════════════════════════════════════════

class MinuteSurvivalModel:
    def __init__(self):
        """
        Stage 1: Volume Model using Cox Proportional Hazards.
        Predicts the survival probability of a player staying in the game at any minute t.
        """
        self.model = lifelines.CoxPHFitter(penalizer=0.1)
        self.is_fitted = False

    def fit(self, df: pd.DataFrame, duration_col: str = 'minutes_played', event_col: str = 'censored_event'):
        """
        Fits the survival model. We model the time until a player is removed 
        from the game permanently due to fouls, blowout, or injury. 
        """
        try:
            self.model.fit(df, duration_col=duration_col, event_col=event_col)
            self.is_fitted = True
            logger.info('CoxPHFitter model fitted successfully.')
        except Exception as e:
            logger.error(f'Failed to fit survival model: {e}')
            raise

    def predict_expected_minutes(self, X_eval: pd.DataFrame, league: str = "NBA") -> np.ndarray:
        """
        Returns the expected minutes (area under the survival curve up to league regulation minutes).
        """
        if not self.is_fitted:
            return np.full(len(X_eval), 25.0)
        from src.config import LEAGUE_CONFIG
        league_key = league.upper() if league else "NBA"
        reg_min = LEAGUE_CONFIG.get(league_key, LEAGUE_CONFIG["NBA"])["regulation_minutes"]
        survival_funcs = self.model.predict_survival_function(X_eval)
        expected_mins = survival_funcs.loc[:int(reg_min)].sum().values
        return expected_mins


def get_nb_objective(alpha: float):
    """
    Returns an XGBoost custom objective function wrapping the C++ quant_core 
    Negative Binomial gradient and hessian calculator.
    """
    def nb_obj(preds: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
        labels = dtrain.get_label()
        if quant_core is None:
            raise ImportError("quant_core C++ extension not loaded.")
        grad, hess = quant_core.calculate_nb_gradient_hessian(preds, labels, alpha)
        return grad, hess
    return nb_obj


# ═══════════════════════════════════════════════════════════════════════════════
# Main Block (direct execution tests)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # This block only runs if you execute this specific file directly from the terminal.
    
    print("\n🏀 RUNNING MONTE CARLO PLAYER PROJECTION 🏀\n")
    
    # Simulating Wembanyama Points: Averages 22.4, Line is 23.5
    results = PlayerProjectionSimulator.simulate_stat_distribution(
        projected_mean=22.4, 
        sportsbook_line=23.5,
        overdispersion_factor=1.35, # High variance player
        n_simulations=10000
    )

    print(f"Played {results['simulations_run']:,} simulated games.")
    print("-" * 40)
    print(f"Probability of hitting the OVER:  {results['prob_over']:.2%}")
    print(f"Probability of hitting the UNDER: {results['prob_under']:.2%}")
    print(f"Fair Odds (Over):                 {results['fair_odds_over']}")
    print("-" * 40)
    print(f"Worst Case Scenario (5th Percentile Floor):     {results['5th_percentile_floor']} points")
    print(f"Best Case Scenario (95th Percentile Ceiling):   {results['95th_percentile_ceiling']} points\n")
