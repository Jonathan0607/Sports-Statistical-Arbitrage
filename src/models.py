import numpy as np
import pandas as pd
from arch import arch_model
import logging
from copulas.bivariate import Clayton, Gumbel, Frank
import statsmodels.api as sm
from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP
from scipy import stats
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
logger = logging.getLogger('VolatilityTracker')

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
            logger.info('GARCH(1,1) model fitted successfully.')
        except Exception as e:
            logger.error(f'Failed to fit GARCH model: {e}')
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
logger = logging.getLogger('CopulaEngine')

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
            logger.info(f"{self.copula.__class__.__name__} Copula fitted successfully. Theta: {getattr(self.copula, 'theta', 'N/A')}")
        except Exception as e:
            logger.error(f'Failed to fit Copula: {e}')
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
logger = logging.getLogger('ZINBModel')

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
            logger.info('ZINB Model fitted (all zeros).')
            return
        sample_mu = np.mean(non_zero_data)
        sample_var = np.var(non_zero_data)
        self.pi = zeros_count / total_count if total_count > 0 else 0
        self.mu = sample_mu
        if sample_var > sample_mu:
            self.n = sample_mu ** 2 / (sample_var - sample_mu)
        else:
            self.n = 1000000.0

    def condition_parameters(self, blowout_risk: float, pace_factor: float):
        """
        Dynamically shifts the fitted distribution parameters based on categorical game context.
        blowout_risk: 0.0 to 1.0 probability of a blowout.
        pace_factor: Scaling multiplier for game pace (e.g., 1.05 for fast, 0.95 for slow).
        """
        if blowout_risk > 0.8:
            logger.info('High blowout risk detected. Widening tails and increasing zero-inflation.')
            self.pi = min(self.pi + 0.15, 0.99)
            self.n = max(self.n * 0.7, 1.0)
        if pace_factor > 1.05:
            logger.info(f'High pace detected ({pace_factor}). Scaling mean up.')
            self.mu = self.mu * pace_factor
            self.n = max(self.n * 0.9, 1.0)
        elif pace_factor < 0.95:
            logger.info(f'Slow pace detected ({pace_factor}). Scaling mean down.')
            self.mu = self.mu * pace_factor

    def predict_over_probability(self, line: float) -> float:
        """
        Calculates the probability of observing a value > 'line' based on the fitted distribution.
        """
        if self.mu == 0 and self.pi == 1.0:
            return 0.0
        domain = np.arange(0, 150)
        p = self.n / (self.n + self.mu)
        nb_pmf = stats.nbinom.pmf(domain, self.n, p)
        pmf = np.zeros(len(domain))
        pmf[0] = self.pi + (1 - self.pi) * nb_pmf[0]
        pmf[1:] = (1 - self.pi) * nb_pmf[1:]
        cutoff = int(np.floor(line)) + 1
        if cutoff >= len(domain):
            return 0.0
        prob_over = np.sum(pmf[cutoff:])
        return float(prob_over)
logger = logging.getLogger('SGPCorrelationEngine')

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
            logger.warning('Frank Copula approximation not implemented, falling back to independent probability.')
            return u * v
        else:
            raise ValueError(f'Unsupported copula type: {copula_type}')
logger = logging.getLogger('PlayerProjectionModel')

class PlayerProjectionModel:

    def __init__(self):
        """
        Gradient Boosted Tree model for forecasting baseline player expected metrics
        based on tabular game logs and categorical interactions.
        """
        self.xmin_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, objective='reg:squarederror')
        self.xusg_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, objective='reg:squarederror')
        self.is_fitted = False

    def fit(self, X_train: np.ndarray, y_min: np.ndarray, y_usg: np.ndarray):
        """
        Trains the XGBoost models.
        X_train: Matrix of features (e.g., historical stats, matchup strength, categorical flags like 'starting_center_out')
        y_min: Target expected minutes
        y_usg: Target expected usage rate
        """
        try:
            logger.info('Training xMin XGBoost sub-model...')
            self.xmin_model.fit(X_train, y_min)
            logger.info('Training xUSG XGBoost sub-model...')
            self.xusg_model.fit(X_train, y_usg)
            self.is_fitted = True
            logger.info('XGBoost training complete.')
        except Exception as e:
            logger.error(f'Failed to fit XGBoost models: {e}')
            raise

    def predict(self, X_eval: np.ndarray) -> dict:
        """
        Generates point projections for minutes and usage based on the active state matrix.
        """
        if not self.is_fitted:
            logger.warning('Model not fitted. Returning baseline defaults.')
            return {'xMin': 25.0, 'xUSG': 0.2}
        x_min = self.xmin_model.predict(X_eval)
        x_usg = self.xusg_model.predict(X_eval)
        return {'xMin': float(x_min[0]), 'xUSG': float(x_usg[0])}
logger = logging.getLogger('PlayerVolatilityTracker')

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
            logger.warning(f'Insufficient history for {player_id}. Falling back to standard sample variance.')
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
            logger.info(f'[{player_id}] Volatility Regime Variance: {forecasted_var:.4f}')
            return float(forecasted_var)
        except Exception as e:
            logger.error(f'GARCH fitting failed for {player_id}: {e}. Returning sample variance.')
            return float(np.var(history_array))
logger = logging.getLogger('ProbabilityCalibrator')

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
            logger.info('Isotonic Regression curve fitted successfully.')
        except Exception as e:
            logger.error(f'Failed to fit Probability Calibrator: {e}')

    def calibrate(self, raw_prob: float) -> float:
        """
        Returns the calibrated probability.
        """
        if not self.is_fitted:
            logger.warning('Calibrator not fitted. Returning uncalibrated probability.')
            return float(raw_prob)
        calibrated = self.ir.predict([raw_prob])[0]
        return float(calibrated)