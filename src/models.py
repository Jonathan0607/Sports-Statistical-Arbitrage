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


# --- Centralized Math and Deviggers (Consolidated from src/execution/__init__.py) ---
from scipy.optimize import minimize_scalar
from typing import Dict, List, Optional, Tuple

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

logger_mask = logging.getLogger('BetMaskingEngine')

class BetMaskingEngine:

    @staticmethod
    def mask_kelly_stake(raw_stake: float) -> int:
        if raw_stake <= 0:
            return 0
        if raw_stake < 50:
            masked = round(raw_stake / 5) * 5
        elif raw_stake <= 200:
            masked = round(raw_stake / 10) * 10
        else:
            masked = round(raw_stake / 25) * 25
        return max(int(masked), 5)

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