import numpy as np
from scipy.optimize import minimize_scalar
import logging
from typing import Dict, List, Optional, Tuple
logger = logging.getLogger('ShinDevigger')

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
            if z >= 1.0:
                return float('inf')
            numerator = np.sqrt(z ** 2 + 4 * (1 - z) * p_implied) - z
            denominator = 2 * (1 - z)
            total += numerator / denominator
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
            logger.warning('Market has no vig or negative vig. Returning raw normalized probabilities.')
            return [p / sum_implied for p in implied_probs]
        try:

            def objective(z):
                return ShinDevigger._shin_equation(z, implied_probs) ** 2
            res = minimize_scalar(objective, bounds=(0.0, 0.999), method='bounded')
            if not res.success or res.fun > 0.0001:
                logger.error("Shin's optimization failed to converge. Falling back to multiplicative de-vigging.")
                return [p / sum_implied for p in implied_probs]
            z = res.x
            true_probs = []
            for p_implied in implied_probs:
                numerator = np.sqrt(z ** 2 + 4 * (1 - z) * p_implied) - z
                denominator = 2 * (1 - z)
                true_probs.append(numerator / denominator)
            return true_probs
        except Exception as e:
            logger.error(f'Error during Shin De-vigging: {e}. Falling back to multiplicative method.')
            return [p / sum_implied for p in implied_probs]
logger = logging.getLogger('MiddleScanner')

class MiddleScanner:

    @staticmethod
    def scan_for_arbitrage(market_data: Dict[str, Dict]) -> List[Dict]:
        """
        Scans a dictionary of market lines from different retail books and flags risk-free middles and pure arbitrage.
        
        Expected structure of market_data:
        {
            'DraftKings': {'line': 8.5, 'over_odds': -120, 'under_odds': +100},
            'FanDuel':    {'line': 10.5, 'over_odds': +110, 'under_odds': -130},
            'PrizePicks': {'line': 9.0, 'over_odds': -137, 'under_odds': -137}
        }
        """
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
        """
        Evaluates if taking the Over on book_over and the Under on book_under creates a profitable scenario.
        """
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
logger = logging.getLogger('StochasticMarketMonitor')

class StochasticMarketMonitor:

    def __init__(self, theta: float=0.1, sigma: float=0.02):
        """
        Models the sharp book implied probability stream using an Ornstein-Uhlenbeck (OU) process 
        with jump diffusion.
        
        dX_t = theta * (mu - X_t)dt + sigma * dW_t + J_t * dN_t
        
        theta: Speed of mean reversion (how fast lines correct back to fair value)
        sigma: Standard deviation of the Wiener process (normal market tick noise)
        """
        self.theta = theta
        self.sigma = sigma

    def detect_overreaction(self, current_implied_prob: float, ou_mean: float, time_step_dt: float=1.0) -> bool:
        """
        Analyzes a sudden change in implied probability and flags if a jump J_t has occurred
        that violates the standard OU process variance bounds.
        
        If an overreaction is detected, the engine flags a mean-reverting opportunity to fade the panic.
        
        current_implied_prob: The latest implied probability of the prop (e.g. 0.65)
        ou_mean: The long-term moving average of the probability (the 'mu' in OU)
        time_step_dt: Time delta since last tick.
        """
        expected_drift = self.theta * (ou_mean - current_implied_prob) * time_step_dt
        volatility_bound = self.sigma * np.sqrt(time_step_dt)
        actual_delta = current_implied_prob - ou_mean
        jump_threshold = abs(expected_drift) + 3 * volatility_bound
        if abs(actual_delta) > jump_threshold:
            logger.warning(f'[JUMP DETECTED] Line gapped by {actual_delta * 100:.2f}%. Exceeds OU noise threshold of {jump_threshold * 100:.2f}%. Flagging for mean-reversion fade.')
            return True
        return False
logger = logging.getLogger('KalmanTracker')

class SharpMoneyTracker:

    def __init__(self, initial_prob: float, process_variance=1e-05, measurement_variance=0.001):
        self.x = initial_prob
        self.P = 1.0
        self.Q = process_variance
        self.R = measurement_variance

    def update(self, measurement: float) -> float:
        """Standard 1D Kalman Filter Predict/Update loop."""
        x_pred = self.x
        P_pred = self.P + self.Q
        K = P_pred / (P_pred + self.R)
        self.x = x_pred + K * (measurement - x_pred)
        self.P = (1 - K) * P_pred
        logger.debug(f'Filtered Prob: {self.x:.4f} (Measurement: {measurement:.4f}, Gain: {K:.4f})')
        return self.x

    def detect_stale_line(self, retail_prob: float, threshold: float=0.02) -> bool:
        """Flags if the retail line is significantly lagging the filtered sharp trajectory."""
        discrepancy = abs(self.x - retail_prob)
        if discrepancy > threshold:
            logger.info(f'🚨 STALE LINE DETECTED: Filtered Sharp ({self.x:.4f}) vs Retail ({retail_prob:.4f})')
            return True
        return False
logger = logging.getLogger('EVEngine')

class EVCalculator:

    @staticmethod
    def calculate_ev(true_prob: float, american_odds: int) -> float:
        """
        Calculates Expected Value (EV) percentage.
        true_prob: The true probability of the event occurring (e.g., 0.55 for 55%)
        american_odds: The odds offered by the retail sportsbook.
        
        Formula: EV = (True_Prob * Decimal_Payout) - 1.0
        """
        if american_odds > 0:
            decimal_payout = american_odds / 100.0 + 1.0
        else:
            decimal_payout = 100.0 / abs(american_odds) + 1.0
        expected_value = true_prob * decimal_payout - 1.0
        return float(expected_value)

    @staticmethod
    def calculate_fractional_kelly(true_prob: float, american_odds: int, fraction: float=0.25) -> float:
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
        """
        Calculates Expected Value against Fixed-Payout DFS platform multipliers.
        
        leg_probabilities: List of true probabilities for each leg.
        platform: String identifier (e.g., 'PrizePicks_2_Power')
        """
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
            logger.warning(f'Unknown DFS platform/slip type: {platform}. Defaulting to 1X.')
            payout = 1.0
        ev = joint_prob * payout - 1.0
        return float(ev)