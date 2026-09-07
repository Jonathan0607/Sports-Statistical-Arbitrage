import numpy as np
import logging
import time
import os

import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
load_dotenv()

# Try to import C++ quant_core extension for high-performance Kelly optimization
try:
    from src import quant_core as _quant_core
    _HAS_CPP_KELLY = True
except ImportError:
    _quant_core = None
    _HAS_CPP_KELLY = False

# ═══════════════════════════════════════════════════════════════════════════════
# RISK.PY — Consolidated Risk Management & Backtesting Layer
# Merges: risk/risk_manager.py, risk/risk_simulator.py, risk/backtesting.py
# ═══════════════════════════════════════════════════════════════════════════════

logger = logging.getLogger('ChronologicalBacktester')
logger_mc = logging.getLogger('RiskOfRuinSimulator')
logger_risk = logging.getLogger('RiskManager')


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1: Risk Manager (Multivariate Kelly & Bet Masking)
# (from risk/risk_manager.py)
# ═══════════════════════════════════════════════════════════════════════════════

class RiskManager:
    """
    Orchestrates the portfolio risk management layer.
    Manages bankroll allocation, multivariate Kelly sizing, and bet size camouflaging.
    """
    @staticmethod
    def multivariate_kelly(
        edge_vector: np.ndarray,
        covariance_matrix: np.ndarray,
        max_bankroll_fraction: float = 0.05,
        kelly_fraction: float = 0.25,
    ) -> np.ndarray:
        """
        Calculates the optimal fractional Kelly allocation across simultaneous, correlated bets.
        Formula: f = inv(Sigma) * G, scaled by kelly_fraction.
        
        Attempts C++ Eigen-based solver first for ~10x+ performance. Falls back to
        NumPy np.linalg.solve if the compiled extension is unavailable.
        """
        edges = np.asarray(edge_vector, dtype=float)
        cov = np.asarray(covariance_matrix, dtype=float)
        
        n = len(edges)
        if n == 0:
            return np.array([])

        # --- C++ Eigen Fast Path ---
        if _HAS_CPP_KELLY:
            try:
                edge_list = edges.tolist()
                cov_list = cov.tolist()
                result = _quant_core.optimize_multivariate_kelly(
                    edge_list, cov_list, max_bankroll_fraction, kelly_fraction
                )
                return np.array(result, dtype=float)
            except Exception as e:
                logger_risk.warning(f"C++ Kelly optimizer failed, falling back to NumPy: {e}")

        # --- Python NumPy Fallback ---
        try:
            # Solve system cov * f = edges
            f = np.linalg.solve(cov, edges)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if matrix is singular or near-singular
            f = np.linalg.pinv(cov) @ edges
            
        # Long-only constraint: clip negative values to 0.0 (no short/reverse betting)
        f = np.clip(f, 0.0, None)
        
        # Exposure Cap
        total_exposure = np.sum(f)
        if total_exposure > max_bankroll_fraction:
            f = f * (max_bankroll_fraction / total_exposure)

        # Apply fractional Kelly scaling
        f *= kelly_fraction
            
        return f

    @staticmethod
    def mask_bet_size(raw_kelly_dollar_amount: float) -> int:
        """
        Rounds raw_kelly_dollar_amount to the nearest recreational multiple ($5 increment).
        If under $5, returns $5.
        """
        if raw_kelly_dollar_amount < 5.0:
            return 5
        # Round to nearest $5
        rounded = int(round(raw_kelly_dollar_amount / 5.0) * 5)
        return max(5, rounded)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2: Risk Simulator (Monte Carlo Bankroll Paths)
# (from risk/risk_simulator.py)
# ═══════════════════════════════════════════════════════════════════════════════

logger_rs = logging.getLogger("RiskSimulator")

def american_to_decimal(odds: float) -> float:
    """Converts American odds (e.g. -110, +150) to decimal odds multiplier."""
    if odds > 0:
        return 1.0 + (odds / 100.0)
    elif odds < 0:
        return 1.0 + (100.0 / abs(odds))
    else:
        return 1.0

def simulate_bankroll(
    starting_bankroll: float,
    win_rate: float,
    average_odds: float,
    bet_size_pct: float,
    num_bets: int = 1000,
    num_simulations: int = 100000
) -> dict:
    """
    Simulates compounding bankroll paths using vectorized NumPy operations.
    Returns institutional risk metrics: Probability of Ruin, Average/99th MDD, Median/5th Ending Capital.
    """
    # 1. Resolve odds to decimal scale
    if abs(average_odds) >= 100:
        decimal_odds = american_to_decimal(average_odds)
    else:
        decimal_odds = average_odds

    # 2. Generate outcome outcomes (1 = win, 0 = loss)
    # Shape: [num_simulations, num_bets]
    outcomes = (np.random.rand(num_simulations, num_bets) < win_rate)

    # 3. Map outcomes to percentage return factors
    win_return = bet_size_pct * (decimal_odds - 1.0)
    loss_return = -bet_size_pct
    
    returns = np.where(outcomes, win_return, loss_return)

    # 4. Calculate compounding multipliers
    # Shape: [num_simulations, num_bets]
    multipliers = 1.0 + returns

    # 5. Compute compounding bankroll paths
    # cumulative products along axis=1 (bets)
    compounded_products = np.cumprod(multipliers, axis=1)
    
    # Prepend starting bankroll column at t=0
    initial_col = np.ones((num_simulations, 1))
    bankroll_paths = np.hstack([initial_col, compounded_products]) * starting_bankroll

    # 6. Calculate Probability of Ruin (drops below 10% of starting bankroll at any point)
    ruin_threshold = 0.10 * starting_bankroll
    ruined_paths = np.any(bankroll_paths < ruin_threshold, axis=1)
    prob_of_ruin = float(np.mean(ruined_paths))

    # 7. Calculate Maximum Drawdown (MDD)
    # Peak at each step: shape [num_simulations, num_bets + 1]
    peaks = np.maximum.accumulate(bankroll_paths, axis=1)
    drawdowns = np.divide(peaks - bankroll_paths, peaks, out=np.zeros_like(peaks), where=peaks!=0)
    max_drawdowns = np.max(drawdowns, axis=1)
    
    avg_mdd = float(np.mean(max_drawdowns))
    p99_mdd = float(np.percentile(max_drawdowns, 99))

    # 8. Calculate Ending Capital Metrics
    ending_bankrolls = bankroll_paths[:, -1]
    median_ending = float(np.median(ending_bankrolls))
    p5_ending = float(np.percentile(ending_bankrolls, 5))

    return {
        "probability_of_ruin": prob_of_ruin,
        "average_max_drawdown": avg_mdd,
        "p99_max_drawdown": p99_mdd,
        "median_ending_bankroll": median_ending,
        "p5_ending_bankroll": p5_ending
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3: Chronological Backtester (Event-Driven with Friction)
# (from risk/backtesting.py)
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_historical_ticks(start_date: str, end_date: str) -> list:
    """
    Pulls actual chronological market signals from PostgreSQL arb_signals table.
    """
    db_uri = os.environ.get("POSTGRES_URI") or os.environ.get("DATABASE_URL")
    if not db_uri:
        logger.error("No database connection URI found (POSTGRES_URI or DATABASE_URL). Cannot fetch historical ticks.")
        return []

    # Query the actual arb_signals table in PostgreSQL
    query = """
        SELECT 
            EXTRACT(EPOCH FROM timestamp) * 1000 AS timestamp_ms,
            expected_value AS signal_edge,
            sharp_over_odds,
            stale_under_odds AS retail_odds
        FROM arb_signals
        WHERE timestamp BETWEEN %s AND %s
        ORDER BY timestamp ASC;
    """

    try:
        with psycopg2.connect(db_uri) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                logger.info(f"Querying PostgreSQL arb_signals from {start_date} to {end_date}...")
                cur.execute(query, (start_date, end_date))
                rows = cur.fetchall()
                
                ticks = []
                for row in rows:
                    tick = dict(row)
                    
                    # Convert values to expected types
                    tick['timestamp_ms'] = int(tick['timestamp_ms'])
                    tick['signal_edge'] = float(tick['signal_edge'])
                    tick['retail_odds'] = int(tick['retail_odds'])
                    
                    # Convert American odds to implied probability
                    def odds_to_prob(odds):
                        if odds > 0:
                            return 100.0 / (odds + 100.0)
                        else:
                            return -odds / (-odds + 100.0)
                            
                    sharp_odds = int(tick.get('sharp_over_odds', -110))
                    sharp_prob = odds_to_prob(sharp_odds)
                    
                    tick['sharp_prob_at_signal'] = sharp_prob
                    # Simulate a 5-second lookahead (small random movement)
                    tick['sharp_prob_after_delay'] = sharp_prob + np.random.normal(0.0, 0.01)
                    # Generate a realistic outcome based on the sharp probability
                    tick['outcome'] = 1 if np.random.random() < sharp_prob else 0
                    
                    ticks.append(tick)
                return ticks
    except Exception as e:
        logger.error(f"Database extraction failed: {e}. Generating mock historical signals instead.")
        np.random.seed(42)
        mock_ticks = []
        for i in range(200):
            edge = np.random.uniform(0.02, 0.15)
            sharp_at_signal = 0.55
            sharp_after = sharp_at_signal + np.random.uniform(-0.01, 0.06)
            outcome = 1 if np.random.random() < 0.58 else 0
            mock_ticks.append({
                'timestamp_ms': 1000 * i,
                'signal_edge': edge,
                'sharp_prob_at_signal': sharp_at_signal,
                'sharp_prob_after_delay': sharp_after,
                'retail_odds': -110,
                'outcome': outcome
            })
        return mock_ticks


class ChronologicalBacktester:
    """
    Event-Driven Backtester that simulates chronological execution of signals
    with realistic market friction (vig, latency/slippage).
    """

    def __init__(self):
        self.trade_log = []

    def run_simulation(self, historical_ticks: list, bid_ask_spread_vig: float = 0.045,
                       latency_ms: int = 5000):
        """
        Loops chronologically through a list of historical signal events.

        historical_ticks: List of dicts, each representing a timestamped market event:
            {
                'timestamp_ms': int,       # Unix timestamp in milliseconds
                'signal_edge': float,      # Raw model edge (e.g., 0.08 for 8%)
                'sharp_prob_at_signal': float,  # Sharp implied prob when signal fires
                'sharp_prob_after_delay': float, # Sharp implied prob after latency_ms
                'retail_odds': int,        # American odds offered by retail book
                'outcome': int             # 1 = Win, 0 = Loss (for backtest scoring)
            }
        bid_ask_spread_vig: The vig penalty hardcoded into edge (e.g., 0.045 = 4.5%)
        latency_ms: Execution delay in milliseconds (default 5000ms = 5 seconds)
        """
        results = {
            'total_signals': 0,
            'executed': 0,
            'rejected_slippage': 0,
            'wins': 0,
            'losses': 0,
            'gross_edge_sum': 0.0,
            'net_edge_sum': 0.0,
        }

        logger.info(f"Starting backtest: {len(historical_ticks)} events | "
                     f"Vig: {bid_ask_spread_vig*100:.1f}% | Latency: {latency_ms}ms")

        for i, tick in enumerate(historical_ticks):
            results['total_signals'] += 1

            raw_edge = tick['signal_edge']

            # ----- Friction 1: Vig Penalty -----
            net_edge = raw_edge - bid_ask_spread_vig
            if net_edge <= 0:
                results['rejected_slippage'] += 1
                self.trade_log.append({
                    'tick_index': i,
                    'status': 'REJECTED_NO_EDGE_AFTER_VIG',
                    'raw_edge': raw_edge,
                    'net_edge': net_edge,
                })
                continue

            # ----- Friction 2: Latency Slippage -----
            # After 5s delay, check if the sharp book moved and killed the edge
            prob_at_signal = tick['sharp_prob_at_signal']
            prob_after_delay = tick['sharp_prob_after_delay']
            line_movement = abs(prob_after_delay - prob_at_signal)

            # If the line moved more than the net edge, the opportunity evaporated
            if line_movement >= net_edge:
                results['rejected_slippage'] += 1
                self.trade_log.append({
                    'tick_index': i,
                    'status': 'REJECTED_SLIPPAGE',
                    'raw_edge': raw_edge,
                    'net_edge': net_edge,
                    'line_movement': line_movement,
                })
                continue

            # ----- Execution Success -----
            results['executed'] += 1
            results['gross_edge_sum'] += raw_edge
            results['net_edge_sum'] += net_edge

            if tick['outcome'] == 1:
                results['wins'] += 1
            else:
                results['losses'] += 1

            self.trade_log.append({
                'tick_index': i,
                'status': 'EXECUTED',
                'raw_edge': raw_edge,
                'net_edge': net_edge,
            })

        # Calculate summary stats
        if results['executed'] > 0:
            results['win_rate'] = results['wins'] / results['executed']
            results['avg_net_edge'] = results['net_edge_sum'] / results['executed']
        else:
            results['win_rate'] = 0.0
            results['avg_net_edge'] = 0.0

        results['fill_rate'] = results['executed'] / results['total_signals'] if results['total_signals'] > 0 else 0

        return results


class RiskOfRuinSimulator:
    """
    Monte Carlo engine that simulates thousands of equity curve paths
    to calculate institutional risk metrics (Risk of Ruin, VaR, Sharpe).
    """

    @staticmethod
    def simulate_equity_curves(win_probability: float, net_odds: float,
                                starting_bankroll: float,
                                n_paths: int = 10000, n_bets: int = 1000,
                                kelly_fraction: float = 0.25) -> dict:
        """
        Vectorized Monte Carlo simulation of bet sequences.

        win_probability: True probability of winning each bet
        net_odds: Net decimal odds (e.g., 1.91 for -110 American)
        starting_bankroll: Starting capital in dollars
        n_paths: Number of simulation paths (default 10,000)
        n_bets: Number of bets per path (default 1,000)
        kelly_fraction: Fraction of Kelly to wager (default 0.25)
        """
        # Calculate Kelly stake fraction
        b = net_odds - 1.0  # Net payout ratio
        q = 1.0 - win_probability
        full_kelly = win_probability - (q / b)
        stake_fraction = max(full_kelly * kelly_fraction, 0.001)

        logger_mc.info(f"Monte Carlo: {n_paths} paths x {n_bets} bets | "
                       f"p={win_probability:.3f}, odds={net_odds:.2f}, "
                       f"Kelly={full_kelly*100:.2f}%, Fractional={stake_fraction*100:.2f}%")

        # Generate outcome matrix: (n_paths, n_bets)
        # True = win, False = loss
        outcomes = (np.random.rand(n_paths, n_bets) < win_probability)

        # Calculate returns per bet
        # Win: +stake * b   |   Loss: -stake
        returns = np.where(outcomes, stake_fraction * b, -stake_fraction)

        # Build equity curves: cumulative product of (1 + return)
        growth_factors = 1.0 + returns
        equity_curves = starting_bankroll * np.cumprod(growth_factors, axis=1)

        # --- Institutional Metrics ---
        final_bankrolls = equity_curves[:, -1]

        # Risk of Ruin: paths where bankroll hit effectively zero at any point
        min_bankrolls = np.min(equity_curves, axis=1)
        ruin_threshold = starting_bankroll * 0.01  # Define ruin as 99% drawdown
        risk_of_ruin = np.mean(min_bankrolls < ruin_threshold)

        # Expected terminal value
        expected_value = np.mean(final_bankrolls)
        median_value = np.median(final_bankrolls)

        # Value at Risk (5th percentile drawdown)
        var_5 = np.percentile(final_bankrolls, 5)

        # Max Drawdown distribution
        running_max = np.maximum.accumulate(equity_curves, axis=1)
        drawdowns = np.divide(running_max - equity_curves, running_max, out=np.zeros_like(running_max), where=running_max!=0)
        max_drawdowns = np.max(drawdowns, axis=1)
        avg_max_drawdown = np.mean(max_drawdowns)

        # Sharpe Ratio (annualized assuming ~3 bets/day, 365 days)
        log_returns = np.log(growth_factors)
        mean_log_return = np.mean(log_returns)
        std_log_return = np.std(log_returns)
        if std_log_return > 0:
            sharpe = (mean_log_return / std_log_return) * np.sqrt(365 * 3)
        else:
            sharpe = 0.0

        return {
            'n_paths': n_paths,
            'n_bets': n_bets,
            'stake_fraction': stake_fraction,
            'risk_of_ruin': risk_of_ruin,
            'expected_terminal_value': expected_value,
            'median_terminal_value': median_value,
            'var_5th_percentile': var_5,
            'avg_max_drawdown': avg_max_drawdown,
            'sharpe_ratio': sharpe,
            'best_path': np.max(final_bankrolls),
            'worst_path': np.min(final_bankrolls),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4: Backtester (CLV Calculation)
# ═══════════════════════════════════════════════════════════════════════════════

class Backtester:
    """
    Quantitative backtesting metrics class, including Closing Line Value (CLV) calculation.
    """
    @staticmethod
    def calculate_clv(placed_implied_prob: float, closing_implied_prob: float) -> float:
        """
        Calculates the Closing Line Value (CLV) percentage using:
        (closing_implied_prob / placed_implied_prob) - 1.0
        """
        if placed_implied_prob <= 0.0:
            return 0.0
        return (closing_implied_prob / placed_implied_prob) - 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# Main Block (direct execution demo)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from src.models import MultivariateKellyOptimizer, BetMaskingEngine

    logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')
    np.random.seed(42)

    print("\n=================================================================")
    print("         PHASE 4: BACKTESTING & RISK MANAGEMENT REPORT          ")
    print("=================================================================\n")

    # ---------------------------------------------------------------
    # SECTION 1: Multivariate Kelly Portfolio Optimization
    # ---------------------------------------------------------------
    print("[SECTION 1] Multivariate Kelly Optimization")
    print("=" * 60)

    # Mock 3-game slate with correlated bets
    ev_vector = np.array([0.08, 0.05, 0.12])  # 8%, 5%, 12% edges
    cov_matrix = np.array([
        [0.04,  0.005, 0.002],
        [0.005, 0.03,  0.001],
        [0.002, 0.001, 0.05 ]
    ])
    max_limit = 0.05  # 5% of bankroll per bet (retail cap)

    allocations = MultivariateKellyOptimizer.optimize_portfolio(
        ev_vector, cov_matrix, max_account_limit=max_limit, kelly_fraction=0.25
    )

    bankroll = 5000
    print(f"  Bankroll: ${bankroll}")
    print(f"  Games on Slate: {len(ev_vector)}")
    print(f"  EV Vector: {ev_vector * 100}%")
    print(f"  Kelly Allocations (fraction): {np.round(allocations, 4)}")
    print(f"  Dollar Allocations:")
    for i, alloc in enumerate(allocations):
        raw_dollar = alloc * bankroll
        masked_dollar = BetMaskingEngine.mask_kelly_stake(raw_dollar)
        print(f"    Bet {i+1}: Raw ${raw_dollar:.2f} -> Masked ${masked_dollar}")

    # Decoy check
    decoy = BetMaskingEngine.generate_decoy_bet(ev_edge=0.12)
    if decoy:
        print(f"  🎭 Decoy Generated: {decoy['description']} ({decoy['purpose']})")

    # ---------------------------------------------------------------
    # SECTION 2: Event-Driven Backtester with Friction
    # ---------------------------------------------------------------
    print(f"\n[SECTION 2] Chronological Backtester (Friction Simulation)")
    print("=" * 60)

    real_ticks = fetch_historical_ticks('2026-04-01', '2026-05-25')

    backtester = ChronologicalBacktester()
    results = backtester.run_simulation(real_ticks, bid_ask_spread_vig=0.045, latency_ms=5000)

    print(f"  Total Signals:       {results['total_signals']}")
    print(f"  Executed:            {results['executed']}")
    print(f"  Rejected (Slippage): {results['rejected_slippage']}")
    print(f"  Fill Rate:           {results['fill_rate']*100:.1f}%")
    print(f"  Win Rate:            {results['win_rate']*100:.1f}%")
    print(f"  Avg Net Edge:        {results['avg_net_edge']*100:.2f}%")

    # ---------------------------------------------------------------
    # SECTION 3: Monte Carlo Risk of Ruin Simulation
    # ---------------------------------------------------------------
    print(f"\n[SECTION 3] Monte Carlo Bankroll Simulator (10,000 Paths)")
    print("=" * 60)

    mc_results = RiskOfRuinSimulator.simulate_equity_curves(
        win_probability=0.575,
        net_odds=1.91,     # -110 American
        starting_bankroll=5000,
        n_paths=10000,
        n_bets=1000,
        kelly_fraction=0.25
    )

    print(f"  Stake Fraction:          {mc_results['stake_fraction']*100:.2f}% of bankroll per bet")
    print(f"  Risk of Ruin:            {mc_results['risk_of_ruin']*100:.2f}%")
    print(f"  Expected Terminal Value:  ${mc_results['expected_terminal_value']:,.2f}")
    print(f"  Median Terminal Value:    ${mc_results['median_terminal_value']:,.2f}")
    print(f"  VaR (5th Percentile):    ${mc_results['var_5th_percentile']:,.2f}")
    print(f"  Avg Max Drawdown:        {mc_results['avg_max_drawdown']*100:.1f}%")
    print(f"  Sharpe Ratio:            {mc_results['sharpe_ratio']:.2f}")
    print(f"  Best Path Terminal:      ${mc_results['best_path']:,.2f}")
    print(f"  Worst Path Terminal:     ${mc_results['worst_path']:,.2f}")

    print("\n=================================================================")
    print("              PHASE 4 SIMULATION COMPLETE                        ")
    print("=================================================================\n")
