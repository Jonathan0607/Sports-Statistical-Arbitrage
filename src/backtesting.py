import numpy as np
import logging
import time

logger = logging.getLogger('ChronologicalBacktester')
logger_mc = logging.getLogger('RiskOfRuinSimulator')


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
        # 1 = win, 0 = loss
        outcomes = np.random.binomial(1, win_probability, size=(n_paths, n_bets))

        # Calculate returns per bet
        # Win: +stake * b   |   Loss: -stake
        returns = np.where(outcomes == 1, stake_fraction * b, -stake_fraction)

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
        drawdowns = (running_max - equity_curves) / running_max
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


if __name__ == '__main__':
    import sys
    import os
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

    # Generate 200 mock historical signals
    mock_ticks = []
    for i in range(200):
        edge = np.random.uniform(0.02, 0.15)
        sharp_at_signal = 0.55
        # Simulate random line movement during latency window
        sharp_after = sharp_at_signal + np.random.uniform(-0.01, 0.06)
        outcome = 1 if np.random.random() < 0.58 else 0  # 58% base win rate

        mock_ticks.append({
            'timestamp_ms': 1000 * i,
            'signal_edge': edge,
            'sharp_prob_at_signal': sharp_at_signal,
            'sharp_prob_after_delay': sharp_after,
            'retail_odds': -110,
            'outcome': outcome
        })

    backtester = ChronologicalBacktester()
    results = backtester.run_simulation(mock_ticks, bid_ask_spread_vig=0.045, latency_ms=5000)

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
