import numpy as np
import pandas as pd
import logging

logger = logging.getLogger("EndogenousSimulationEngine")

class EndogenousGameSimulator:
    def __init__(self, n_iterations: int = 10000):
        self.n_iterations = n_iterations
        
    def simulate_game(self, match_context: dict, players: list) -> pd.DataFrame:
        """
        Simulates a game minute-by-minute for n_iterations.
        match_context: {
            'home_team': 'LAL',
            'away_team': 'DEN',
            'home_ortg': 115.0, # Expected pts per 100 pos
            'away_ortg': 118.0,
            'pace': 100.0       # Possessions per 48 min
        }
        players: list of dicts with:
            - 'player_id'
            - 'team'
            - 'is_starter'
            - 'survival_probs': list/array of 48 probabilities (survival function)
            - 'efficiencies': dict of target -> { 'lambda': 25.0, 'alpha': 0.15 }
              where lambda is points per 48 mins, alpha is dispersion.
              
        Returns a DataFrame of simulated stats per player, averaged over iterations,
        as well as raw simulation matrix for downstream O/U pricing.
        """
        # We process all 10,000 iterations in parallel using NumPy arrays.
        n_iters = self.n_iterations
        n_mins = 48
        n_players = len(players)
        
        # Game State
        home_score = np.zeros(n_iters)
        away_score = np.zeros(n_iters)
        
        # Expected points per minute per team
        home_ppm = (match_context['home_ortg'] * (match_context['pace'] / 100.0)) / 48.0
        away_ppm = (match_context['away_ortg'] * (match_context['pace'] / 100.0)) / 48.0
        
        # Initialize Player states
        # player_active: shape (n_players, n_iters) - True if currently playing
        # player_mins: shape (n_players, n_iters) - Total minutes played so far
        player_active = np.zeros((n_players, n_iters), dtype=bool)
        player_mins = np.zeros((n_players, n_iters), dtype=float)
        
        # Initialize all players as active at minute 1 to allow their survival curve to dictate minutes
        for i, p in enumerate(players):
            player_active[i, :] = True
                
        # To simulate survival efficiently, we convert the survival function 
        # (probability of surviving PAST t) into a discrete hazard rate
        # (probability of exiting exactly at minute t given survived to t-1).
        hazard_rates = np.zeros((n_players, n_mins))
        for i, p in enumerate(players):
            S = np.array(p.get('survival_probs', np.linspace(1, 0, n_mins)))
            S_shifted = np.insert(S[:-1], 0, 1.0) # S(t-1)
            # Hazard = (S(t-1) - S(t)) / S(t-1)
            h = (S_shifted - S) / (S_shifted + 1e-9)
            hazard_rates[i, :] = np.clip(h, 0, 1)

        # Minute-by-Minute State Machine
        for t in range(n_mins):
            # 1. Scoring (Add Poisson noise to expected points per minute)
            home_pts_this_min = np.random.poisson(lam=home_ppm, size=n_iters)
            away_pts_this_min = np.random.poisson(lam=away_ppm, size=n_iters)
            home_score += home_pts_this_min
            away_score += away_pts_this_min
            
            # 2. Blowout Truncation Rule
            # In 4th quarter (t >= 36), if point differential > 20, starters are pulled.
            blowout_mask = (t >= 36) & (np.abs(home_score - away_score) >= 20)
            
            # 3. Process Player Survival and Minutes
            for i, p in enumerate(players):
                # If blowout and player is starter, they are pulled (active = False)
                if p['is_starter']:
                    player_active[i, blowout_mask] = False
                
                # For those currently active, check natural substitution/foul hazard
                # Generate uniform randoms [0, 1)
                u = np.random.uniform(0, 1, size=n_iters)
                exit_mask = u < hazard_rates[i, t]
                player_active[i, exit_mask] = False
                
                # If they were active this minute, add 1 minute to their total
                player_mins[i, :] += player_active[i, :]

        # 4. Convolve Time (T) and Efficiency (lambda) via Negative Binomial
        results = []
        for i, p in enumerate(players):
            simulated_t = player_mins[i, :]
            
            player_res = {
                'player_id': p['player_id'],
                'expected_minutes': np.mean(simulated_t)
            }
            
            for target_key, eff_params in p['efficiencies'].items():
                lam_per_48 = eff_params['lambda']
                alpha = eff_params['alpha']
                
                # mu_i for each iteration based on simulated T
                mu_i = (lam_per_48 / 48.0) * simulated_t
                
                # If alpha > 0, generate Negative Binomial
                # np.random.negative_binomial takes n (successes) and p (prob of success)
                # var = mu + alpha * mu^2
                # p = mu / var = 1 / (1 + alpha * mu)
                # n = mu^2 / (var - mu) = 1 / alpha
                
                var_i = mu_i + alpha * (mu_i ** 2)
                
                # Handle zeros carefully
                valid_mask = (mu_i > 0) & (var_i > mu_i)
                simulated_stats = np.zeros(n_iters)
                
                if np.any(valid_mask):
                    mu_valid = mu_i[valid_mask]
                    var_valid = var_i[valid_mask]
                    p_nb = mu_valid / var_valid
                    n_nb = (mu_valid ** 2) / (var_valid - mu_valid)
                    simulated_stats[valid_mask] = np.random.negative_binomial(n=n_nb, p=p_nb)
                
                # Handle Poisson limit if variance is tight (alpha = 0)
                poisson_mask = (mu_i > 0) & (var_i <= mu_i)
                if np.any(poisson_mask):
                    simulated_stats[poisson_mask] = np.random.poisson(lam=mu_i[poisson_mask])
                
                player_res[f'expected_{target_key}'] = np.mean(simulated_stats)
                player_res[f'{target_key}_simulations'] = simulated_stats
                
            results.append(player_res)
            
        return pd.DataFrame(results)
