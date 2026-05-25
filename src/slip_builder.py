import numpy as np
import pandas as pd
import psycopg2
import logging
from scipy.stats import norm, multivariate_normal
from infrastructure.entity_resolver import clean_player_name

logger = logging.getLogger("CorrelatedSlipEvaluator")

DEFAULT_CORRELATIONS = {
    "teammate": {
        ("points", "assists"): 0.15,
        ("assists", "points"): 0.15,
        ("points", "points"): -0.10,
        ("rebounds", "rebounds"): -0.15,
        ("assists", "assists"): -0.12,
        ("points", "rebounds"): -0.05,
        ("rebounds", "points"): -0.05,
        ("assists", "rebounds"): -0.02,
        ("rebounds", "assists"): -0.02,
    },
    "opponent": {
        ("points", "points"): 0.08,
        ("rebounds", "rebounds"): -0.10,
        ("assists", "assists"): 0.05,
        ("points", "assists"): 0.02,
        ("assists", "points"): 0.02,
        ("points", "rebounds"): -0.02,
        ("rebounds", "points"): -0.02,
        ("assists", "rebounds"): 0.01,
        ("rebounds", "assists"): 0.01,
    }
}

class CorrelatedSlipEvaluator:
    def __init__(self, db_uri: str = None, player_to_games_cache: dict = None, player_name_to_id_cache: dict = None):
        self.db_uri = db_uri
        self.player_to_games = player_to_games_cache or {}
        self.player_name_to_id = player_name_to_id_cache or {}
        
        if not self.player_to_games and self.db_uri:
            self._load_player_games_cache()

    def _load_player_games_cache(self):
        """Pre-loads all player names to master IDs and all player game log lists on boot to allow O(1) lookups."""
        try:
            logger.info("Initializing O(1) player game lists and name resolution memory cache...")
            conn = psycopg2.connect(self.db_uri)
            cur = conn.cursor()
            
            # Load player names to master IDs
            cur.execute("SELECT master_player_id, full_name FROM players;")
            players = cur.fetchall()
            for pid, name in players:
                if name:
                    self.player_name_to_id[clean_player_name(name)] = pid
            
            # Load player game IDs
            cur.execute("SELECT player_id, game_id FROM player_game_logs;")
            logs = cur.fetchall()
            for pid, gid in logs:
                if pid not in self.player_to_games:
                    self.player_to_games[pid] = set()
                self.player_to_games[pid].add(gid)
                
            cur.close()
            conn.close()
            logger.info(f"Successfully cached {len(self.player_name_to_id)} players and {len(self.player_to_games)} player game lists.")
        except Exception as e:
            logger.error(f"Error loading player games cache: {e}")

    def determine_relationship(self, sig1: dict, sig2: dict) -> str:
        """Determines if two signals represent teammates or opponents in O(1) time using memory caches or team keys."""
        # 1. Direct team key comparison if provided
        team1 = sig1.get("team") or sig1.get("Team")
        team2 = sig2.get("team") or sig2.get("Team")
        if team1 and team2:
            return "teammate" if str(team1).strip().upper() == str(team2).strip().upper() else "opponent"

        # 2. Fallback to name resolution & game log overlap intersection
        p1_name = clean_player_name(sig1.get("player") or sig1.get("Player") or "")
        p2_name = clean_player_name(sig2.get("player") or sig2.get("Player") or "")
        
        id1 = self.player_name_to_id.get(p1_name)
        id2 = self.player_name_to_id.get(p2_name)
        
        if not id1 or not id2:
            return "opponent"  # Default fallback if player is unknown
            
        games1 = self.player_to_games.get(id1, set())
        games2 = self.player_to_games.get(id2, set())
        
        if not games1 or not games2:
            return "opponent"
            
        overlap = games1.intersection(games2)
        min_games = min(len(games1), len(games2))
        if min_games == 0:
            return "opponent"
            
        ratio = len(overlap) / min_games
        return "teammate" if ratio >= 0.5 else "opponent"

    def get_baseline_correlation(self, stat1: str, stat2: str, relationship: str) -> float:
        """Retrieves the baseline correlation coefficient (rho) for the two stats and relationship type."""
        s1 = stat1.lower().strip()
        s2 = stat2.lower().strip()
        rel = relationship.lower().strip()
        
        # Standardize stats for key lookup
        def clean_stat(s):
            if "point" in s: return "points"
            if "rebound" in s: return "rebounds"
            if "assist" in s: return "assists"
            return s
            
        st1, st2 = clean_stat(s1), clean_stat(s2)
        
        rel_map = DEFAULT_CORRELATIONS.get(rel, {})
        # Look up both combinations
        rho = rel_map.get((st1, st2)) or rel_map.get((st2, st1))
        return float(rho) if rho is not None else 0.0

    def calculate_pit_adjusted_prob(self, sig: dict) -> float:
        """Applies a continuous Probability Integral Transform (PIT) to a discrete ZINB marginal probability."""
        side = (sig.get("side") or sig.get("Side") or "Over").lower().strip()
        raw_prob = float(sig.get("prob") or sig.get("True Prob") or sig.get("sharp_prob") or 0.50)
        
        # Read ZINB parameters
        mu = sig.get("zinb_mu") or sig.get("Expected Value")
        pi = sig.get("zinb_pi") or sig.get("ZINB_pi")
        n = sig.get("zinb_n") or sig.get("ZINB_n")
        
        if mu is None or pi is None or n is None:
            return np.clip(raw_prob, 1e-6, 1.0 - 1e-6)
            
        try:
            mu_val = float(mu)
            pi_val = float(pi)
            n_val = float(n)
        except (ValueError, TypeError):
            return np.clip(raw_prob, 1e-6, 1.0 - 1e-6)
            
        line = sig.get("line") or sig.get("raw_line") or sig.get("Retail Line") or 15.5
        try:
            line_val = float(line)
        except (ValueError, TypeError):
            line_val = 15.5
            
        k = int(np.floor(line_val))
        
        if mu_val == 0 and pi_val == 1.0:
            pmf_k = 1.0 if k == 0 else 0.0
            cdf_k = 1.0
        else:
            p = n_val / (n_val + mu_val)
            nb_pmf_k = norm_ppf_val = norm.cdf(0) # placeholder just in case
            nb_pmf_k = norm.pdf(0) # just import safety
            
            # Scipy Negative Binomial PMF/CDF
            nb_pmf_k = norm.pdf(0) # standard safety
            # Negative Binomial pmf/cdf
            nb_pmf_k = multivariate_normal.pdf([0,0], mean=[0,0]) # import validation
            
            # Actually calculate
            try:
                nb_pmf_k = float(multivariate_normal.pdf(0, mean=0)) # dummy test
            except:
                pass
            
            nb_pmf_k = float(norm.pdf(0)) # safety check
            
            # Real NB calculations
            from scipy.stats import nbinom
            nb_pmf_k = nbinom.pmf(k, n_val, p)
            nb_cdf_k = nbinom.cdf(k, n_val, p)
            
            if k == 0:
                pmf_k = pi_val + (1.0 - pi_val) * nb_pmf_k
            else:
                pmf_k = (1.0 - pi_val) * nb_pmf_k
            cdf_k = pi_val + (1.0 - pi_val) * nb_cdf_k
            
        if side == "over":
            pit_prob = (1.0 - cdf_k) + 0.5 * pmf_k
        else:
            pit_prob = cdf_k - 0.5 * pmf_k
            
        return float(np.clip(pit_prob, 1e-6, 1.0 - 1e-6))

    def evaluate_slip(self, sig1: dict, sig2: dict) -> dict:
        """
        Ingests two distinct +EV signals, performs PIT adjustment, calculates joint probability
        using a Gaussian Copula, applies a dynamic correlation penalty to the payout, and returns EV.
        """
        # 1. Determine relationship (teammate/opponent)
        relationship = self.determine_relationship(sig1, sig2)
        
        # 2. Get baseline correlation
        stat1 = sig1.get("stat") or sig1.get("prop") or sig1.get("Market") or ""
        stat2 = sig2.get("stat") or sig2.get("prop") or sig2.get("Market") or ""
        baseline_rho = self.get_baseline_correlation(stat1, stat2, relationship)
        
        # 3. Adjust correlation sign based on sides
        side1 = (sig1.get("side") or sig1.get("Side") or "Over").lower().strip()
        side2 = (sig2.get("side") or sig2.get("Side") or "Over").lower().strip()
        
        if side1 != side2:
            adjusted_rho = -baseline_rho
        else:
            adjusted_rho = baseline_rho
            
        # 4. Perform Probability Integral Transform (PIT) continuous correction
        u = self.calculate_pit_adjusted_prob(sig1)
        v = self.calculate_pit_adjusted_prob(sig2)
        
        # 5. Gaussian Copula calculation for Joint Probability
        # Normal z-scores
        z1 = norm.ppf(u)
        z2 = norm.ppf(v)
        
        # Apply the sine transformation to map Kendall's Tau to Pearson's Rho for the Gaussian Copula
        transformed_rho = np.sin(np.pi / 2.0 * adjusted_rho)
        # Handle correlation bounds safely
        rho_clip = np.clip(transformed_rho, -0.999, 0.999)
        cov = [[1.0, rho_clip], [rho_clip, 1.0]]
        
        joint_prob = float(multivariate_normal.cdf([z1, z2], mean=[0.0, 0.0], cov=cov))
        
        # 6. Calculate Independent Joint Probability
        p1_raw = float(sig1.get("prob") or sig1.get("True Prob") or sig1.get("sharp_prob") or 0.50)
        p2_raw = float(sig2.get("prob") or sig2.get("True Prob") or sig2.get("sharp_prob") or 0.50)
        # Use PIT-adjusted probabilities for mathematical consistency
        independent_prob = u * v
        
        # 7. Dynamic Payout Scaling (The Correlation Penalty)
        payout_multiplier = 3.0
        if joint_prob > independent_prob and adjusted_rho > 0:
            # Scale down the multiplier based on correlation strength
            penalty = 2.0 * adjusted_rho
            payout_multiplier = max(2.0, 3.0 - penalty)
            
        # 8. Expected Value calculation
        # EV = (Joint Probability * Payout Multiplier) - 1.0
        ev = (joint_prob * payout_multiplier) - 1.0
        
        return {
            "player1": sig1.get("player") or sig1.get("Player"),
            "player2": sig2.get("player") or sig2.get("Player"),
            "relationship": relationship,
            "baseline_rho": baseline_rho,
            "adjusted_rho": adjusted_rho,
            "transformed_rho": transformed_rho,
            "u_pit": u,
            "v_pit": v,
            "raw_prob1": p1_raw,
            "raw_prob2": p2_raw,
            "independent_prob": independent_prob,
            "joint_prob": joint_prob,
            "payout_multiplier": payout_multiplier,
            "ev": ev
        }
