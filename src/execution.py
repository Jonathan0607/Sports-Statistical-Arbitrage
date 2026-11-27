import os
from dotenv import load_dotenv
load_dotenv()

import sys
import json
import time
import hashlib
import asyncio
import logging
import urllib.request
import urllib.error
import uuid
import numpy as np
import pandas as pd
import psycopg2
from pathlib import Path
from scipy.stats import norm, multivariate_normal
from datetime import datetime, timezone
from typing import Dict, List, Tuple

# Dynamic project root path resolution
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infrastructure.entity_resolver import clean_player_name
from infrastructure.redis_client import RedisClient
from src.config import LEAGUE_CONFIG

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from datetime import tzinfo, timedelta
    class ZoneInfo(tzinfo):
        def __init__(self, name):
            self.name = name
        def utcoffset(self, dt):
            return timedelta(hours=-5)
        def tzname(self, dt):
            return self.name
        def dst(self, dt):
            return timedelta(0)

# ═══════════════════════════════════════════════════════════════════════════════
# EXECUTION.PY — Consolidated Execution Layer
# Merges: execution/slip_builder.py, execution/sgp_worker.py, execution/discord_notifier.py
# ═══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

import threading
import urllib.request
import json
import unittest.mock


def _discord_alerts_suppressed() -> bool:
    """Returns True in pytest/disabled-alert runs unless HTTP/thread transport is mocked."""
    disable_flag = os.getenv("DISABLE_DISCORD_ALERTS", "").strip().lower() in ("1", "true", "yes", "on")
    if not (disable_flag or os.getenv("PYTEST_CURRENT_TEST")):
        return False
    return not (
        isinstance(urllib.request.urlopen, unittest.mock.Base)
        or isinstance(threading.Thread, unittest.mock.Base)
    )


class DiscordErrorHandler(logging.Handler):
    def __init__(self, webhook_url):
        super().__init__()
        self.webhook_url = webhook_url
        self.error_cache = {}           # Tracks recently emitted errors by fingerprint
        self.cooldown_seconds = 3600    # 1-hour cooldown per unique error signature

    def emit(self, record):
        if record.levelno < logging.ERROR or _discord_alerts_suppressed():
            return
            
        try:
            import hashlib
            import time as _time

            log_entry = self.format(record)
            
            # Fingerprint this error by module + line + message to deduplicate
            error_signature = f"{record.module}_{record.lineno}_{record.getMessage()}"
            error_hash = hashlib.md5(error_signature.encode()).hexdigest()
            
            current_time = _time.time()
            last_seen = self.error_cache.get(error_hash, 0)
            
            # If we've seen this exact error recently, silently drop it
            if current_time - last_seen < self.cooldown_seconds:
                return
                
            self.error_cache[error_hash] = current_time
            
            exc_info = ""
            if record.exc_info:
                import traceback
                exc_info = "".join(traceback.format_exception(*record.exc_info))
                
            desc = log_entry
            if exc_info:
                desc = f"{desc}\n\n**Traceback:**\n```python\n{exc_info[:1500]}\n```"
            else:
                desc = f"```\n{desc[:1900]}\n```"

            payload = {
                "embeds": [{
                    "title": f"⚠️ System {record.levelname} Detected",
                    "color": 15158332,
                    "description": desc,
                    "fields": [
                        {"name": "Logger", "value": str(record.name), "inline": True},
                        {"name": "File", "value": f"{record.pathname}:{record.lineno}", "inline": True},
                        {"name": "Function", "value": str(record.funcName), "inline": True}
                    ],
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }]
            }
            
            def send_error():
                try:
                    if _discord_alerts_suppressed():
                        return
                    req = urllib.request.Request(
                        self.webhook_url,
                        data=json.dumps(payload).encode('utf-8'),
                        headers={'Content-Type': 'application/json', 'User-Agent': 'Mozilla/5.0'}
                    )
                    with urllib.request.urlopen(req, timeout=5.0) as response:
                        pass
                except Exception:
                    pass
                    
            threading.Thread(target=send_error, daemon=True).start()
        except Exception:
            pass

discord_webhook = os.getenv("DISCORD_WEBHOOK_URL")
_disable_flag = os.getenv("DISABLE_DISCORD_ALERTS", "").strip().lower() in ("1", "true", "yes", "on")
if discord_webhook and not (_disable_flag or os.getenv("PYTEST_CURRENT_TEST")):
    handler = DiscordErrorHandler(discord_webhook)
    handler.setLevel(logging.ERROR)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1: Correlated Slip Evaluator (SGP Pricing Engine)
# (from execution/slip_builder.py)
# ═══════════════════════════════════════════════════════════════════════════════

logger_slip = logging.getLogger("CorrelatedSlipEvaluator")

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
            logger_slip.info("Initializing O(1) player game lists and name resolution memory cache...")
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
            logger_slip.info(f"Successfully cached {len(self.player_name_to_id)} players and {len(self.player_to_games)} player game lists.")
        except Exception as e:
            logger_slip.error(f"Error loading player games cache: {e}")

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

        # Transform rho for copula covariance
        transformed_rho = float(np.sin(np.pi / 2.0 * adjusted_rho))
            
        # 4. Perform Probability Integral Transform (PIT) continuous correction
        u = self.calculate_pit_adjusted_prob(sig1)
        v = self.calculate_pit_adjusted_prob(sig2)
        
        # 5. Dynamic Copula calculation for Joint Probability
        if side1 == "over" and side2 == "over" and adjusted_rho > 0:
            # Gumbel Copula for upper-tail dependence
            tau = max(1e-5, min(adjusted_rho, 0.99))
            theta = 1.0 / (1.0 - tau)
            theta = max(1.0, theta)
            u_g = -np.log(u)
            v_g = -np.log(v)
            joint_prob = float(np.exp(-((u_g**theta + v_g**theta)**(1.0/theta))))
        elif side1 == "under" and side2 == "under" and adjusted_rho > 0:
            # Clayton Copula for lower-tail dependence
            tau = max(1e-5, min(adjusted_rho, 0.99))
            theta = 2.0 * tau / (1.0 - tau)
            theta = max(1e-5, theta)
            joint_prob = float((u**(-theta) + v**(-theta) - 1.0)**(-1.0/theta))
        else:
            # Gaussian Copula for mixed/symmetric or negative dependence
            z1 = norm.ppf(u)
            z2 = norm.ppf(v)
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

        # 9. Exact Analytical Binary Covariance (\Sigma)
        # \Sigma_ii = p_i * (1 - p_i)
        # \Sigma_ij = p_ij - (p_i * p_j)
        var1 = u * (1.0 - u)
        var2 = v * (1.0 - v)
        cov_12 = joint_prob - (u * v)
        binary_cov = np.array([
            [var1, cov_12],
            [cov_12, var2]
        ])

        # 10. Pass to Multivariate Kelly Optimizer
        from src.risk import RiskManager
        # Approximate marginal EVs for the legs (PrizePicks 2-leg pays ~1.732x per leg)
        marginal_payout = np.sqrt(payout_multiplier)
        ev1 = (u * marginal_payout) - 1.0
        ev2 = (v * marginal_payout) - 1.0
        expected_returns = np.array([ev1, ev2])
        
        kelly_fractions = RiskManager.multivariate_kelly(expected_returns, binary_cov)
        
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
            "ev": ev,
            "binary_cov": binary_cov.tolist(),
            "kelly_fractions": kelly_fractions.tolist()
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2: SGP Worker (Execution Loop)
# (from execution/sgp_worker.py)
# ═══════════════════════════════════════════════════════════════════════════════

logger_sgp = logging.getLogger("SGPWorker")

def are_players_in_same_game(evaluator: CorrelatedSlipEvaluator, sig1: dict, sig2: dict) -> bool:
    """Helper to check if two players represent teammates or opponents in the same game."""
    p1 = clean_player_name(sig1.get("player") or sig1.get("Player") or "")
    p2 = clean_player_name(sig2.get("player") or sig2.get("Player") or "")
    if not p1 or not p2 or p1 == p2:
        return False
        
    # Check if team abbreviations are specified and match
    team1 = sig1.get("team") or sig1.get("Team")
    team2 = sig2.get("team") or sig2.get("Team")
    if team1 and team2:
        if str(team1).strip().upper() == str(team2).strip().upper():
            return True
            
    # Fallback to game log overlap intersection
    id1 = evaluator.player_name_to_id.get(p1)
    id2 = evaluator.player_name_to_id.get(p2)
    if id1 and id2:
        games1 = evaluator.player_to_games.get(id1, set())
        games2 = evaluator.player_to_games.get(id2, set())
        if len(games1.intersection(games2)) > 0:
            return True
            
    return False

def get_dedup_key(sig1: dict, sig2: dict) -> str:
    """Generates a canonical sorted key for a 2-leg parlay to prevent duplicate processing."""
    p1 = clean_player_name(sig1.get("player") or sig1.get("Player") or "")
    s1 = (sig1.get("stat") or sig1.get("prop") or sig1.get("Market") or "").lower().strip()
    d1 = (sig1.get("side") or sig1.get("Side") or "").lower().strip()
    l1 = str(sig1.get("line") or sig1.get("raw_line") or sig1.get("Retail Line") or "")
    
    p2 = clean_player_name(sig2.get("player") or sig2.get("Player") or "")
    s2 = (sig2.get("stat") or sig2.get("prop") or sig2.get("Market") or "").lower().strip()
    d2 = (sig2.get("side") or sig2.get("Side") or "").lower().strip()
    l2 = str(sig2.get("line") or sig2.get("raw_line") or sig2.get("Retail Line") or "")
    
    leg1_str = f"{p1}:{s1}:{d1}:{l1}"
    leg2_str = f"{p2}:{s2}:{d2}:{l2}"
    
    # Sort alphabetically to keep it commutative
    sorted_legs = sorted([leg1_str, leg2_str])
    return f"processed:slip:{sorted_legs[0]}:{sorted_legs[1]}"

async def run_worker():
    from src.models import BetMaskingEngine

    db_uri = os.getenv("POSTGRES_URI", "postgresql://postgres:password@localhost:5432/sports_db")
    logger_sgp.info(f"Starting SGP Worker with db: {db_uri}")
    
    evaluator = CorrelatedSlipEvaluator(db_uri=db_uri)
    tracker = GlobalExposureTracker()
    queue_manager = SlipQueueManager("execution_queue:prizepicks")
    redis_client = RedisClient().get_client()
    
    # Check cache status
    if not evaluator.player_name_to_id:
        logger_sgp.warning("Database cache is empty. Active database connection might be offline or unseeded.")
        
    logger_sgp.info("Decoupled SGP Worker initialized. Entering execution loop...")
    
    while True:
        try:
            # 1. Fetch all active signals from Redis
            keys = redis_client.keys("signal:*")
            if not keys:
                await asyncio.sleep(5)
                continue
                
            logger_sgp.debug(f"Retrieved {len(keys)} active signals from Redis.")
            
            signals = []
            for key in keys:
                data = redis_client.hgetall(key)
                if data:
                    signals.append(data)
                    
            # 2. Group signals by Retail Book
            # PrizePicks signals will have "Book" == "PrizePicks" (or keys starting with "signal:pp:")
            pp_signals = []
            
            for sig in signals:
                book = sig.get("Book") or ""
                if "prizepicks" in book.lower() or "pp" in book.lower():
                    pp_signals.append(sig)
            # 3. Process SGP candidates for PrizePicks
            # Note: The request focuses on the list 'execution_queue:prizepicks'
            if len(pp_signals) >= 2:
                for i in range(len(pp_signals)):
                    for j in range(i + 1, len(pp_signals)):
                        sig1 = pp_signals[i]
                        sig2 = pp_signals[j]
                        
                        # Verify they are in the same game
                        if not are_players_in_same_game(evaluator, sig1, sig2):
                            continue
                            
                        # Deduplication check: check if we already processed this slip recently
                        dedup_key = get_dedup_key(sig1, sig2)
                        if redis_client.exists(dedup_key):
                            continue
                            
                        # Evaluate slip SGP EV
                        res = evaluator.evaluate_slip(sig1, sig2)
                        ev = res.get("ev", 0.0)
                        
                        if ev > 0.0:
                            logger_sgp.info(
                                f"🔥 Detected +EV Same Game Parlay on PrizePicks: "
                                f"{res['player1']} ({sig1['Market']}) & {res['player2']} ({sig2['Market']}) | "
                                f"EV: {ev * 100.0:.2f}%"
                            )
                            
                            # Calculate Kelly Stake using Analytical Binary Covariance
                            # evaluate_slip now computes the binary covariance matrix and routes it through
                            # RiskManager.multivariate_kelly, returning the optimal fractions for the legs.
                            kelly_fractions = res.get("kelly_fractions", [0.0, 0.0])
                            # Conservatively size the parlay based on the minimum optimal allocation of its legs
                            kelly_f = min(kelly_fractions)
                            
                            # The multivariate_kelly function already applied quarter-kelly scaling (0.25)
                            kelly_fraction = min(kelly_f, 0.05)  # Cap at 5%
                            recommended_stake = kelly_fraction * 50000.0
                            
                            # Determine league from signals
                            league = (
                                sig1.get("league") or sig2.get("league") or
                                sig1.get("sport") or sig2.get("sport") or
                                sig1.get("Sport") or sig2.get("Sport") or
                                "NBA"
                            ).strip().upper()
                            if league not in LEAGUE_CONFIG:
                                league = "NBA"
                            max_retail_limit = LEAGUE_CONFIG[league]["max_retail_bet_limit"]
                            recommended_stake = min(recommended_stake, max_retail_limit)

                            # Run exposure check & scale
                            player_names = [res["player1"], res["player2"]]
                            # Extract slate_id or game_date from signals if available
                            game_date = sig1.get("game_date") or sig2.get("game_date") or sig1.get("game_id") or sig2.get("game_id")
                            slate_id = sig1.get("slate_id") or sig2.get("slate_id")
                            identifier = game_date or slate_id
                            
                            scaled_stake, is_allowed = tracker.check_and_scale_stake(player_names, recommended_stake, identifier=identifier)
                            
                            if not is_allowed:
                                logger_sgp.info(f"Slip rejected due to player exposure limits: {player_names}")
                                # Mark as processed so we don't spam warnings
                                redis_client.setex(dedup_key, 1800, "rejected")
                                continue
                                
                            # Apply dynamic league bet limit cap
                            scaled_stake = min(scaled_stake, max_retail_limit)

                            # Apply Bet Masking/Camouflage roundings with dynamic league ceiling
                            masked_stake = BetMaskingEngine.mask_kelly_stake(scaled_stake, league=league)
                            masked_stake = min(masked_stake, int(max_retail_limit))
                            if masked_stake > scaled_stake:
                                masked_stake = max(5, int(scaled_stake))
                                
                            if masked_stake < 5:
                                logger_sgp.info(f"Slip stake scaled below sportsbook minimum $5: ${scaled_stake:.2f} (masked: ${masked_stake})")
                                continue
                                
                            # Formulate Execution Payload
                            payload = queue_manager.formulate_execution_payload(sig1, sig2, masked_stake, ev, league=league)
                            
                            # Commit risk exposure to Redis
                            tracker.commit_exposure(player_names, masked_stake, identifier=identifier)
                            
                            # Push payload to Redis execution list
                            queue_manager.enqueue_slip(payload)
                            
                            # Set deduplication key in Redis with 1-hour expiry (3600 seconds)
                            redis_client.setex(dedup_key, 3600, "enqueued")
                            
        except Exception as err:
            logger_sgp.error(f"Error in SGP Worker loop: {err}", exc_info=True)
            
        await asyncio.sleep(5)

class GlobalExposureTracker:
    MAX_EXPOSURE_PER_PLAYER = 500.00

    def __init__(self, max_exposure_per_player: float = 500.00):
        self.max_exposure = max_exposure_per_player
        self.redis_client = RedisClient().get_client()
        self.tz = ZoneInfo("America/New_York")

    def get_current_date_est(self) -> str:
        """Returns the current date in YYYY-MM-DD format based strictly on America/New_York timezone."""
        return datetime.now(self.tz).strftime("%Y-%m-%d")

    def get_slate_id(self, game_date: str = None) -> str:
        """
        Returns a standardized slate ID or game date.
        If game_date is provided (e.g. YYYY-MM-DD), it uses the date portion.
        Otherwise, it shifts America/New_York time back by 4 hours so the sports slate rollover occurs at 4:00 AM EST.
        """
        if game_date:
            return game_date[:10]
        # Shift back by 4 hours to group nocturnal games on the same slate
        from datetime import timedelta
        slate_dt = datetime.now(self.tz) - timedelta(hours=4)
        return slate_dt.strftime("%Y-%m-%d")

    def _get_key(self, player_name: str, identifier: str) -> str:
        normalized_name = clean_player_name(player_name)
        return f"exposure:player:{normalized_name}:{identifier}"

    def get_current_exposures(self, player_names: List[str], date_str: str = None, identifier: str = None) -> List[float]:
        """Queries Redis to get the exposures for a list of players for a specific slate/identifier."""
        id_val = identifier or date_str or self.get_slate_id()
            
        pipe = self.redis_client.pipeline()
        for player in player_names:
            pipe.get(self._get_key(player, id_val))
        results = pipe.execute()
        
        exposures = []
        for res in results:
            if res is None:
                exposures.append(0.0)
            else:
                try:
                    exposures.append(float(res))
                except (ValueError, TypeError):
                    exposures.append(0.0)
        return exposures

    def check_and_scale_stake(self, player_names: List[str], recommended_stake: float, date_str: str = None, identifier: str = None) -> Tuple[float, bool]:
        """
        Validates exposure caps and scales down stake if it exceeds limits.
        If a player is already at or above cap, returns (0.0, False) meaning rejected.
        Otherwise returns (scaled_stake, True) meaning accepted (possibly scaled).
        """
        if recommended_stake <= 0:
            return 0.0, False
            
        id_val = identifier or date_str or self.get_slate_id()
            
        exposures = self.get_current_exposures(player_names, identifier=id_val)
        
        remaining_limits = []
        for player, exp in zip(player_names, exposures):
            rem = self.max_exposure - exp
            if rem <= 0.001:  # cap already hit
                logger_sgp.warning(
                    f"Slip rejected: Player '{player}' already at/above max exposure limit of {self.max_exposure:.2f} "
                    f"(Current: {exp:.2f})"
                )
                return 0.0, False
            remaining_limits.append(rem)
            
        # The limiting factor is the minimum remaining exposure across all players in the parlay
        min_remaining = min(remaining_limits)
        
        # Scale down if recommended stake exceeds the minimum remaining capacity
        scaled_stake = min(recommended_stake, min_remaining)
        
        return scaled_stake, True

    def commit_exposure(self, player_names: List[str], stake: float, date_str: str = None, identifier: str = None) -> bool:
        """Atomically increments the exposure of all listed players in Redis."""
        if stake <= 0:
            return False
            
        id_val = identifier or date_str or self.get_slate_id()
            
        # Default key expiry to 48 hours (172800 seconds) to prevent midnight resets on slates
        ttl_seconds = 172800
            
        pipe = self.redis_client.pipeline()
        for player in player_names:
            key = self._get_key(player, id_val)
            pipe.incrbyfloat(key, stake)
            pipe.expire(key, ttl_seconds)
        pipe.execute()
        
        logger_sgp.info(f"Committed exposure of ${stake:.2f} for players: {player_names} on slate/date {id_val}")
        return True


class SlipQueueManager:
    def __init__(self, queue_name: str = "execution_queue:prizepicks"):
        self.queue_name = queue_name
        self.redis_client = RedisClient().get_client()

    def formulate_execution_payload(self, leg1: dict, leg2: dict, risk_adjusted_stake: float, ev_edge: float, league: str = "NBA") -> dict:
        """
        Constructs a structured Execution Payload dictionary containing:
        - player name, stat category, side, line, and target_execution_line for each leg
        - risk adjusted stake (capped by league limit)
        - EV edge
        - metadata (slip ID, league, timestamp)
        """
        league_key = league.upper() if league else "NBA"
        max_limit = LEAGUE_CONFIG.get(league_key, LEAGUE_CONFIG["NBA"])["max_retail_bet_limit"]
        capped_stake = min(float(risk_adjusted_stake), float(max_limit))

        def format_leg(leg: dict) -> dict:
            player = leg.get("player") or leg.get("Player")
            stat = leg.get("stat") or leg.get("prop") or leg.get("Market")
            side = leg.get("side") or leg.get("Side")
            line = leg.get("line") or leg.get("raw_line") or leg.get("Retail Line")
            
            val = float(line) if line is not None else None
            return {
                "player": player,
                "stat": stat,
                "side": side,
                "line": val,
                "target_execution_line": val
            }
            
        payload = {
            "slip_id": str(uuid.uuid4()),
            "league": league_key,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "legs": [
                format_leg(leg1),
                format_leg(leg2)
            ],
            "risk_adjusted_stake": float(capped_stake),
            "ev_edge": float(ev_edge)
        }
        return payload

    def enqueue_slip(self, payload: dict) -> bool:
        """Pushes the payload dict to the Redis list using LPUSH."""
        try:
            serialized = json.dumps(payload)
            self.redis_client.lpush(self.queue_name, serialized)
            logger_sgp.info(f"Enqueued execution payload to Redis list '{self.queue_name}': {payload['slip_id']}")
            return True
        except Exception as e:
            logger_sgp.error(f"Failed to enqueue execution payload: {e}")
            return False


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3: Discord Notifier (Alert Dispatch)
# (from execution/discord_notifier.py)
# ═══════════════════════════════════════════════════════════════════════════════

logger_discord = logging.getLogger("DiscordNotifier")

QUEUE_NAME = "execution_queue:prizepicks"
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")
MAX_ALERT_AGE_SECONDS = 20  # Drop alerts older than this to avoid stale line drift

# Liquidity Filter Settings
MAX_ANCHOR_SPREAD = 0.10    # Max 10% bid-ask spread on anchors
MIN_ANCHOR_VOLUME = 500.0   # Min $500 daily volume/open interest on anchors
DEDUP_TTL_SECONDS = 300     # Suppress duplicate alerts for 5 minutes

def get_semantic_hash(legs: list) -> str:
    """
    Generates a unique, order-invariant hash based on leg properties to deduplicate slips.
    """
    normalized_legs = []
    for leg in legs:
        player = str(leg.get("player", "")).strip().lower()
        stat = str(leg.get("stat", "")).strip().lower()
        side = str(leg.get("side", "")).strip().lower()
        line = str(leg.get("target_execution_line", 0.0))
        normalized_legs.append(f"{player}:{stat}:{side}:{line}")
    
    # Sort to make order-invariant
    normalized_legs.sort()
    legs_string = "|".join(normalized_legs)
    return hashlib.md5(legs_string.encode('utf-8')).hexdigest()

def passes_liquidity_filter(legs: list) -> bool:
    """
    Filters out phantom liquidity edges from thin prediction or sharp markets.
    """
    for leg in legs:
        spread = leg.get("anchor_spread", 0.0)
        volume = leg.get("anchor_volume", 1000.0)  # Default high if not populated
        
        if spread > MAX_ANCHOR_SPREAD:
            logger_discord.warning(f"Skipping slip: Anchor spread ({spread:.2%}) exceeds limit ({MAX_ANCHOR_SPREAD:.2%}) for leg {leg.get('player')}")
            return False
            
        if volume < MIN_ANCHOR_VOLUME:
            logger_discord.warning(f"Skipping slip: Anchor volume (${volume:.2f}) below threshold (${MIN_ANCHOR_VOLUME:.2f}) for leg {leg.get('player')}")
            return False
    return True

def format_discord_embed(payload: dict) -> dict:
    """
    Unpacks payload and maps details into a clean Discord Embed payload structure.
    """
    ev_edge = payload.get("ev_edge", 0.0)
    kelly = payload.get("kelly_fraction", 0.0)
    stake = payload.get("risk_adjusted_stake", 5.0)
    slip_id = payload.get("slip_id", "unknown")
    
    # Color coding: Green for premium edges, Orange for regular edges
    color = 65280 if ev_edge >= 0.05 else 16753920
    
    embed = {
        "title": "🚨 +EV Arbitrage Opportunity Detected",
        "description": f"**Slip ID**: `{slip_id}`\n**Model Edge**: `{ev_edge:.2%}` | **Recommended Stake**: `${stake:.2f}` (Kelly: `{kelly:.2%}`)",
        "color": color,
        "fields": [],
        "timestamp": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(payload.get("timestamp", time.time()))),
        "footer": {
            "text": "Antigravity StatArb Alert System"
        }
    }
    
    for idx, leg in enumerate(payload.get("legs", [])):
        player = leg.get("player")
        stat = leg.get("stat")
        side = str(leg.get("side", "")).upper()
        line = leg.get("target_execution_line")
        
        embed["fields"].append({
            "name": f"Leg {idx+1}: {player} ({stat})",
            "value": f"**Selection**: {side} {line}\n[Open PrizePicks Board](https://app.prizepicks.com/)",
            "inline": False
        })
        
    return {"embeds": [embed]}

def send_discord_webhook_sync(url: str, data: dict) -> tuple[int, str]:
    """
    Sends the embed payload synchronously. Returns (status_code, response_body).
    """
    if _discord_alerts_suppressed():
        return 0, "suppressed_in_tests"
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode('utf-8'),
        headers={
            'Content-Type': 'application/json',
            'User-Agent': 'DiscordNotifier/1.0'
        },
        method='POST'
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.status, response.read().decode('utf-8')
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode('utf-8')
        except Exception:
            body = str(e)
        return e.code, body
    except Exception as e:
        return 0, str(e)

async def send_discord_webhook(loop, url: str, data: dict):
    """
    Sends the embed payload with built-in rate-limiting retry handling.
    """
    for attempt in range(3):
        status, body = await loop.run_in_executor(None, send_discord_webhook_sync, url, data)
        if status in (200, 204):
            logger_discord.info("Alert successfully dispatched to Discord.")
            return
        elif status == 429:
            retry_after = 1.0
            try:
                res_json = json.loads(body)
                retry_after = float(res_json.get("retry_after", 1.0))
            except Exception:
                pass
            logger_discord.warning(f"Discord rate limit hit. Retrying in {retry_after} seconds...")
            await asyncio.sleep(retry_after)
        else:
            logger_discord.error(f"Failed to post to Discord (HTTP {status}): {body}")
            return

async def run_notifier():
    logger_discord.info("Discord Notifier Service is disabled in execution.py to prevent duplicate alerts (hfp_poller.py is the exclusive notifier).")
    return
    logger_discord.info("Discord Notifier Service live. Monitoring Redis queue...")
    
    while True:
        try:
            # Poll Redis non-blocking pop
            loop = asyncio.get_event_loop()
            res = await loop.run_in_executor(None, lambda: redis_client.rpop(QUEUE_NAME))
            
            if res:
                payload = json.loads(res)
                slip_id = payload.get("slip_id", "unknown")
                timestamp = payload.get("timestamp", 0)
                legs = payload.get("legs", [])
                
                # 1. Latency check (Drift Mitigation)
                age = time.time() - timestamp
                if age > MAX_ALERT_AGE_SECONDS:
                    logger_discord.warning(f"Skipping stale slip {slip_id} (age: {age:.1f}s > max: {MAX_ALERT_AGE_SECONDS}s)")
                    continue
                    
                # 2. Phantom Liquidity check
                if not passes_liquidity_filter(legs):
                    continue
                    
                # 3. Deduplication check
                sem_hash = get_semantic_hash(legs)
                dedup_key = f"dedup:slip:{sem_hash}"
                
                # Attempt to set key with EX TTL to lock it
                is_unique = await loop.run_in_executor(
                    None, 
                    lambda: redis_client.set(dedup_key, "1", nx=True, ex=DEDUP_TTL_SECONDS)
                )
                
                if not is_unique:
                    logger_discord.info(f"Skipping duplicate slip alert for {slip_id} (semantically identical alert sent recently).")
                    continue
                
                # 4. Format & Send alert
                embed_payload = format_discord_embed(payload)
                await send_discord_webhook(loop, DISCORD_WEBHOOK_URL, embed_payload)
                
        except Exception as e:
            logger_discord.error(f"Error in Notifier loop: {e}", exc_info=True)
            
        await asyncio.sleep(0.1)


# ═══════════════════════════════════════════════════════════════════════════════
# Main Block (direct execution)
# ═══════════════════════════════════════════════════════════════════════════════

async def main():
    logger_sgp.info("Booting consolidated execution cluster...")
    # Schedule only the SGP Correlation Engine (Discord notifier is disabled here to prevent dual dispatcher leak)
    await asyncio.gather(run_worker())

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger_sgp.info("Execution cluster cleanly terminated by user.")

