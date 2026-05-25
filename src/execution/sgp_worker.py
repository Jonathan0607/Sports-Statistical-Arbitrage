import os
import sys
# Dynamic project root path resolution
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Tuple

from infrastructure.entity_resolver import clean_player_name
from src.models import BetMaskingEngine
from src.slip_builder import CorrelatedSlipEvaluator
from infrastructure.redis_client import RedisClient

import uuid
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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SGPWorker")

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
    db_uri = os.getenv("POSTGRES_URI", "postgresql://postgres:password@localhost:5432/sports_db")
    logger.info(f"Starting SGP Worker with db: {db_uri}")
    
    evaluator = CorrelatedSlipEvaluator(db_uri=db_uri)
    tracker = GlobalExposureTracker()
    queue_manager = SlipQueueManager("execution_queue:prizepicks")
    redis_client = RedisClient().get_client()
    
    # Check cache status
    if not evaluator.player_name_to_id:
        logger.warning("Database cache is empty. Active database connection might be offline or unseeded.")
        
    logger.info("Decoupled SGP Worker initialized. Entering execution loop...")
    
    while True:
        try:
            # 1. Fetch all active signals from Redis
            keys = redis_client.keys("signal:*")
            if not keys:
                await asyncio.sleep(5)
                continue
                
            logger.debug(f"Retrieved {len(keys)} active signals from Redis.")
            
            signals = []
            for key in keys:
                data = redis_client.hgetall(key)
                if data:
                    signals.append(data)
                    
            # 2. Group signals by Retail Book
            # PrizePicks signals will have "Book" == "PrizePicks" (or keys starting with "signal:pp:")
            pp_signals = []
            ud_signals = []
            
            for sig in signals:
                book = sig.get("Book") or ""
                if "prizepicks" in book.lower() or "pp" in book.lower():
                    pp_signals.append(sig)
                elif "underdog" in book.lower() or "ud" in book.lower():
                    ud_signals.append(sig)
            
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
                            logger.info(
                                f"🔥 Detected +EV Same Game Parlay on PrizePicks: "
                                f"{res['player1']} ({sig1['Market']}) & {res['player2']} ({sig2['Market']}) | "
                                f"EV: {ev * 100.0:.2f}%"
                            )
                            
                            # Calculate Kelly Stake
                            joint_prob = res["joint_prob"]
                            payout_multiplier = res["payout_multiplier"]
                            net_odds = payout_multiplier - 1.0
                            if net_odds > 0:
                                kelly_f = joint_prob - (1.0 - joint_prob) / net_odds
                            else:
                                kelly_f = 0.0
                                
                            kelly_fraction = max(0.0, kelly_f) * 0.25  # Quarter Kelly
                            kelly_fraction = min(kelly_fraction, 0.05)  # Cap at 5%
                            recommended_stake = kelly_fraction * 50000.0
                            
                            # Run exposure check & scale
                            player_names = [res["player1"], res["player2"]]
                            # Extract slate_id or game_date from signals if available
                            game_date = sig1.get("game_date") or sig2.get("game_date") or sig1.get("game_id") or sig2.get("game_id")
                            slate_id = sig1.get("slate_id") or sig2.get("slate_id")
                            identifier = game_date or slate_id
                            
                            scaled_stake, is_allowed = tracker.check_and_scale_stake(player_names, recommended_stake, identifier=identifier)
                            
                            if not is_allowed:
                                logger.info(f"Slip rejected due to player exposure limits: {player_names}")
                                # Mark as processed so we don't spam warnings
                                redis_client.setex(dedup_key, 1800, "rejected")
                                continue
                                
                            # Apply Bet Masking/Camouflage roundings
                            masked_stake = BetMaskingEngine.mask_kelly_stake(scaled_stake)
                            if masked_stake > scaled_stake:
                                masked_stake = max(5, int(scaled_stake))
                                
                            if masked_stake < 5:
                                logger.info(f"Slip stake scaled below sportsbook minimum $5: ${scaled_stake:.2f} (masked: ${masked_stake})")
                                continue
                                
                            # Formulate Execution Payload
                            payload = queue_manager.formulate_execution_payload(sig1, sig2, masked_stake, ev)
                            
                            # Commit risk exposure to Redis
                            tracker.commit_exposure(player_names, masked_stake, identifier=identifier)
                            
                            # Push payload to Redis execution list
                            queue_manager.enqueue_slip(payload)
                            
                            # Set deduplication key in Redis with 1-hour expiry (3600 seconds)
                            redis_client.setex(dedup_key, 3600, "enqueued")
                            
        except Exception as err:
            logger.error(f"Error in SGP Worker loop: {err}", exc_info=True)
            
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
                logger.warning(
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
        
        logger.info(f"Committed exposure of ${stake:.2f} for players: {player_names} on slate/date {id_val}")
        return True


class SlipQueueManager:
    def __init__(self, queue_name: str = "execution_queue:prizepicks"):
        self.queue_name = queue_name
        self.redis_client = RedisClient().get_client()

    def formulate_execution_payload(self, leg1: dict, leg2: dict, risk_adjusted_stake: float, ev_edge: float) -> dict:
        """
        Constructs a structured Execution Payload dictionary containing:
        - player name, stat category, side, line, and target_execution_line for each leg
        - risk adjusted stake
        - EV edge
        - metadata (slip ID, timestamp)
        """
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
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "legs": [
                format_leg(leg1),
                format_leg(leg2)
            ],
            "risk_adjusted_stake": float(risk_adjusted_stake),
            "ev_edge": float(ev_edge)
        }
        return payload

    def enqueue_slip(self, payload: dict) -> bool:
        """Pushes the payload dict to the Redis list using LPUSH."""
        try:
            serialized = json.dumps(payload)
            self.redis_client.lpush(self.queue_name, serialized)
            logger.info(f"Enqueued execution payload to Redis list '{self.queue_name}': {payload['slip_id']}")
            return True
        except Exception as e:
            logger.error(f"Failed to enqueue execution payload: {e}")
            return False


if __name__ == "__main__":
    asyncio.run(run_worker())
