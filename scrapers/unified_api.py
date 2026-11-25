import os
import sys
import json
import logging
import asyncio
from datetime import datetime, timezone, timedelta
from pathlib import Path
import re
import unicodedata

import httpx
from dotenv import load_dotenv

# Ensure root directory is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import ShinDevigger, MarketMicrostructure
from infrastructure.entity_resolver import clean_player_name

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SharpLineProvider")


# ── Custom Exception ──────────────────────────────────────────────
class CriticalAPIError(Exception):
    """Raised on 403/429 or other fatal API errors that must NOT be swallowed."""
    pass


# ── Constants ─────────────────────────────────────────────────────
PARLAY_API_BASE = "https://api.parlay-api.com/v1"
CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sharp_cache.json")
# ACCEPTED MRM RISK (R-03): TTL maintained at 3 hours to enforce 1,000/month API credit budget.
# Stale line risk is mitigated downstream by the $100 liquidity requirement on real-time prediction market anchors.
CACHE_TTL_MINUTES = 30


class SharpLineProvider:
    """
    Cached ParlayAPI client that fetches Pinnacle/DraftKings sharp lines,
    devigs them using Shin's method, and caches the results locally.

    Public entry point: get_sharp_lines() — always returns a dict
    (empty on error for graceful degradation).
    """

    def __init__(self):
        # Load .env using absolute path to avoid cwd issues
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        load_dotenv(dotenv_path=os.path.join(root_dir, ".env"))

        self.api_keys = self._load_keys()
        self.cache_ttl_minutes = CACHE_TTL_MINUTES

    @property
    def api_key(self) -> str:
        return self.api_keys[0] if self.api_keys else ""

    @api_key.setter
    def api_key(self, value: str):
        if not value:
            self.api_keys = []
        else:
            self.api_keys = [value]

    def _load_keys(self) -> list:
        """Dynamically loads all available keys from the .env file."""
        keys = []
        for i in range(1, 10):
            key = os.getenv(f"PARLAY_API_KEY_{i}")
            if key:
                keys.append(key)
                
        # Fallback for legacy configuration
        if not keys:
            legacy_key = os.getenv("PARLAY_API_KEY")
            if legacy_key:
                keys.append(legacy_key)
                
        if keys:
            logger.info(f"Loaded {len(keys)} ParlayAPI keys for round-robin rotation.")
        else:
            logger.error("No PARLAY_API_KEYs found in environment.")
        return keys

    def _get_next_key(self, cache_data: dict) -> tuple[str, int]:
        """Determines the next API key in the rotation."""
        if not self.api_keys:
            return "", -1
        last_index = cache_data.get("last_key_index", -1)
        next_index = (last_index + 1) % len(self.api_keys)
        return self.api_keys[next_index], next_index

    # ══════════════════════════════════════════════════════════
    # CACHE MANAGEMENT
    # ══════════════════════════════════════════════════════════

    def _deserialize_lines(self, lines_data: dict) -> dict:
        """Safely casts serialized string keys of the third-dimension back to floats."""
        converted_lines = {}
        for player, stats in lines_data.items():
            converted_lines[player] = {}
            for stat, lines_dict in stats.items():
                converted_lines[player][stat] = {}
                for line_str, val in lines_dict.items():
                    try:
                        line_float = float(line_str)
                        converted_lines[player][stat][line_float] = val
                    except ValueError:
                        converted_lines[player][stat][line_str] = val
        return converted_lines

    def load_cache(self) -> dict | None:
        """
        Reads sharp_cache.json and returns the cached data if it exists
        and is less than 30 minutes old. Returns None if stale or missing.
        """
        try:
            if not Path(CACHE_FILE).exists():
                logger.info("No sharp_cache.json found. Will fetch from API.")
                return None

            with open(CACHE_FILE, 'r') as f:
                cache = json.load(f)

            fetched_at_str = cache.get("fetched_at")
            if not fetched_at_str:
                logger.warning("Cache file missing 'fetched_at' timestamp. Treating as stale.")
                return None

            fetched_at = datetime.fromisoformat(fetched_at_str)
            age = datetime.now(timezone.utc) - fetched_at
            age_minutes = age.total_seconds() / 60.0

            if age_minutes < self.cache_ttl_minutes:
                logger.info(
                    f"Sharp cache is fresh ({age_minutes:.1f} min old, "
                    f"TTL={self.cache_ttl_minutes} min). Returning cached data."
                )
                return self._deserialize_lines(cache.get("lines", {}))
            else:
                logger.info(
                    f"Sharp cache is stale ({age_minutes:.1f} min old). "
                    f"Will refresh from API."
                )
                return None

        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse sharp_cache.json: {e}. Treating as stale.")
            return None

    def save_cache(self, lines: dict, api_calls_this_month: int, last_key_index: int):
        """Writes devigged sharp lines + metadata to sharp_cache.json."""
        cache = {
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "api_calls_this_month": api_calls_this_month,
            "last_key_index": last_key_index,
            "lines": lines,
        }
        try:
            with open(CACHE_FILE, 'w') as f:
                json.dump(cache, f, indent=2)
            logger.info(f"Sharp cache saved ({len(lines)} player entries, key index {last_key_index}).")
        except Exception as e:
            logger.error(f"Failed to save sharp_cache.json: {e}")

    def _get_monthly_call_count(self) -> int:
        """Reads the current month's API call counter from cache file."""
        try:
            if not Path(CACHE_FILE).exists():
                return 0
            with open(CACHE_FILE, 'r') as f:
                cache = json.load(f)

            # Reset counter if month has changed
            fetched_at_str = cache.get("fetched_at", "")
            if fetched_at_str:
                fetched_at = datetime.fromisoformat(fetched_at_str)
                now = datetime.now(timezone.utc)
                if fetched_at.month != now.month or fetched_at.year != now.year:
                    logger.info("New month detected. Resetting API call counter.")
                    return 0

            return cache.get("api_calls_this_month", 0)
        except Exception:
            return 0

    # ══════════════════════════════════════════════════════════
    # API FETCH + DEVIG
    # ══════════════════════════════════════════════════════════

    async def fetch_sharp_lines(self, api_key: str = None, sports: list = None) -> dict:
        """
        Fetches player prop lines from ParlayAPI for Pinnacle and DraftKings.
        Devigs Pinnacle odds using Shin's method.
        Returns {player_name: {stat_type: {line: {pinnacle_true_prob, pinnacle_line, dk_line}}}}.
        """
        if sports is None:
            sports = ["basketball_nba", "basketball_wnba"]
            
        # Determine keys to try
        if api_key is not None:
            keys_to_try = [api_key]
        else:
            keys_to_try = self.api_keys

        if not keys_to_try:
            logger.critical("No ParlayAPI keys resolved/provided. Returning empty sharp lines.")
            return {}

        params = {
            "bookmakers": "pinnacle,draftkings",
            "markets": "player_points,player_rebounds,player_assists",
        }

        async with httpx.AsyncClient(timeout=15.0) as client:
            for idx, key in enumerate(keys_to_try, start=1):
                try:
                    logger.info(f"Attempting to refresh sharp lines using API Key #{idx}...")
                    headers = {"x-api-key": key}
                    combined_data = []
                    success = True
                    
                    for sport in sports:
                        url = f"{PARLAY_API_BASE}/sports/{sport}/props"
                        resp = await client.get(url, headers=headers, params=params)

                        if resp.status_code == 200:
                            data = resp.json()
                            if isinstance(data, list):
                                combined_data.extend(data)
                            elif isinstance(data, dict):
                                combined_data.extend(data.get("data", []))
                        elif resp.status_code == 403:
                            try:
                                resp_json = resp.json()
                                error_msg = resp_json.get("detail", {}).get("message", "")
                            except Exception:
                                error_msg = resp.text

                            if "Credit limit exceeded" in error_msg:
                                logger.warning(f"⚠️ API Key #{idx} exhausted (Credit limit exceeded). Attempting failover...")
                            else:
                                logger.warning(f"⚠️ API Key #{idx} returned 403 Forbidden: {error_msg}. Attempting failover...")
                            success = False
                            break
                        elif resp.status_code == 429:
                            logger.warning(f"⚠️ API Key #{idx} rate limited (HTTP 429). Attempting failover...")
                            success = False
                            break
                        else:
                            logger.error(f"Unexpected API response from Key #{idx}: {resp.status_code}")
                            success = False
                            break

                    if success:
                        logger.info(f"Successfully refreshed sharp lines using API Key #{idx}.")
                        if api_key is None:
                            self.last_successful_key_index = idx - 1
                        else:
                            try:
                                self.last_successful_key_index = self.api_keys.index(key)
                            except ValueError:
                                self.last_successful_key_index = 0
                        return self._parse_and_devig(combined_data)
                    else:
                        continue

                except Exception as e:
                    logger.error(f"Network error during sharp lines fetch with Key #{idx}: {str(e)}")
                    continue

        # If all keys failed
        logger.critical("🚨 Graceful degradation: All available ParlayAPI keys failed or exhausted. Returning empty dict.")
        return {}

    def _parse_and_devig(self, raw_data: list) -> dict:
        """
        Parses the new flat list structure from ParlayAPI and devigs the sharp lines.

        Returns:
            {
                "lebron james": {
                    "points": {
                        25.5: {
                            "pinnacle_true_prob": 0.542,
                            "pinnacle_line": 25.5,
                            "dk_line": 25.5,
                        }
                    },
                    "rebounds": {...},
                },
                ...
            }
        """
        lines = {}
        if not isinstance(raw_data, list):
            raw_data = raw_data.get("data", []) if isinstance(raw_data, dict) else []

        # Print the raw count of players retrieved from the API before EntityResolver touches them
        raw_players = {entry.get("player") for entry in raw_data if entry.get("player")}
        print(f"RAW PLAYER COUNT FROM API: {len(raw_players)}")

        # Group records by (player_clean, stat_type, line_val)
        grouped = {}
        for entry in raw_data:
            player_name = entry.get("player")
            if not player_name:
                continue

            market = entry.get("market_key") or entry.get("market") or ""
            stat_type = self._normalize_market(market)
            if not stat_type:
                continue

            player_clean = clean_player_name(player_name)
            line = entry.get("line")
            if line is None:
                continue
            try:
                line_val = float(line)
            except (ValueError, TypeError):
                continue

            key = (player_clean, stat_type, line_val)
            if key not in grouped:
                grouped[key] = {}

            bookmaker = entry.get("bookmaker", "").lower()
            grouped[key][bookmaker] = entry

        for (player_clean, stat_type, line_val), books in grouped.items():
            # We want to devig Pinnacle if available, otherwise DraftKings
            ref_book = None
            if "pinnacle" in books:
                ref_book = "pinnacle"
            elif "draftkings" in books:
                ref_book = "draftkings"
            elif books:
                ref_book = list(books.keys())[0]

            if not ref_book:
                continue

            ref_entry = books[ref_book]
            over_price = ref_entry.get("over_price")
            under_price = ref_entry.get("under_price")

            if over_price is None or under_price is None:
                continue

            try:
                # Convert decimal odds to American if they are floats (like 1.91)
                # But if they are ints (like -110), use them directly
                over_american = int(over_price) if isinstance(over_price, (int, float)) and abs(over_price) > 5.0 else self._decimal_to_american(float(over_price))
                under_american = int(under_price) if isinstance(under_price, (int, float)) and abs(under_price) > 5.0 else self._decimal_to_american(float(under_price))

                shin_res = MarketMicrostructure.shins_method(over_american, under_american)
                true_prob = shin_res["true_prob_over"]
            except Exception as e:
                logger.debug(f"Devig failed for {player_clean} {stat_type} line {line_val}: {e}")
                continue

            # DraftKings line
            dk_line_val = None
            if "draftkings" in books:
                dk_line_val = float(books["draftkings"].get("line", line_val))

            if player_clean not in lines:
                lines[player_clean] = {}
            if stat_type not in lines[player_clean]:
                lines[player_clean][stat_type] = {}

            lines[player_clean][stat_type][line_val] = {
                "pinnacle_true_prob": true_prob,
                "pinnacle_line": line_val,
                "dk_line": dk_line_val,
            }

        logger.info(f"Parsed and devigged sharp lines for {len(lines)} players.")
        return lines

    @staticmethod
    def _normalize_market(market: str) -> str | None:
        """Normalizes ParlayAPI market names to our internal stat types."""
        if not market:
            return None
        m = market.lower()
        if "point" in m:
            return "points"
        elif "rebound" in m:
            return "rebounds"
        elif "assist" in m:
            return "assists"
        elif "turnover" in m:
            return "turnovers"
        return None

    @staticmethod
    def _decimal_to_american(decimal_odds: float) -> int:
        """
        Translates decimal odds into bounded American Odds.
        Clamps boundaries to stay cleanly within standard 4-byte database types.
        """
        MAX_SQL_INT = 2000000000
        
        # Ensure invalid decimals don't crash the calculation
        if decimal_odds <= 1.0:
            return MAX_SQL_INT  
            
        if decimal_odds >= 2.0:
            odds = (decimal_odds - 1.0) * 100.0
        else:
            odds = -100.0 / (decimal_odds - 1.0)
            
        return max(-MAX_SQL_INT, min(int(round(odds)), MAX_SQL_INT))

    # ══════════════════════════════════════════════════════════
    # PUBLIC ENTRY POINT
    # ══════════════════════════════════════════════════════════

    async def get_sharp_lines(self) -> dict:
        """
        The primary entry point. Returns devigged sharp lines.

        1. Try cache first. If fresh (< 30 min), return immediately.
        2. If stale/missing, fetch from API using the next rotated key, devig, cache, return.
        3. On CriticalAPIError or other failures, return empty dict for graceful degradation.
        """
        if not self.api_keys:
            logger.warning("No ParlayAPI keys loaded. Returning empty sharp lines.")
            return {}

        # Step 1: Check cache
        cached = self.load_cache()
        if cached is not None:
            return cached

        # Read cache_data for monthly count and last key index
        cache_data = {}
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, 'r') as f:
                    cache_data = json.load(f)
            except Exception:
                pass

        # Step 2: Check budget
        call_count = self._get_monthly_call_count()
        monthly_budget = len(self.api_keys) * 1000
        budget_warning_threshold = int(monthly_budget * 0.9)
        if call_count >= monthly_budget:
            logger.critical(
                f"MONTHLY API BUDGET EXHAUSTED ({call_count}/{monthly_budget}). "
                f"Returning stale/empty cache. NO API call will be made."
            )
            # Try to return stale data from cache file even if expired
            try:
                return self._deserialize_lines(cache_data.get("lines", {}))
            except Exception:
                return {}

        if call_count >= budget_warning_threshold:
            logger.critical(
                f"⚠️  API BUDGET WARNING: {call_count}/{monthly_budget} calls used this month. "
                f"Only {monthly_budget - call_count} remaining!"
            )

        # Step 3: Fetch from API using the key pool (starting from index 0/Key 1)
        logger.info("Sharp cache expired/missing. Fetching fresh lines from ParlayAPI sequential pool...")
        try:
            self.last_successful_key_index = 0
            lines = await self.fetch_sharp_lines()
            new_call_count = call_count + 1
            succeeded_index = getattr(self, 'last_successful_key_index', 0)
            self.save_cache(lines, new_call_count, succeeded_index)
            logger.info(
                f"Sharp lines refreshed from ParlayAPI. "
                f"API calls this month: {new_call_count}/{monthly_budget}"
            )
            return lines
        except Exception as e:
            logger.critical(f"🚨 Graceful degradation: failed to refresh sharp lines: {e}. Returning empty dict.")
            return {}

    # ══════════════════════════════════════════════════════════
    # SYNCHRONOUS FORCE SEED
    # ══════════════════════════════════════════════════════════

    def force_sync_seed(self, sports: list = None) -> dict:
        """
        Forces an immediate synchronous API call to ParlayAPI to seed the cache.
        Returns parsed/devigged lines, and saves them to sharp_cache.json.
        """
        if sports is None:
            sports = ["basketball_nba", "basketball_wnba"]
            
        keys_to_try = self.api_keys
        if not keys_to_try:
            logger.critical("No ParlayAPI keys available. Cannot force seed sharp cache.")
            return {}

        params = {
            "bookmakers": "pinnacle,draftkings",
            "markets": "player_points,player_rebounds,player_assists",
        }

        with httpx.Client(timeout=15.0) as client:
            for idx, key in enumerate(keys_to_try, start=1):
                try:
                    logger.info(f"Forcing synchronous ParlayAPI call using API Key #{idx}...")
                    headers = {"x-api-key": key}
                    combined_data = []
                    success = True
                    
                    for sport in sports:
                        url = f"{PARLAY_API_BASE}/sports/{sport}/props"
                        resp = client.get(url, headers=headers, params=params)

                        if resp.status_code == 200:
                            data = resp.json()
                            if isinstance(data, list):
                                combined_data.extend(data)
                            elif isinstance(data, dict):
                                combined_data.extend(data.get("data", []))
                        elif resp.status_code == 403:
                            try:
                                resp_json = resp.json()
                                error_msg = resp_json.get("detail", {}).get("message", "")
                            except Exception:
                                error_msg = resp.text

                            if "Credit limit exceeded" in error_msg:
                                logger.warning(f"⚠️ Sync API Key #{idx} exhausted (Credit limit exceeded). Attempting failover...")
                            else:
                                logger.warning(f"⚠️ Sync API Key #{idx} returned 403 Forbidden: {error_msg}. Attempting failover...")
                            success = False
                            break
                        elif resp.status_code == 429:
                            logger.warning(f"⚠️ Sync API Key #{idx} rate limited (HTTP 429). Attempting failover...")
                            success = False
                            break
                        else:
                            logger.error(f"Unexpected sync API response from Key #{idx}: {resp.status_code}")
                            success = False
                            break

                    if success:
                        logger.info(f"Successfully force-seeded sharp cache using API Key #{idx}.")
                        lines = self._parse_and_devig(combined_data)
                        call_count = self._get_monthly_call_count()
                        self.save_cache(lines, call_count + 1, idx - 1)
                        return lines
                    else:
                        continue

                except Exception as e:
                    logger.error(f"Network error during sync sharp lines fetch with Key #{idx}: {str(e)}")
                    continue

        logger.critical("🚨 Graceful degradation: All available ParlayAPI keys failed or exhausted during sync seed. Returning empty dict.")
        return {}


if __name__ == "__main__":
    async def main():
        provider = SharpLineProvider()

        print("\n" + "=" * 70)
        print("  SHARP LINE PROVIDER — ParlayAPI + Shin Devig Engine")
        print("=" * 70)
        print(f"  API Base:     {PARLAY_API_BASE}")
        print(f"  Cache File:   {CACHE_FILE}")
        print(f"  Cache TTL:    {CACHE_TTL_MINUTES} minutes")
        print(f"  Monthly Cap:  {len(provider.api_keys) * 1000} calls")
        print("=" * 70 + "\n")

        try:
            lines = await provider.get_sharp_lines()
            if lines:
                print(f"{'─'*70}")
                print(f"  📊 DEVIGGED SHARP LINES ({len(lines)} players)")
                print(f"{'─'*70}\n")
                for player, stats in list(lines.items())[:10]:
                    for stat, lines_dict in stats.items():
                        for line, data in lines_dict.items():
                            prob = data.get('pinnacle_true_prob')
                            prob_str = f"{prob*100:.1f}%" if prob else "N/A"
                            dk_line = data.get('dk_line')
                            dk_val_str = str(dk_line) if dk_line is not None else 'N/A'
                            print(f"  {player:<25} | {stat:<10} | "
                                  f"PIN: {line:<6} | "
                                  f"DK: {dk_val_str:<6} | "
                                  f"True: {prob_str}")
                print(f"\n{'─'*70}\n")
            else:
                print("  No sharp lines available (cache empty or API key missing).\n")
        except Exception as e:
            print(f"\n  🚨 CRITICAL: {e}\n")

    asyncio.run(main())
