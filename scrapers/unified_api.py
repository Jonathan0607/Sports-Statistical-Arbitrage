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

from src.models import ShinDevigger
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
CACHE_TTL_MINUTES = 180
MONTHLY_BUDGET = 1000
BUDGET_WARNING_THRESHOLD = 900


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

        self.api_key = os.getenv("PARLAY_API_KEY", "")
        if self.api_key:
            logger.info("PARLAY_API_KEY successfully detected.")
        else:
            logger.critical(
                "PARLAY_API_KEY not found in environment (status: NOT DETECTED). "
                "Sharp line provider will degrade gracefully and return empty data."
            )

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
        and is less than 45 minutes old. Returns None if stale or missing.
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

            if age_minutes < CACHE_TTL_MINUTES:
                logger.info(
                    f"Sharp cache is fresh ({age_minutes:.1f} min old, "
                    f"TTL={CACHE_TTL_MINUTES} min). Returning cached data."
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

    def save_cache(self, lines: dict, api_calls_this_month: int):
        """Writes devigged sharp lines + metadata to sharp_cache.json."""
        cache = {
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "api_calls_this_month": api_calls_this_month,
            "lines": lines,
        }
        try:
            with open(CACHE_FILE, 'w') as f:
                json.dump(cache, f, indent=2)
            logger.info(f"Sharp cache saved ({len(lines)} player entries).")
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

    async def fetch_sharp_lines(self, sport: str = "basketball_nba") -> dict:
        """
        Fetches player prop lines from ParlayAPI for Pinnacle and DraftKings.
        Devigs Pinnacle odds using Shin's method.
        Returns {player_name: {stat_type: {line: {pinnacle_true_prob, pinnacle_line, dk_line}}}}.

        Raises CriticalAPIError on 403/429.
        """
        if not self.api_key:
            logger.warning("No PARLAY_API_KEY. Returning empty sharp lines.")
            return {}

        url = f"{PARLAY_API_BASE}/sports/{sport}/props"
        headers = {"x-api-key": self.api_key}
        params = {
            "bookmakers": "pinnacle,draftkings",
            "markets": "player_points,player_rebounds,player_assists",
        }

        async with httpx.AsyncClient(timeout=15.0) as client:
            try:
                resp = await client.get(url, headers=headers, params=params)

                if resp.status_code == 403:
                    raise CriticalAPIError(
                        f"ParlayAPI returned 403 Forbidden. "
                        f"API key may be invalid or revoked. Response: {resp.text[:300]}"
                    )
                elif resp.status_code == 429:
                    raise CriticalAPIError(
                        f"ParlayAPI returned 429 Rate Limited. "
                        f"Monthly quota likely exceeded. Response: {resp.text[:300]}"
                    )
                elif resp.status_code != 200:
                    logger.error(
                        f"ParlayAPI returned unexpected status {resp.status_code}: "
                        f"{resp.text[:300]}"
                    )
                    return {}

                data = resp.json()
                return self._parse_and_devig(data)

            except CriticalAPIError:
                raise  # Always propagate critical errors to public wrapper
            except Exception as e:
                logger.error(f"ParlayAPI request failed: {e}")
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

                true_probs = ShinDevigger.devig([over_american, under_american])
                true_prob = true_probs[0]  # Over probability
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
        """Converts decimal odds to American odds."""
        if decimal_odds >= 2.0:
            return int(round((decimal_odds - 1.0) * 100))
        elif decimal_odds > 1.0:
            return int(round(-100.0 / (decimal_odds - 1.0)))
        else:
            return -110  # Fallback

    # ══════════════════════════════════════════════════════════
    # PUBLIC ENTRY POINT
    # ══════════════════════════════════════════════════════════

    async def get_sharp_lines(self) -> dict:
        """
        The primary entry point. Returns devigged sharp lines.

        1. Try cache first. If fresh (< 45 min), return immediately.
        2. If stale/missing, fetch from API, devig, cache, return.
        3. On CriticalAPIError or other failures, return empty dict for graceful degradation.
        """
        # Step 1: Check cache
        cached = self.load_cache()
        if cached is not None:
            return cached

        # Step 2: Check budget
        call_count = self._get_monthly_call_count()
        if call_count >= MONTHLY_BUDGET:
            logger.critical(
                f"MONTHLY API BUDGET EXHAUSTED ({call_count}/{MONTHLY_BUDGET}). "
                f"Returning stale/empty cache. NO API call will be made."
            )
            # Try to return stale data from cache file even if expired
            try:
                with open(CACHE_FILE, 'r') as f:
                    cache = json.load(f)
                return self._deserialize_lines(cache.get("lines", {}))
            except Exception:
                return {}

        if call_count >= BUDGET_WARNING_THRESHOLD:
            logger.critical(
                f"⚠️  API BUDGET WARNING: {call_count}/{MONTHLY_BUDGET} calls used this month. "
                f"Only {MONTHLY_BUDGET - call_count} remaining!"
            )

        # Step 3: Fetch from API
        try:
            lines = await self.fetch_sharp_lines()
            new_call_count = call_count + 1
            self.save_cache(lines, new_call_count)
            logger.info(
                f"Sharp lines refreshed from ParlayAPI. "
                f"API calls this month: {new_call_count}/{MONTHLY_BUDGET}"
            )
            return lines
        except Exception as e:
            logger.critical(f"🚨 Graceful degradation: failed to refresh sharp lines: {e}. Returning empty dict.")
            return {}

    # ══════════════════════════════════════════════════════════
    # SYNCHRONOUS FORCE SEED
    # ══════════════════════════════════════════════════════════

    def force_sync_seed(self) -> dict:
        """
        Forces an immediate synchronous API call to ParlayAPI to seed the cache.
        Returns parsed/devigged lines, and saves them to sharp_cache.json.
        """
        if not self.api_key:
            logger.critical("PARLAY_API_KEY missing. Cannot force seed sharp cache.")
            return {}

        url = f"{PARLAY_API_BASE}/sports/basketball_nba/props"
        headers = {"x-api-key": self.api_key}
        params = {
            "bookmakers": "pinnacle,draftkings",
            "markets": "player_points,player_rebounds,player_assists",
        }
        logger.info("Forcing synchronous ParlayAPI call to seed sharp cache on boot...")
        try:
            with httpx.Client(timeout=15.0) as client:
                resp = client.get(url, headers=headers, params=params)
                if resp.status_code != 200:
                    logger.critical(f"ParlayAPI sync seed failed with status {resp.status_code}: {resp.text[:300]}")
                    return {}

                raw_data = resp.json()
                lines = self._parse_and_devig(raw_data)

                # Increment and save cache call count
                call_count = self._get_monthly_call_count()
                self.save_cache(lines, call_count + 1)
                logger.info(f"Successfully seeded sharp cache with {len(lines)} players.")
                return lines
        except Exception as e:
            logger.critical(f"Exception during synchronous cache seeding: {e}. Returning empty dictionary.")
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
        print(f"  Monthly Cap:  {MONTHLY_BUDGET} calls")
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
                            print(f"  {player:<25} | {stat:<10} | "
                                  f"PIN: {line:<6} | "
                                  f"DK: {data.get('dk_line', 'N/A'):<6} | "
                                  f"True: {prob_str}")
                print(f"\n{'─'*70}\n")
            else:
                print("  No sharp lines available (cache empty or API key missing).\n")
        except Exception as e:
            print(f"\n  🚨 CRITICAL: {e}\n")

    asyncio.run(main())
