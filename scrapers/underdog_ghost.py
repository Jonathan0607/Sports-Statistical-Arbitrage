"""
UnderdogGhostClient — TLS-Fingerprint-Spoofed Underdog Fantasy Interceptor
===========================================================================
High-frequency REST poller for Underdog Fantasy player props.

Uses curl_cffi with chrome120 TLS fingerprint spoofing to bypass
Datadome/Cloudflare protections on Underdog Fantasy's internal API.

Target: Underdog Fantasy (v1/over_under_lines)
TLS Bypass: curl_cffi impersonating chrome120
"""

import asyncio
import logging
from curl_cffi.requests import AsyncSession, Session

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("UnderdogGhost")

# ── Target Parameters ──────────────────────────────────────────────
REST_URL = "https://api.underdogfantasy.com/v1/over_under_lines"
ORIGIN = "https://underdogfantasy.com"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/110.0.0.0 Safari/537.36"
)


class UnderdogGhostClient:
    """
    Hardened client that spoofs browser TLS fingerprints to intercept
    live odds data from Underdog Fantasy.

    Mirrors the architecture of GhostWebSocketClient (PrizePicks) exactly.
    Polls the internal v1/over_under_lines REST endpoint using curl_cffi.
    """

    def __init__(self):
        self.headers = {
            "Origin": ORIGIN,
            "Referer": f"{ORIGIN}/pick-em",
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        }
        self.message_count = 0
        self.data_messages = 0
        self.projections_cache = {}

    # ══════════════════════════════════════════════════════════
    # SYNCHRONOUS ONE-SHOT FETCH
    # ══════════════════════════════════════════════════════════

    def fetch_projections_sync(self) -> list:
        """
        Synchronous one-shot fetch of live over/under lines from Underdog Fantasy.
        Uses curl_cffi to spoof TLS fingerprint.
        Returns a list of parsed projection dicts.
        """
        with Session(impersonate="chrome110", headers=self.headers) as session:
            try:
                resp = session.get(REST_URL)
                if resp.status_code != 200:
                    logger.error(f"Underdog REST API returned {resp.status_code}")
                    return []

                raw = resp.json()
                projections = self._parse_projections(raw)
                logger.info(f"Fetched {len(projections)} projections from Underdog Fantasy")
                return projections
            except Exception as e:
                logger.error(f"Underdog sync fetch error: {e}")
                return []

    # ══════════════════════════════════════════════════════════
    # ASYNC POLLING LOOP
    # ══════════════════════════════════════════════════════════

    async def poll_projections(self, interval_seconds: int = 30, max_cycles: int = 0):
        """
        Async polling loop that continuously fetches projections and
        detects line movements between cycles.

        interval_seconds: Seconds between polls (default 30)
        max_cycles: Number of poll cycles (0 = infinite)
        """
        cycle = 0
        async with AsyncSession(impersonate="chrome110", headers=self.headers) as session:
            while max_cycles == 0 or cycle < max_cycles:
                cycle += 1
                try:
                    resp = await session.get(REST_URL)

                    if resp.status_code != 200:
                        logger.warning(f"[Cycle {cycle}] Underdog API returned {resp.status_code}")
                        await asyncio.sleep(interval_seconds)
                        continue

                    raw = resp.json()
                    projections = self._parse_projections(raw)
                    self.message_count += 1

                    # Detect line movements
                    movements = self._detect_movements(projections)
                    if movements:
                        for m in movements:
                            self.data_messages += 1
                            print(f"\n{'─'*70}")
                            print(f"  🚨 LINE MOVEMENT (UNDERDOG) | {m['player']} | {m['stat']}")
                            print(f"{'─'*70}")
                            print(f"  Old Line: {m['old_line']}  →  New Line: {m['new_line']}")
                            print(f"  Delta: {m['delta']:+.1f}")
                    else:
                        logger.debug(f"[Cycle {cycle}] {len(projections)} projections. No movements.")

                    # Update cache
                    for p in projections:
                        self.projections_cache[p['id']] = p

                except Exception as e:
                    logger.error(f"[Cycle {cycle}] Underdog polling error: {e}")

                await asyncio.sleep(interval_seconds)

    # ══════════════════════════════════════════════════════════
    # ASYNC SINGLE-SHOT FETCH (for HFP orchestrator)
    # ══════════════════════════════════════════════════════════

    async def fetch_current_lines(self, session: AsyncSession) -> dict:
        """
        Single async fetch cycle for use by the HFP master brain.
        Returns a dict keyed by prop_id for diff-engine compatibility.
        """
        try:
            resp = await session.get(REST_URL)
            if resp.status_code != 200:
                logger.warning(f"Underdog API returned {resp.status_code}")
                return {}

            raw = resp.json()
            projections = self._parse_projections(raw)

            # Convert to dict keyed by prop_id for diff engine
            parsed = {}
            for p in projections:
                parsed[p['id']] = {
                    'player': p['player'],
                    'prop': p['stat_display'],
                    'line': p['line'],
                    'odds_type': p.get('odds_type', 'standard')
                }
            return parsed
        except Exception as e:
            logger.error(f"Underdog fetch error: {e}")
            return {}

    # ══════════════════════════════════════════════════════════
    # PARSERS
    # ══════════════════════════════════════════════════════════

    def _parse_projections(self, raw_json: dict) -> list:
        """
        Parses the Underdog Fantasy over/under lines JSON response into flat dicts.

        Underdog's response structure typically contains:
        - over_under_lines: list of line objects
        - players: dict/list of player data (keyed by player_id)
        - appearances: appearance metadata linking players to games
        """
        results = []

        over_under_lines = raw_json.get("over_under_lines", [])
        # Build player lookup from appearances and players
        players_map = {}
        for player in raw_json.get("players", []):
            pid = player.get("id", "")
            players_map[pid] = {
                "first_name": player.get("first_name", ""),
                "last_name": player.get("last_name", ""),
            }

        appearances_map = {}
        for app in raw_json.get("appearances", []):
            app_id = app.get("id", "")
            player_id = app.get("player_id", "")
            appearances_map[app_id] = player_id

        for line in over_under_lines:
            line_id = line.get("id", "")
            appearance_id = line.get("over_under", {}).get("appearance_stat", {}).get("appearance_id", "")

            # Try nested structure first, then flat structure
            over_under = line.get("over_under", line)
            appearance_stat = over_under.get("appearance_stat", {})
            if not appearance_id:
                appearance_id = appearance_stat.get("appearance_id", "")

            player_id = appearances_map.get(appearance_id, "")
            player_info = players_map.get(player_id, {})
            player_name = f"{player_info.get('first_name', '')} {player_info.get('last_name', '')}".strip()
            if not player_name:
                player_name = "Unknown"

            stat_value = (
                line.get("stat_value") or
                over_under.get("stat_value") or
                appearance_stat.get("stat_value") or
                ""
            )

            stat_display = (
                appearance_stat.get("display_stat", "") or
                over_under.get("title", "") or
                ""
            )

            stat_type = appearance_stat.get("stat", stat_display)

            proj = {
                "id": line_id,
                "player": player_name,
                "stat": stat_type,
                "stat_display": stat_display,
                "line": float(stat_value) if stat_value else None,
                "odds_type": "standard",
                "status": line.get("status", ""),
                "book": "Underdog",
            }
            results.append(proj)

        return results

    def _detect_movements(self, new_projections: list) -> list:
        """Compares new projections against cache to detect line changes."""
        movements = []
        for p in new_projections:
            pid = p['id']
            if pid in self.projections_cache:
                old = self.projections_cache[pid]
                if (old['line'] != p['line']
                        and old['line'] is not None
                        and p['line'] is not None):
                    movements.append({
                        'player': p['player'],
                        'stat': p['stat_display'],
                        'old_line': old['line'],
                        'new_line': p['line'],
                        'delta': p['line'] - old['line'],
                    })
        return movements


if __name__ == "__main__":
    client = UnderdogGhostClient()

    print("\n" + "=" * 70)
    print("  UNDERDOG GHOST CLIENT — TLS FINGERPRINT SPOOFING ACTIVE (REST MODE)")
    print("=" * 70)
    print(f"  Target:      {REST_URL}")
    print(f"  TLS Spoof:   chrome110")
    print("=" * 70 + "\n")

    projections = client.fetch_projections_sync()

    if projections:
        print(f"{'─'*70}")
        print(f"  📊 LIVE UNDERDOG FANTASY PROJECTIONS ({len(projections)} props)")
        print(f"{'─'*70}\n")

        for i, p in enumerate(projections[:15]):
            print(f"  {i+1:>2}. {p['player']:<25} | {p['stat_display']:<12} | "
                  f"Line: {p['line']:<6}")

        print(f"\n{'─'*70}")
        print(f"  Total Underdog props on board: {len(projections)}")
        print(f"{'─'*70}\n")
    else:
        print("  Failed to fetch projections. Check network/TLS configuration.\n")
