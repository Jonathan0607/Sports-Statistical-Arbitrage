"""
GhostWebSocketClient — TLS-Fingerprint-Spoofed Sportsbook Interceptor
======================================================================
Dual-mode interceptor for PrizePicks:
  1. REST Poller   — Hits the unauthenticated public projections API
                     (no login required, bypasses Cloudflare via curl_cffi)
  2. WSS Listener  — Template for authenticated ActionCable streaming
                     (requires session token from a logged-in browser)

Target: PrizePicks (Rails / ActionCable)
TLS Bypass: curl_cffi impersonating chrome120
"""

import asyncio
import json
import logging
import time
from curl_cffi.requests import AsyncSession, Session

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("GhostWebSocket")

# ── Target Parameters ──────────────────────────────────────────────
REST_URL = "https://api.prizepicks.com/projections"
WSS_URL = "wss://ws.prizepicks.com/connection/websocket"
ORIGIN = "https://app.prizepicks.com"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/148.0.0.0 Safari/537.36"
)

# NBA league_id = 7 on PrizePicks
NBA_LEAGUE_ID = "7"


class GhostWebSocketClient:
    """
    Hardened client that spoofs browser TLS fingerprints to intercept
    live odds data from PrizePicks.

    Mode 1 (REST Polling): Works immediately, no auth needed.
    Mode 2 (WebSocket):    Requires an ActionCable session token.
    """

    def __init__(self):
        self.headers = {
            "Origin": ORIGIN,
            "User-Agent": USER_AGENT,
            "Accept-Language": "en-US,en;q=0.9",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        }
        self.message_count = 0
        self.data_messages = 0
        self.projections_cache = {}

    # ══════════════════════════════════════════════════════════
    # MODE 1: REST API POLLER (No Auth Required)
    # ══════════════════════════════════════════════════════════

    def fetch_projections_sync(self, league_id: str = NBA_LEAGUE_ID,
                                per_page: int = 100) -> list:
        """
        Synchronous one-shot fetch of live projections from PrizePicks.
        Uses curl_cffi to spoof TLS fingerprint.
        Returns a list of parsed projection dicts.
        """
        with Session(impersonate="chrome120", headers=self.headers) as session:
            params = {
                "league_id": league_id,
                "per_page": str(per_page),
            }
            resp = session.get(REST_URL, params=params)

            if resp.status_code != 200:
                logger.error(f"REST API returned {resp.status_code}")
                return []

            raw = resp.json()
            projections = self._parse_projections(raw)
            logger.info(f"Fetched {len(projections)} projections from REST API")
            return projections

    async def poll_projections(self, league_id: str = NBA_LEAGUE_ID,
                                interval_seconds: int = 30,
                                max_cycles: int = 0):
        """
        Async polling loop that continuously fetches projections and
        detects line movements between cycles.

        interval_seconds: Seconds between polls (default 30)
        max_cycles: Number of poll cycles (0 = infinite)
        """
        cycle = 0
        async with AsyncSession(impersonate="chrome120", headers=self.headers) as session:
            while max_cycles == 0 or cycle < max_cycles:
                cycle += 1
                try:
                    params = {"league_id": league_id, "per_page": "100"}
                    resp = await session.get(REST_URL, params=params)

                    if resp.status_code != 200:
                        logger.warning(f"[Cycle {cycle}] REST API returned {resp.status_code}")
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
                            print(f"  🚨 LINE MOVEMENT DETECTED | {m['player']} | {m['stat']}")
                            print(f"{'─'*70}")
                            print(f"  Old Line: {m['old_line']}  →  New Line: {m['new_line']}")
                            print(f"  Delta: {m['delta']:+.1f}")
                    else:
                        logger.debug(f"[Cycle {cycle}] {len(projections)} projections. No movements.")

                    # Update cache
                    for p in projections:
                        self.projections_cache[p['id']] = p

                except Exception as e:
                    logger.error(f"[Cycle {cycle}] Error: {e}")

                await asyncio.sleep(interval_seconds)

    def _parse_projections(self, raw_json: dict) -> list:
        """Parses the PrizePicks JSON:API response into flat dicts."""
        results = []
        data = raw_json.get("data", [])
        included = raw_json.get("included", [])

        # Build lookup for included resources (players, games, etc.)
        included_map = {}
        for item in included:
            key = (item.get("type"), item.get("id"))
            included_map[key] = item.get("attributes", {})

        for entry in data:
            attrs = entry.get("attributes", {})
            relationships = entry.get("relationships", {})

            # Resolve player name
            player_rel = relationships.get("new_player", {}).get("data", {})
            player_attrs = included_map.get((player_rel.get("type"), player_rel.get("id")), {})
            player_name = player_attrs.get("display_name") or player_attrs.get("name", "Unknown")

            proj = {
                "id": entry.get("id"),
                "player": player_name,
                "stat": attrs.get("stat_type", ""),
                "stat_display": attrs.get("stat_display_name", ""),
                "line": attrs.get("line_score"),
                "odds_type": attrs.get("odds_type", "standard"),
                "status": attrs.get("status", ""),
                "description": attrs.get("description", ""),
                "start_time": attrs.get("start_time", ""),
                "is_live": attrs.get("is_live", False),
                "adjusted_odds": attrs.get("adjusted_odds", False),
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
                if old['line'] != p['line'] and old['line'] is not None and p['line'] is not None:
                    movements.append({
                        'player': p['player'],
                        'stat': p['stat_display'],
                        'old_line': old['line'],
                        'new_line': p['line'],
                        'delta': p['line'] - old['line'],
                    })
        return movements

    # ══════════════════════════════════════════════════════════
    # MODE 2: WEBSOCKET (Requires Auth Token)
    # ══════════════════════════════════════════════════════════

    async def connect_websocket(self, session_token: str = ""):
        """
        Connects to the ActionCable WebSocket for real-time streaming.
        Requires a valid session token (extracted from a logged-in browser).

        NOTE: PrizePicks disconnects anonymous connections with 'stale'.
        This method is a template for when you have a session token.
        """
        url = WSS_URL
        if session_token:
            url = f"{WSS_URL}?token={session_token}"

        logger.info(f"Initiating WSS connection to {url}")

        async with AsyncSession(impersonate="chrome120", headers=self.headers) as session:
            try:
                ws = await session.ws_connect(url)
                logger.info("WebSocket connection established.")

                # Phase 1: Wait for ActionCable Welcome
                msg = await ws.recv()
                frame_data = msg[0] if isinstance(msg, tuple) else msg
                if isinstance(frame_data, bytes):
                    frame_data = frame_data.decode("utf-8", errors="replace")

                logger.info(f"Server response: {frame_data[:200]}")

                try:
                    welcome = json.loads(frame_data)
                    if welcome.get("type") == "welcome":
                        logger.info("✅ ActionCable WELCOME received.")

                        # Phase 2: Subscribe
                        sub = {"command": "subscribe", "identifier": json.dumps({"channel": "ProjectionsChannel"})}
                        await ws.send_str(json.dumps(sub))
                        logger.info("📡 Subscribed to ProjectionsChannel")

                        # Phase 3: Listen
                        while True:
                            raw = await ws.recv()
                            frame = raw[0] if isinstance(raw, tuple) else raw
                            if isinstance(frame, bytes):
                                frame = frame.decode("utf-8", errors="replace")
                            print(frame)
                    elif "stale" in frame_data.lower() or "unauthorized" in frame_data.lower():
                        logger.warning("❌ Server rejected: auth required. Use REST poller mode instead.")
                except json.JSONDecodeError:
                    if b"stale" in (msg[0] if isinstance(msg, tuple) else msg):
                        logger.warning("❌ Server sent 'stale' disconnect. Auth token required.")
                    else:
                        logger.warning(f"Non-JSON response: {frame_data[:200]}")

            except Exception as e:
                logger.error(f"WebSocket connection failed: {e}")


if __name__ == "__main__":
    client = GhostWebSocketClient()

    print("\n" + "=" * 70)
    print("  GHOST CLIENT — TLS FINGERPRINT SPOOFING ACTIVE (REST MODE)")
    print("=" * 70)
    print(f"  Target:      {REST_URL}")
    print(f"  TLS Spoof:   chrome120")
    print(f"  League:      NBA (id={NBA_LEAGUE_ID})")
    print("=" * 70 + "\n")

    # One-shot fetch to prove the pipeline works
    projections = client.fetch_projections_sync()

    if projections:
        print(f"{'─'*70}")
        print(f"  📊 LIVE PRIZEPICKS PROJECTIONS ({len(projections)} props)")
        print(f"{'─'*70}\n")

        # Print first 15 for terminal proof
        for i, p in enumerate(projections[:15]):
            odds_flag = " ⚡ BOOSTED" if p['adjusted_odds'] else ""
            live_flag = " 🔴 LIVE" if p['is_live'] else ""
            print(f"  {i+1:>2}. {p['player']:<25} | {p['stat_display']:<12} | "
                  f"Line: {p['line']:<6} | {p['odds_type']:<10}{odds_flag}{live_flag}")

        print(f"\n{'─'*70}")
        print(f"  Total NBA props on board: {len(projections)}")
        print(f"{'─'*70}\n")
    else:
        print("Failed to fetch projections. Check network/TLS configuration.")
