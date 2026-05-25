import os
import sys
import json
import logging
import asyncio
import re
from curl_cffi.requests import AsyncSession, Session

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infrastructure.entity_resolver import clean_player_name

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# ══════════════════════════════════════════════════════════════════════════════
# KALSHI CLIENT
# ══════════════════════════════════════════════════════════════════════════════
logger_kalshi = logging.getLogger("KalshiClob")
KALSHI_BASE_URL = "https://external-api.kalshi.com/trade-api/v2"
KALSHI_ORIGIN = "https://kalshi.com"
KALSHI_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/148.0.0.0 Safari/537.36"
)
KALSHI_SERIES_TICKERS = ["KXNBAPTS", "KXNBAAST", "KXNBAREB", "KXNBA3PT"]

class KalshiClobClient:
    def __init__(self):
        self.headers = {
            "Origin": KALSHI_ORIGIN,
            "User-Agent": KALSHI_USER_AGENT,
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
        }

    def parse_nba_contract(self, title: str):
        text = title.strip().rstrip('?').strip()

        def clean_stat(s: str) -> str:
            s_lower = s.lower().strip()
            if "point" in s_lower and "rebound" in s_lower and "assist" in s_lower:
                return "PRA"
            elif "point" in s_lower and "rebound" in s_lower:
                return "Pts+Rebs"
            elif "point" in s_lower and "assist" in s_lower:
                return "Pts+Asts"
            elif "rebound" in s_lower and "assist" in s_lower:
                return "Rebs+Asts"
            elif "point" in s_lower:
                return "Points"
            elif "assist" in s_lower:
                return "Assists"
            elif "rebound" in s_lower:
                return "Rebounds"
            elif "three" in s_lower or "3pt" in s_lower or "3-pointer" in s_lower or "threes" in s_lower:
                return "3-Pointers Made"
            elif "turnover" in s_lower:
                return "Turnovers"
            return s.title()

        def is_matchup(p: str) -> bool:
            p_lower = p.lower()
            return any(marker in p_lower for marker in [" vs ", " vs.", " v ", "@", "spread", "1h", "2h", "quarter", "half", "team", "total"])

        player, stat, line = None, None, None

        if ":" in text:
            parts = text.split(":", 1)
            player_candidate = parts[0].strip()
            if is_matchup(player_candidate):
                return None, None, None
            
            rest = parts[1].strip()
            
            match_ou = re.search(
                r"([\w\s\-+]+?)\s+(?:o/u|over/under|over\s+or\s+under)\s+(\d+(?:\.\d+)?)",
                rest, re.IGNORECASE
            )
            if match_ou:
                player = player_candidate
                stat = clean_stat(match_ou.group(1))
                line = float(match_ou.group(2))
            
            if line is None:
                match_kplus = re.search(
                    r"(\d+(?:\.\d+)?)\+\s*(.+)$",
                    rest, re.IGNORECASE
                )
                if match_kplus:
                    player = player_candidate
                    threshold = float(match_kplus.group(1))
                    stat = clean_stat(match_kplus.group(2))
                    line = threshold - 0.5
            
            if line is None:
                match_or_more = re.search(
                    r"(\d+(?:\.\d+)?)\s+or\s+more\s*(.+)$",
                    rest, re.IGNORECASE
                )
                if match_or_more:
                    player = player_candidate
                    threshold = float(match_or_more.group(1))
                    stat = clean_stat(match_or_more.group(2))
                    line = threshold - 0.5
                    
            if line is None:
                match_over_under = re.search(
                    r"(?:over|under)\s+(\d+(?:\.\d+)?)\s*(.+)$",
                    rest, re.IGNORECASE
                )
                if match_over_under:
                    player = player_candidate
                    line = float(match_over_under.group(1))
                    stat = clean_stat(match_over_under.group(2))
        else:
            match_will_over = re.search(
                r"^will\s+(.+?)\s+(?:score|record|make|get|have)\s+(?:over|under)\s+(\d+(?:\.\d+)?)\s+(.+)$",
                text, re.IGNORECASE
            )
            if match_will_over:
                player_candidate = match_will_over.group(1).strip()
                if not is_matchup(player_candidate):
                    player = player_candidate
                    line = float(match_will_over.group(2))
                    stat = clean_stat(match_will_over.group(3))

            if line is None:
                match_will_kplus = re.search(
                    r"^will\s+(.+?)\s+(?:score|record|make|get|have)\s+(\d+(?:\.\d+)?)\+\s*(.+)$",
                    text, re.IGNORECASE
                )
                if match_will_kplus:
                    player_candidate = match_will_kplus.group(1).strip()
                    if not is_matchup(player_candidate):
                        player = player_candidate
                        threshold = float(match_will_kplus.group(2))
                        stat = clean_stat(match_will_kplus.group(3))
                        line = threshold - 0.5

            if line is None:
                match_will_ormore = re.search(
                    r"^will\s+(.+?)\s+(?:score|record|make|get|have)\s+(\d+(?:\.\d+)?)\s+or\s+more\s+(.+)$",
                    text, re.IGNORECASE
                )
                if match_will_ormore:
                    player_candidate = match_will_ormore.group(1).strip()
                    if not is_matchup(player_candidate):
                        player = player_candidate
                        threshold = float(match_will_ormore.group(2))
                        stat = clean_stat(match_will_ormore.group(3))
                        line = threshold - 0.5

        if player and stat and line is not None:
            stat_clean = re.split(r"\s+(?:against|in\s+game|vs\.?|@)\s+", stat, flags=re.IGNORECASE)[0].strip()
            player_clean = clean_player_name(player)
            return player_clean, stat_clean, line

        return None, None, None

    async def fetch_current_lines(self, session: AsyncSession) -> dict:
        url = f"{KALSHI_BASE_URL}/markets"
        parsed_props = {}
        tasks = []
        for ticker in KALSHI_SERIES_TICKERS:
            params = {
                "limit": "100",
                "status": "open",
                "series_ticker": ticker
            }
            tasks.append(session.get(url, params=params))

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        for ticker, resp in zip(KALSHI_SERIES_TICKERS, responses):
            if isinstance(resp, Exception):
                logger_kalshi.error(f"Error fetching Kalshi series {ticker}: {resp}")
                continue

            if resp.status_code != 200:
                logger_kalshi.warning(f"Kalshi API returned {resp.status_code} for series {ticker}")
                continue

            try:
                data = resp.json()
                markets = data.get("markets", [])
                for m in markets:
                    title = m.get("title", "")
                    player, stat, line = self.parse_nba_contract(title)

                    if player and stat and line is not None:
                        yes_ask = float(m.get("yes_ask_dollars", 0.0))
                        yes_bid = float(m.get("yes_bid_dollars", 0.0))
                        no_ask = float(m.get("no_ask_dollars", 0.0))
                        no_bid = float(m.get("no_bid_dollars", 0.0))

                        if yes_ask > 0.0 or yes_bid > 0.0:
                            yes_ask_size = float(m.get("yes_ask_size_fp") or 0.0)
                            yes_bid_size = float(m.get("yes_bid_size_fp") or 0.0)
                            ask_usd = yes_ask * yes_ask_size
                            bid_usd = yes_bid * yes_bid_size
                            mid_prob = (yes_ask + yes_bid) / 2.0

                            if player not in parsed_props:
                                parsed_props[player] = {}
                            if stat not in parsed_props[player]:
                                parsed_props[player][stat] = {}
                            
                            parsed_props[player][stat][line] = {
                                "player": player,
                                "prop": stat,
                                "line": line,
                                "yes_ask": yes_ask,
                                "yes_bid": yes_bid,
                                "no_ask": no_ask,
                                "no_bid": no_bid,
                                "odds_type": "implied_prob",
                                "book": "Kalshi",
                                "prob": mid_prob,
                                "ask_liquidity_usd": ask_usd,
                                "bid_liquidity_usd": bid_usd
                            }
            except Exception as e:
                logger_kalshi.error(f"Error parsing Kalshi series {ticker} payload: {e}")

        logger_kalshi.info(f"Retrieved {len(parsed_props)} active translated players from Kalshi.")
        return parsed_props

    def fetch_projections_sync(self) -> list:
        with Session(impersonate="chrome120", headers=self.headers) as s:
            url = f"{KALSHI_BASE_URL}/markets"
            results = []
            for ticker in KALSHI_SERIES_TICKERS:
                params = {"limit": "100", "status": "open", "series_ticker": ticker}
                try:
                    resp = s.get(url, params=params)
                    if resp.status_code == 200:
                        markets = resp.json().get("markets", [])
                        for m in markets:
                            title = m.get("title", "")
                            player, stat, line = self.parse_nba_contract(title)
                            if player and stat and line is not None:
                                results.append({
                                    "ticker": m.get("ticker"),
                                    "player": player,
                                    "stat": stat,
                                    "line": line,
                                    "yes_ask": float(m.get("yes_ask_dollars", 0.0)),
                                    "yes_bid": float(m.get("yes_bid_dollars", 0.0))
                                })
                except Exception as e:
                    logger_kalshi.error(f"Sync fetch error for {ticker}: {e}")
            return results


# ══════════════════════════════════════════════════════════════════════════════
# POLYMARKET CLIENT
# ══════════════════════════════════════════════════════════════════════════════
logger_poly = logging.getLogger("PolymarketClob")
POLY_GAMMA_URL = "https://gamma-api.polymarket.com/events"
POLY_CLOB_URL = "https://clob.polymarket.com/book"
POLY_ORIGIN = "https://polymarket.com"
POLY_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/148.0.0.0 Safari/537.36"
)
POLY_NBA_SERIES_ID = 10345

class PolymarketClobClient:
    def __init__(self):
        self.headers = {
            "Origin": POLY_ORIGIN,
            "User-Agent": POLY_USER_AGENT,
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
        }

    def parse_nba_contract(self, title: str):
        text = title.strip().rstrip('?').strip()

        def clean_stat(s: str) -> str:
            s_lower = s.lower().strip()
            if "point" in s_lower and "rebound" in s_lower and "assist" in s_lower:
                return "PRA"
            elif "point" in s_lower and "rebound" in s_lower:
                return "Pts+Rebs"
            elif "point" in s_lower and "assist" in s_lower:
                return "Pts+Asts"
            elif "rebound" in s_lower and "assist" in s_lower:
                return "Rebs+Asts"
            elif "point" in s_lower:
                return "Points"
            elif "assist" in s_lower:
                return "Assists"
            elif "rebound" in s_lower:
                return "Rebounds"
            elif "three" in s_lower or "3pt" in s_lower or "3-pointer" in s_lower or "threes" in s_lower:
                return "3-Pointers Made"
            elif "turnover" in s_lower:
                return "Turnovers"
            return s.title()

        def is_matchup(p: str) -> bool:
            p_lower = p.lower()
            return any(marker in p_lower for marker in [" vs ", " vs.", " v ", "@", "spread", "1h", "2h", "quarter", "half", "team", "total"])

        player, stat, line = None, None, None

        if ":" in text:
            parts = text.split(":", 1)
            player_candidate = parts[0].strip()
            if is_matchup(player_candidate):
                return None, None, None
            
            rest = parts[1].strip()
            
            match_ou = re.search(
                r"([\w\s\-+]+?)\s+(?:o/u|over/under|over\s+or\s+under)\s+(\d+(?:\.\d+)?)",
                rest, re.IGNORECASE
            )
            if match_ou:
                player = player_candidate
                stat = clean_stat(match_ou.group(1))
                line = float(match_ou.group(2))
            
            if line is None:
                match_kplus = re.search(
                    r"(\d+(?:\.\d+)?)\+\s*(.+)$",
                    rest, re.IGNORECASE
                )
                if match_kplus:
                    player = player_candidate
                    threshold = float(match_kplus.group(1))
                    stat = clean_stat(match_kplus.group(2))
                    line = threshold - 0.5
            
            if line is None:
                match_or_more = re.search(
                    r"(\d+(?:\.\d+)?)\s+or\s+more\s*(.+)$",
                    rest, re.IGNORECASE
                )
                if match_or_more:
                    player = player_candidate
                    threshold = float(match_or_more.group(1))
                    stat = clean_stat(match_or_more.group(2))
                    line = threshold - 0.5
                    
            if line is None:
                match_over_under = re.search(
                    r"(?:over|under)\s+(\d+(?:\.\d+)?)\s*(.+)$",
                    rest, re.IGNORECASE
                )
                if match_over_under:
                    player = player_candidate
                    line = float(match_over_under.group(1))
                    stat = clean_stat(match_over_under.group(2))
        else:
            match_will_over = re.search(
                r"^will\s+(.+?)\s+(?:score|record|make|get|have)\s+(?:over|under)\s+(\d+(?:\.\d+)?)\s+(.+)$",
                text, re.IGNORECASE
            )
            if match_will_over:
                player_candidate = match_will_over.group(1).strip()
                if not is_matchup(player_candidate):
                    player = player_candidate
                    line = float(match_will_over.group(2))
                    stat = clean_stat(match_will_over.group(3))

            if line is None:
                match_will_kplus = re.search(
                    r"^will\s+(.+?)\s+(?:score|record|make|get|have)\s+(\d+(?:\.\d+)?)\+\s*(.+)$",
                    text, re.IGNORECASE
                )
                if match_will_kplus:
                    player_candidate = match_will_kplus.group(1).strip()
                    if not is_matchup(player_candidate):
                        player = player_candidate
                        threshold = float(match_will_kplus.group(2))
                        stat = clean_stat(match_will_kplus.group(3))
                        line = threshold - 0.5

            if line is None:
                match_will_ormore = re.search(
                    r"^will\s+(.+?)\s+(?:score|record|make|get|have)\s+(\d+(?:\.\d+)?)\s+or\s+more\s+(.+)$",
                    text, re.IGNORECASE
                )
                if match_will_ormore:
                    player_candidate = match_will_ormore.group(1).strip()
                    if not is_matchup(player_candidate):
                        player = player_candidate
                        threshold = float(match_will_ormore.group(2))
                        stat = clean_stat(match_will_ormore.group(3))
                        line = threshold - 0.5

        if player and stat and line is not None:
            stat_clean = re.split(r"\s+(?:against|in\s+game|vs\.?|@)\s+", stat, flags=re.IGNORECASE)[0].strip()
            player_clean = clean_player_name(player)
            return player_clean, stat_clean, line

        return None, None, None

    async def fetch_book(self, session: AsyncSession, token_id: str) -> dict | None:
        try:
            url = f"{POLY_CLOB_URL}?token_id={token_id}"
            resp = await session.get(url, timeout=5.0)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger_poly.debug(f"Error fetching Polymarket CLOB book for {token_id}: {e}")
        return None

    async def fetch_current_lines(self, session: AsyncSession) -> dict:
        url = POLY_GAMMA_URL
        params = {
            "series_id": str(POLY_NBA_SERIES_ID),
            "active": "true",
            "closed": "false"
        }

        parsed_props = {}
        try:
            resp = await session.get(url, params=params, timeout=10.0)
            if resp.status_code != 200:
                logger_poly.warning(f"Polymarket Gamma API returned status {resp.status_code}")
                return {}

            events = resp.json()
        except Exception as e:
            logger_poly.error(f"Failed to fetch Polymarket Gamma API: {e}")
            return {}

        markets_to_fetch = []
        for event in events:
            for m in event.get("markets", []):
                question = m.get("question", "")
                player, stat, line = self.parse_nba_contract(question)

                if player and stat and line is not None:
                    clob_tokens = []
                    clob_str = m.get("clobTokenIds", "")
                    if clob_str:
                        try:
                            clob_tokens = json.loads(clob_str)
                        except Exception:
                            pass

                    outcome_prices_str = m.get("outcomePrices")
                    gamma_yes = None
                    gamma_no = None
                    if outcome_prices_str:
                        try:
                            if isinstance(outcome_prices_str, str):
                                outcome_prices = json.loads(outcome_prices_str)
                            else:
                                outcome_prices = outcome_prices_str
                            if outcome_prices and len(outcome_prices) >= 2:
                                gamma_yes = float(outcome_prices[0])
                                gamma_no = float(outcome_prices[1])
                        except Exception:
                            pass

                    markets_to_fetch.append({
                        "market_id": m.get("id"),
                        "player": player,
                        "prop": stat,
                        "line": line,
                        "yes_token": clob_tokens[0] if clob_tokens else None,
                        "gamma_yes": gamma_yes,
                        "gamma_no": gamma_no
                    })

        if not markets_to_fetch:
            return {}

        tasks = []
        for item in markets_to_fetch:
            if item["yes_token"]:
                tasks.append(self.fetch_book(session, item["yes_token"]))
            else:
                tasks.append(asyncio.sleep(0, result=None))

        books = await asyncio.gather(*tasks, return_exceptions=True)

        for item, book_data in zip(markets_to_fetch, books):
            player = item["player"]
            stat = item["prop"]
            line = item["line"]

            yes_bid = 0.0
            yes_ask = 0.0
            no_bid = 0.0
            no_ask = 0.0
            ask_usd = 0.0
            bid_usd = 0.0

            resolved_via_clob = False

            if book_data and not isinstance(book_data, Exception):
                bids = book_data.get("bids", [])
                asks = book_data.get("asks", [])
                if bids or asks:
                    if bids:
                        best_bid = max(bids, key=lambda x: float(x.get("price", 0.0)))
                        yes_bid = float(best_bid.get("price", 0.0))
                        yes_bid_size = float(best_bid.get("size", 0.0))
                        bid_usd = yes_bid * yes_bid_size
                    if asks:
                        best_ask = min(asks, key=lambda x: float(x.get("price", 0.0)))
                        yes_ask = float(best_ask.get("price", 0.0))
                        yes_ask_size = float(best_ask.get("size", 0.0))
                        ask_usd = yes_ask * yes_ask_size
                    no_bid = 1.0 - yes_ask if yes_ask > 0.0 else 0.0
                    no_ask = 1.0 - yes_bid if yes_bid > 0.0 else 0.0
                    resolved_via_clob = True

            if not resolved_via_clob or (yes_ask == 0.0 and yes_bid == 0.0):
                if item["gamma_yes"] is not None:
                    yes_bid = item["gamma_yes"]
                    yes_ask = item["gamma_yes"]
                    no_bid = item["gamma_no"] if item["gamma_no"] is not None else 1.0 - yes_bid
                    no_ask = no_bid
                    ask_usd = 0.0
                    bid_usd = 0.0

            if yes_ask > 0.0 or yes_bid > 0.0:
                mid_prob = (yes_ask + yes_bid) / 2.0
                if player not in parsed_props:
                    parsed_props[player] = {}
                if stat not in parsed_props[player]:
                    parsed_props[player][stat] = {}
                
                parsed_props[player][stat][line] = {
                    "player": player,
                    "prop": stat,
                    "line": line,
                    "yes_ask": yes_ask,
                    "yes_bid": yes_bid,
                    "no_ask": no_ask,
                    "no_bid": no_bid,
                    "odds_type": "implied_prob",
                    "book": "Polymarket",
                    "prob": mid_prob,
                    "ask_liquidity_usd": ask_usd,
                    "bid_liquidity_usd": bid_usd
                }

        logger_poly.info(f"Retrieved {len(parsed_props)} active translated player props from Polymarket.")
        return parsed_props

    def fetch_projections_sync(self) -> list:
        with Session(impersonate="chrome120", headers=self.headers) as s:
            url = POLY_GAMMA_URL
            params = {
                "series_id": str(POLY_NBA_SERIES_ID),
                "active": "true",
                "closed": "false"
            }
            results = []
            try:
                resp = s.get(url, params=params)
                if resp.status_code == 200:
                    events = resp.json()
                    for event in events:
                        for m in event.get("markets", []):
                            question = m.get("question", "")
                            player, stat, line = self.parse_nba_contract(question)
                            if player and stat and line is not None:
                                outcome_prices_str = m.get("outcomePrices")
                                yes_price = 0.0
                                if outcome_prices_str:
                                    try:
                                        if isinstance(outcome_prices_str, str):
                                            outcome_prices = json.loads(outcome_prices_str)
                                        else:
                                            outcome_prices = outcome_prices_str
                                        if outcome_prices and len(outcome_prices) >= 2:
                                            yes_price = float(outcome_prices[0])
                                    except Exception:
                                        pass
                                results.append({
                                    "market_id": m.get("id"),
                                    "player": player,
                                    "stat": stat,
                                    "line": line,
                                    "yes_price": yes_price,
                                    "question": question
                                })
            except Exception as e:
                logger_poly.error(f"Polymarket sync fetch error: {e}")
            return results


# ══════════════════════════════════════════════════════════════════════════════
# UNDERDOG GHOST CLIENT
# ══════════════════════════════════════════════════════════════════════════════
logger_ud = logging.getLogger("UnderdogGhost")
UD_REST_URL = "https://api.underdogfantasy.com/v1/over_under_lines"
UD_ORIGIN = "https://underdogfantasy.com"
UD_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/110.0.0.0 Safari/537.36"
)

class UnderdogGhostClient:
    def __init__(self):
        self.headers = {
            "Origin": UD_ORIGIN,
            "Referer": f"{UD_ORIGIN}/pick-em",
            "User-Agent": UD_USER_AGENT,
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        }
        self.message_count = 0
        self.data_messages = 0
        self.projections_cache = {}

    def fetch_projections_sync(self) -> list:
        with Session(impersonate="chrome110", headers=self.headers) as session:
            try:
                resp = session.get(UD_REST_URL)
                if resp.status_code != 200:
                    logger_ud.error(f"Underdog REST API returned {resp.status_code}")
                    return []

                raw = resp.json()
                projections = self._parse_projections(raw)
                logger_ud.info(f"Fetched {len(projections)} projections from Underdog Fantasy")
                return projections
            except Exception as e:
                logger_ud.error(f"Underdog sync fetch error: {e}")
                return []

    async def poll_projections(self, interval_seconds: int = 30, max_cycles: int = 0):
        cycle = 0
        async with AsyncSession(impersonate="chrome110", headers=self.headers) as session:
            while max_cycles == 0 or cycle < max_cycles:
                cycle += 1
                try:
                    resp = await session.get(UD_REST_URL)

                    if resp.status_code != 200:
                        logger_ud.warning(f"[Cycle {cycle}] Underdog API returned {resp.status_code}")
                        await asyncio.sleep(interval_seconds)
                        continue

                    raw = resp.json()
                    projections = self._parse_projections(raw)
                    self.message_count += 1

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
                        logger_ud.debug(f"[Cycle {cycle}] {len(projections)} projections. No movements.")

                    for p in projections:
                        self.projections_cache[p['id']] = p

                except Exception as e:
                    logger_ud.error(f"[Cycle {cycle}] Underdog polling error: {e}")

                await asyncio.sleep(interval_seconds)

    async def fetch_current_lines(self, session: AsyncSession) -> dict:
        try:
            resp = await session.get(UD_REST_URL)
            if resp.status_code != 200:
                logger_ud.warning(f"Underdog API returned {resp.status_code}")
                return {}

            raw = resp.json()
            projections = self._parse_projections(raw)

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
            logger_ud.error(f"Underdog fetch error: {e}")
            return {}

    def _parse_projections(self, raw_json: dict) -> list:
        results = []
        over_under_lines = raw_json.get("over_under_lines", [])
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

            over_under = line.get("over_under", line)
            appearance_stat = over_under.get("appearance_stat", {})
            if not appearance_id:
                appearance_id = appearance_stat.get("appearance_id", "")

            player_id = appearances_map.get(appearance_id, "")
            player_info = players_map.get(player_id, {})
            player_name_raw = f"{player_info.get('first_name', '')} {player_info.get('last_name', '')}".strip()
            player_name = clean_player_name(player_name_raw) if player_name_raw else "Unknown"

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
