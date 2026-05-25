"""
High-Frequency Poller (HFP) — Master Diff-Engine Brain
========================================================
Dual-Tier Scraping Architecture:
  Tier 1 (Free/Retail): curl_cffi pollers for PrizePicks + Underdog Fantasy
  Tier 2 (Paid/Sharp):  Cached ParlayAPI for Pinnacle/DraftKings devigged lines

Orchestration:
  1. Every 30 seconds, poll PrizePicks and Underdog concurrently via asyncio.gather.
  2. Load the latest devigged Pinnacle true_probs from sharp_cache.json.
  3. For every retail prop, diff against the sharp anchor.
  4. If ev_edge >= 5.0%, fire DiscordAlertEngine + cache signal in Redis.

Core Model Pipeline (preserved from Sprint 10):
  - ZINBModel (true probability distribution)
  - PlayerProjectionModel (XGBoost-based minutes/usage projection)
  - ShinDevigger (sharp odds de-vigging)
  - EVCalculator + BetMaskingEngine (Kelly sizing + stake camouflage)
"""
import os
import sys
# Dynamic project root path resolution
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import json
import logging
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from curl_cffi.requests import AsyncSession
import psycopg2
from dotenv import load_dotenv
load_dotenv()

from src.models import ZINBModel, PlayerProjectionModel, ShinDevigger, EVCalculator, BetMaskingEngine
from src.slip_builder import CorrelatedSlipEvaluator
from infrastructure.entity_resolver import EntityResolver, clean_player_name
from infrastructure.redis_client import RedisClient

# Import retail pollers and exchanges
from scrapers.exchanges import UnderdogGhostClient, KalshiClobClient, PolymarketClobClient

# Import sharp line provider
from scrapers.unified_api import SharpLineProvider

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("HFP")


class UnifiedRestPoller:
    """
    Master Diff-Engine Brain for the Dual-Tier Scraping Architecture.

    Orchestrates concurrent polling of PrizePicks and Underdog Fantasy (Retail),
    loads devigged Pinnacle sharp lines from the cached ParlayAPI provider,
    and runs the EV diff engine to detect +EV opportunities and fire alerts.
    """

    def __init__(self):
        # ── Retail Book URLs ──────────────────────────────────
        self.pp_url = "https://api.prizepicks.com/projections"

        self.headers = {
            "Accept-Language": "en-US,en;q=0.9",
        }

        # ── State tracking for diff engine ────────────────────
        # Keyed by retail book: { 'pp': {prop_id: data}, 'ud': {prop_id: data}, 'kalshi': {prop_id: data}, 'polymarket': {prop_id: data} }
        self.last_state = {
            'pp': {},
            'ud': {},
            'kalshi': {},
            'polymarket': {}
        }

        # ── Retail & Exchange Pollers ─────────────────────────
        self.ud_client = UnderdogGhostClient()
        self.kalshi_client = KalshiClobClient()
        self.polymarket_client = PolymarketClobClient()

        # ── Sharp Line Provider (Tier 2) ──────────────────────
        self.sharp_provider = SharpLineProvider()
        self.sharp_cache = {}  # In-memory sharp lines loaded each cycle
        self.anchors = {}      # Combined anchors mapping for each cycle

        # Force sync seed on startup if cache is missing or empty
        cache_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sharp_cache.json")
        needs_seeding = True
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                lines = cache_data.get("lines", {})
                if lines and len(lines) > 0:
                    needs_seeding = False
                    logger.info(f"Existing sharp cache detected with {len(lines)} players. Skipping boot-time sync seed.")
            except Exception:
                pass
        
        if needs_seeding:
            logger.info("Sharp cache is missing or empty. Forcing synchronous boot-time seed...")
            self.sharp_cache = self.sharp_provider.force_sync_seed()

        # Caches for history and models to avoid redundant PG hits
        self.history_cache = {}
        self.zinb_cache = {}

        # ── Discord Alerting Engine ───────────────────────────
        self.min_ev_threshold = 5.0  # 5.0% minimum EV edge threshold

        # ── Database connectivity & Entity Resolution ─────────
        self.db_uri = os.getenv("POSTGRES_URI", "postgresql://postgres:password@localhost:5432/sports_db")
        self.resolver = EntityResolver()

        # ── Slip Builder & SGP Evaluator ─────────────────────
        try:
            self.slip_evaluator = CorrelatedSlipEvaluator(db_uri=self.db_uri)
            logger.info("CorrelatedSlipEvaluator initialized and player games cache loaded.")
        except Exception as se:
            logger.error(f"Failed to initialize CorrelatedSlipEvaluator: {se}")
            self.slip_evaluator = None

    # ══════════════════════════════════════════════════════════
    # UTILITY METHODS
    # ══════════════════════════════════════════════════════════

    def _map_stat_type(self, prop_name: str) -> str:
        """Maps API prop category string to db column name."""
        if not prop_name:
            return None
        prop_lower = prop_name.lower().strip()
        if 'point' in prop_lower:
            return 'points'
        elif 'rebound' in prop_lower:
            return 'rebounds'
        elif 'assist' in prop_lower:
            return 'assists'
        elif 'turnover' in prop_lower:
            return 'turnovers'
        return None

    # ══════════════════════════════════════════════════════════
    # DATABASE & MODEL PIPELINE (Preserved from Sprint 10)
    # ══════════════════════════════════════════════════════════

    def get_player_history(self, player_id: str) -> pd.DataFrame:
        """Queries up to the last 100 game logs for the target player from PostgreSQL."""
        try:
            conn = psycopg2.connect(self.db_uri)
            query = """
                SELECT minutes_played, points, rebounds, assists, turnovers
                FROM player_game_logs
                WHERE player_id = %s
                ORDER BY game_id ASC
                LIMIT 100;
            """
            df = pd.read_sql(query, conn, params=(player_id,))
            conn.close()
            return df
        except Exception as e:
            logger.error(f"Error fetching history for player {player_id}: {e}")
            return pd.DataFrame()

    def project_player_baseline(self, history_df: pd.DataFrame, stat_col: str) -> dict:
        """
        Engineers rolling 3-game average features and dynamically trains
        an XGBoost PlayerProjectionModel to project minutes and usage.
        """
        if len(history_df) < 5:
            return {'xMin': 25.0, 'xUSG': 0.20}

        try:
            df = history_df.copy()
            df['usage_rate'] = (
                (df['points'] + 1.5 * df['assists'] + df['turnovers'])
                / (df['minutes_played'] * 2.0 + 1e-5)
            )
            df['usage_rate'] = df['usage_rate'].clip(0.05, 0.45)

            df['roll_min'] = df['minutes_played'].shift(1).rolling(3).mean()
            df['roll_stat'] = df[stat_col].shift(1).rolling(3).mean()
            df['roll_usg'] = df['usage_rate'].shift(1).rolling(3).mean()

            train_df = df.dropna(subset=['roll_min', 'roll_stat', 'roll_usg', 'minutes_played', 'usage_rate'])

            if len(train_df) < 2:
                return {'xMin': 25.0, 'xUSG': 0.20}

            X_train = train_df[['roll_min', 'roll_stat', 'roll_usg']].values
            y_min = train_df['minutes_played'].values
            y_usg = train_df['usage_rate'].values

            model = PlayerProjectionModel()
            model.fit(X_train, y_min, y_usg)

            last_3 = df.tail(3)
            pred_features = np.array([[
                last_3['minutes_played'].mean(),
                last_3[stat_col].mean(),
                last_3['usage_rate'].mean()
            ]])

            pred = model.predict(pred_features)
            return pred
        except Exception as e:
            logger.error(f"XGBoost baseline projection error: {e}")
            return {'xMin': 25.0, 'xUSG': 0.20}

    # ══════════════════════════════════════════════════════════
    # RETAIL BOOK FETCHERS
    # ══════════════════════════════════════════════════════════

    async def fetch_pp(self, session: AsyncSession) -> dict:
        """Fetch PrizePicks unauthenticated projections."""
        try:
            params = {"league_id": "7", "per_page": "250"}
            resp = await session.get(self.pp_url, params=params)
            if resp.status_code == 200:
                raw = resp.json()
                return self._parse_pp(raw)
            else:
                logger.warning(f"PP API returned {resp.status_code}")
                return {}
        except Exception as e:
            logger.error(f"PP fetch error: {e}")
            return {}

    async def fetch_ud(self, session: AsyncSession) -> dict:
        """Fetch Underdog Fantasy lines via the ghost client."""
        try:
            return await self.ud_client.fetch_current_lines(session)
        except Exception as e:
            logger.error(f"Underdog fetch error: {e}")
            return {}

    def _parse_pp(self, raw: dict) -> dict:
        """Extract odds from PP structure."""
        parsed = {}
        data = raw.get("data", [])
        included = raw.get("included", [])

        included_map = {}
        for item in included:
            key = (item.get("type"), item.get("id"))
            included_map[key] = item.get("attributes", {})

        for entry in data:
            proj_id = entry.get("id")
            attrs = entry.get("attributes", {})
            rels = entry.get("relationships", {})

            player_rel = rels.get("new_player", {}).get("data", {})
            player_attrs = included_map.get((player_rel.get("type"), player_rel.get("id")), {})
            player_name = player_attrs.get("display_name") or player_attrs.get("name", "Unknown")

            if proj_id:
                parsed[proj_id] = {
                    'player': clean_player_name(player_name),
                    'prop': attrs.get("stat_display_name", ""),
                    'line': attrs.get("line_score"),
                    'odds_type': attrs.get("odds_type", "standard")
                }
        return parsed

    def calculate_theoretical_prob(self, player_name: str, stat_col: str, line: float, master_id: str = None) -> tuple[float, ZINBModel]:
        """
        Fits the Zero-Inflated Negative Binomial distribution to the player's historical rolling baseline
        and returns the model-implied Over probability and the fitted ZINBModel.
        """
        if not master_id:
            for book_opt in ["PrizePicks", "Underdog", "Kalshi", "Polymarket"]:
                master_id = self.resolver.resolve_player(book_opt, player_name)
                if master_id:
                    break
            if not master_id:
                master_id = player_name

        cache_key = (master_id, stat_col)
        if cache_key in self.zinb_cache:
            zinb, pred = self.zinb_cache[cache_key]
            logger.debug(f"[{player_name}] Using cached ZINB and XGBoost baseline.")
        else:
            history_df = self.get_player_history(master_id)
            zinb = ZINBModel()

            if not history_df.empty:
                pred = self.project_player_baseline(history_df, stat_col)

                stat_series = history_df[stat_col]
                zinb.fit(stat_series)

                hist_mean_min = history_df['minutes_played'].mean()
                history_df['usage_rate'] = (
                    (history_df['points'] + 1.5 * history_df['assists'] + history_df['turnovers'])
                    / (history_df['minutes_played'] * 2.0 + 1e-5)
                )
                history_df['usage_rate'] = history_df['usage_rate'].clip(0.05, 0.45)
                hist_mean_usg = history_df['usage_rate'].mean()

                min_ratio = pred['xMin'] / (hist_mean_min + 1e-5)
                usg_ratio = pred['xUSG'] / (hist_mean_usg + 1e-5)
                min_ratio = np.clip(min_ratio, 0.5, 1.5)
                usg_ratio = np.clip(usg_ratio, 0.5, 1.5)

                zinb.mu = zinb.mu * min_ratio * usg_ratio
                logger.info(f"[{player_name}] ZINB mu scaled to: {zinb.mu:.4f}")
            else:
                logger.warning(f"No history for '{player_name}' (ID: {master_id}). Using fallback.")
                zinb.mu = line
                zinb.pi = 0.03
                zinb.n = 15.0
                pred = {'xMin': 25.0, 'xUSG': 0.20}

            self.zinb_cache[cache_key] = (zinb, pred)

        # Condition ZINB parameters
        zinb.condition_parameters(blowout_risk=0.10, pace_factor=1.0)

        prob_over = zinb.predict_over_probability(line)
        return prob_over, zinb

    # ══════════════════════════════════════════════════════════
    # DIFF ENGINE (Core EV Detection)
    # ══════════════════════════════════════════════════════════

    async def _diff_engine(self, book: str, new_state: dict):
        """
        Compares new state with last state, generates Synthetic Ticks,
        runs the full ZINB/XGBoost model pipeline, and alerts on +EV.
        """
        old_state = self.last_state[book]
        ticks = []

        if not old_state:
            self.last_state[book] = new_state
            logger.info(f"Seeded initial state for {book} ({len(new_state)} props)")
            return ticks

        for prop_id, new_data in new_state.items():
            if prop_id in old_state:
                old_data = old_state[prop_id]

                # Check for line movement or price change
                if new_data != old_data:
                    tick = {
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        'book': book,
                        'player': new_data['player'],
                        'prop': new_data['prop'],
                        'old': old_data,
                        'new': new_data
                    }
                    ticks.append(tick)

                    print(f"\n{'='*60}")
                    print(f" 🚨 SYNTHETIC TICK | {book.upper()} | {new_data['player']} - {new_data['prop']}")
                    print(f"{'='*60}")
                    print(f" Old: {old_data}")
                    print(f" New: {new_data}\n")

                    # Extract line value
                    raw_line = float(new_data.get('line', 15.5) or 15.5)

                    # Map prop string to db stat column
                    stat_col = self._map_stat_type(new_data['prop'])
                    if not stat_col:
                        continue

                    # Look up anchors for this player/stat using normalized player name
                    player_clean = clean_player_name(new_data['player'])
                    player_anchors = self.anchors.get(player_clean, {})
                    available_stat_lines = player_anchors.get(stat_col, {})

                    # Find matching line in anchors (with 0.01 tolerance)
                    matching_line_key = None
                    for line_key in available_stat_lines.keys():
                        try:
                            line_val = float(line_key)
                        except (ValueError, TypeError):
                            continue
                        if abs(line_val - raw_line) < 0.01:
                            matching_line_key = line_key
                            break

                    # Retrieve matching anchor dict { book: {'prob': prob, ...} }
                    matching_anchors = available_stat_lines.get(matching_line_key, {}) if matching_line_key is not None else {}

                    # Resolve player name to database master ID
                    mapped_book = "PrizePicks" if book == 'pp' else "Underdog" if book == 'ud' else "Kalshi" if book == 'kalshi' else "Polymarket"
                    master_id = self.resolver.resolve_player(mapped_book, new_data['player'])

                    # Calculate ZINB theoretical probability and obtain fitted model
                    theoretical_prob, zinb = self.calculate_theoretical_prob(new_data['player'], stat_col, raw_line, master_id)

                    # Determine target cost/probability and american odds based on book/odds_type
                    if book in ('pp', 'ud'):
                        # Standard DFS implied cost/odds (-119)
                        target_yes_american = -119
                        target_no_american = -119
                        target_yes_prob = 0.5434
                        target_no_prob = 0.5434
                    elif book in ('kalshi', 'polymarket'):
                        # Exchange markets: cost to buy YES is yes_ask, cost to buy NO is 1.0 - yes_bid
                        target_yes_prob = float(new_data.get('yes_ask', 0.0))
                        target_no_prob = 1.0 - float(new_data.get('yes_bid', 0.0))
                        target_yes_american = self._prob_to_american(target_yes_prob)
                        target_no_american = self._prob_to_american(target_no_prob)
                    else:
                        continue

                    MIN_LIQUIDITY_USD = 100.00

                    # 1. Gather all candidate exchange anchors (excluding the target book itself)
                    exchange_candidates = []
                    for abook, adata in matching_anchors.items():
                        if abook == 'Pinnacle':
                            continue
                        if abook.lower() == book.lower():  # Prevent self-comparison
                            continue
                        exchange_candidates.append((abook, adata))

                    # 2. Filter to those that pass the spoofing safeguard: min(ask, bid) >= MIN_LIQUIDITY_USD
                    valid_exchange_anchors = []
                    for abook, adata in exchange_candidates:
                        ask_liq = adata.get('ask_liquidity_usd') or 0.0
                        bid_liq = adata.get('bid_liquidity_usd') or 0.0
                        min_liq = min(ask_liq, bid_liq)
                        if min_liq >= MIN_LIQUIDITY_USD:
                            valid_exchange_anchors.append((abook, adata, min_liq))
                        else:
                            logger.info(
                                f"Thin prediction market line rejected: {new_data['player']} on {abook} "
                                f"(ask_liq=${ask_liq:.2f}, bid_liq=${bid_liq:.2f} < ${MIN_LIQUIDITY_USD}). "
                                f"Falling back to Pinnacle 3-hour cache."
                            )

                    # 3. Select the best anchor based on cascade
                    chosen_anchor_book = None
                    chosen_anchor_data = None

                    if valid_exchange_anchors:
                        # If multiple, choose the one with the highest min_liq
                        valid_exchange_anchors.sort(key=lambda x: x[2], reverse=True)
                        chosen_anchor_book = valid_exchange_anchors[0][0]
                        chosen_anchor_data = valid_exchange_anchors[0][1]
                    elif 'Pinnacle' in matching_anchors:
                        chosen_anchor_book = 'Pinnacle'
                        chosen_anchor_data = matching_anchors['Pinnacle']

                    if not chosen_anchor_book or not chosen_anchor_data:
                        # No valid sharp anchor found: Fallback to ZINB theoretical model!
                        chosen_anchor_book = "ZINB"
                        sharp_prob = theoretical_prob
                        anchor_liquidity_usd = None
                        sharp_line = str(raw_line)
                    else:
                        # Extract probability and liquidity
                        sharp_prob = chosen_anchor_data['prob']
                        sharp_line = matching_line_key
                        if chosen_anchor_book == 'Pinnacle':
                            anchor_liquidity_usd = None
                        else:
                            ask_liq = chosen_anchor_data.get('ask_liquidity_usd') or 0.0
                            bid_liq = chosen_anchor_data.get('bid_liquidity_usd') or 0.0
                            anchor_liquidity_usd = min(ask_liq, bid_liq)

                    # ── Evaluation 1: YES / Over Contract ─────────────────
                    if target_yes_prob > 0.0:
                        ev_yes = 0.0
                        # If the exchange itself is the execution target, validate the executable side
                        if book in ('kalshi', 'polymarket'):
                            target_ask_liq = new_data.get('ask_liquidity_usd') or 0.0
                            if target_ask_liq < MIN_LIQUIDITY_USD:
                                logger.info(
                                    f"Skipping YES execution on {book}: "
                                    f"Target ask liquidity (${target_ask_liq:.2f}) < ${MIN_LIQUIDITY_USD}."
                                )
                                ev_yes_pct = 0.0
                            else:
                                ev_yes = (sharp_prob / target_yes_prob) - 1.0
                                ev_yes_pct = ev_yes * 100.0
                        else:
                            ev_yes = (sharp_prob / target_yes_prob) - 1.0
                            ev_yes_pct = ev_yes * 100.0

                        if ev_yes_pct >= self.min_ev_threshold:
                            # Net odds received
                            b = (1.0 / target_yes_prob) - 1.0
                            kelly_yes = (sharp_prob - (1.0 - sharp_prob) / b) * 0.25 if b > 0 else 0.0
                            kelly_yes = min(max(kelly_yes, 0.0), 0.05)

                            await self._fire_alert_and_save(
                                book=book,
                                player_name=new_data['player'],
                                player_id=master_id,
                                prop_name=new_data['prop'],
                                side="Over",
                                raw_line=raw_line,
                                sharp_book=chosen_anchor_book,
                                sharp_line=sharp_line,
                                sharp_prob=sharp_prob,
                                target_cost=target_yes_prob,
                                target_american=target_yes_american,
                                ev_edge=ev_yes,
                                kelly_fraction=kelly_yes,
                                zinb_mu=zinb.mu,
                                zinb_pi=zinb.pi,
                                zinb_n=zinb.n,
                                anchor_liquidity_usd=anchor_liquidity_usd
                            )

                    # ── Evaluation 2: NO / Under Contract ─────────────────
                    if target_no_prob > 0.0:
                        ev_no = 0.0
                        # If the exchange itself is the execution target, validate the executable side
                        if book in ('kalshi', 'polymarket'):
                            target_bid_liq = new_data.get('bid_liquidity_usd') or 0.0
                            if target_bid_liq < MIN_LIQUIDITY_USD:
                                logger.info(
                                    f"Skipping NO execution on {book}: "
                                    f"Target bid liquidity (${target_bid_liq:.2f}) < ${MIN_LIQUIDITY_USD}."
                                )
                                ev_no_pct = 0.0
                            else:
                                ev_no = ((1.0 - sharp_prob) / target_no_prob) - 1.0
                                ev_no_pct = ev_no * 100.0
                        else:
                            ev_no = ((1.0 - sharp_prob) / target_no_prob) - 1.0
                            ev_no_pct = ev_no * 100.0

                        if ev_no_pct >= self.min_ev_threshold:
                            # Net odds received
                            b = (1.0 / target_no_prob) - 1.0
                            kelly_no = ((1.0 - sharp_prob) - sharp_prob / b) * 0.25 if b > 0 else 0.0
                            kelly_no = min(max(kelly_no, 0.0), 0.05)

                            await self._fire_alert_and_save(
                                book=book,
                                player_name=new_data['player'],
                                player_id=master_id,
                                prop_name=new_data['prop'],
                                side="Under",
                                raw_line=raw_line,
                                sharp_book=chosen_anchor_book,
                                sharp_line=sharp_line,
                                sharp_prob=1.0 - sharp_prob,
                                target_cost=target_no_prob,
                                target_american=target_no_american,
                                ev_edge=ev_no,
                                kelly_fraction=kelly_no,
                                zinb_mu=zinb.mu,
                                zinb_pi=zinb.pi,
                                zinb_n=zinb.n,
                                anchor_liquidity_usd=anchor_liquidity_usd
                            )

        self.last_state[book] = new_state
        return ticks

    def _prob_to_american(self, prob: float) -> int:
        """Converts a probability decimal to American odds."""
        if prob <= 0.0 or prob >= 1.0:
            return -110
        if prob > 0.5:
            return int(round(-100.0 * prob / (1.0 - prob)))
        else:
            return int(round(100.0 * (1.0 - prob) / prob))

    async def _fire_alert_and_save(self, book, player_name, player_id, prop_name, side, raw_line,
                                   sharp_book, sharp_line, sharp_prob, target_cost, target_american,
                                   ev_edge, kelly_fraction, zinb_mu, zinb_pi, zinb_n, anchor_liquidity_usd=None):
        """Dispatches alerts, caches signals in Redis, and stores in Postgres database."""
        mapped_book = "PrizePicks" if book == 'pp' else "Underdog" if book == 'ud' else "Kalshi" if book == 'kalshi' else "Polymarket"
        retail_line = f"Line {raw_line} ({side})"

        # 1. Fire Discord Alert
        raw_stake = kelly_fraction * 50000.0
        masked_stake = BetMaskingEngine.mask_kelly_stake(raw_stake)

        logger.info(f"+EV {side} edge ({ev_edge*100.0:.2f}%) on {mapped_book} crosses threshold. Firing alert...")
        
        logger.info(
            f"Alert Event: player={player_name}, prop={prop_name}, "
            f"sharp_book={sharp_book}, retail_book={mapped_book}, "
            f"retail_line={retail_line}, ev_edge={ev_edge:.2%}, "
            f"masked_stake=${float(masked_stake):.2f}, "
            f"liquidity=${anchor_liquidity_usd}"
        )

        # 2. Cache in Redis
        try:
            r = RedisClient().get_client()
            signal_key = f"signal:{book}:{player_name}:{prop_name}:{side}"
            r.hset(signal_key, mapping={
                "Player": player_name,
                "Market": prop_name,
                "Side": side,
                "Sharp Book": sharp_book,
                "Sharp Line": str(sharp_line),
                "Retail Line": str(raw_line),
                "True Prob": str(sharp_prob),
                "EV Edge": str(ev_edge),
                "Raw Kelly": str(kelly_fraction),
                "Masked Size": str(masked_stake),
                "Book": mapped_book,
                "Expected Value": str(zinb_mu),
                "ZINB_pi": str(zinb_pi),
                "ZINB_n": str(zinb_n),
                "Anchor Liquidity USD": str(anchor_liquidity_usd) if anchor_liquidity_usd is not None else "",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            r.expire(signal_key, 300)
            logger.info(f"Cached +EV signal in Redis: {signal_key}")
        except Exception as re:
            logger.error(f"Failed to cache +EV signal in Redis: {re}")

        # 3. Save to PostgreSQL
        sharp_odds = self._prob_to_american(sharp_prob)
        stale_odds = target_american
        
        asyncio.create_task(
            asyncio.to_thread(
                self.save_signal_to_db,
                player_name=player_name,
                player_id=player_id,
                sharp_book=sharp_book,
                stale_book=mapped_book,
                sharp_line=float(sharp_line),
                stale_line=float(raw_line),
                sharp_odds=sharp_odds,
                stale_odds=stale_odds,
                expected_value=ev_edge,
                kelly_fraction=kelly_fraction,
                anchor_liquidity_usd=anchor_liquidity_usd
            )
        )

    def save_signal_to_db(self, player_name: str, player_id: str, sharp_book: str, stale_book: str,
                          sharp_line: float, stale_line: float, sharp_odds: int, stale_odds: int,
                          expected_value: float, kelly_fraction: float, anchor_liquidity_usd: float | None = None):
        """Asynchronously inserts the detected +EV signal into PostgreSQL."""
        try:
            conn = psycopg2.connect(self.db_uri)
            cursor = conn.cursor()
            query = """
                INSERT INTO arb_signals (
                    timestamp, player_id, player_name, sharp_book, stale_book,
                    sharp_line, stale_line, sharp_over_odds, stale_under_odds,
                    expected_value, z_score, kelly_fraction, anchor_liquidity_usd
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
            """
            cursor.execute(query, (
                datetime.now(timezone.utc),
                player_id,
                player_name,
                sharp_book,
                stale_book,
                sharp_line,
                stale_line,
                sharp_odds,
                stale_odds,
                expected_value,
                None,  # z_score
                kelly_fraction,
                anchor_liquidity_usd
            ))
            conn.commit()
            cursor.close()
            conn.close()
            logger.info(f"Logged +EV signal to PostgreSQL: {player_name} ({sharp_book} vs {stale_book})")
        except Exception as e:
            logger.error(f"Failed to log signal to PostgreSQL: {e}")

    def _build_anchors(self, kalshi_data: dict, polymarket_data: dict):
        """
        Builds the current cycle's anchors dictionary containing all sharp probabilities.
        Constructs a 3D map: self.anchors[player][stat][line] = {book: {prob, ask_liquidity_usd, bid_liquidity_usd}}
        We combine Pinnacle (cached) with Kalshi (mid-price) and Polymarket (mid-price).
        """
        self.anchors = {}  # Format: self.anchors[player][stat][line] = {book: {prob, ask_liquidity_usd, bid_liquidity_usd}}

        # 1. Load Pinnacle anchors
        for player, stats in self.sharp_cache.items():
            player_norm = clean_player_name(player)
            if not player_norm:
                continue
            if player_norm not in self.anchors:
                self.anchors[player_norm] = {}
            for stat, lines_dict in stats.items():
                stat_norm = stat.lower()
                if stat_norm not in self.anchors[player_norm]:
                    self.anchors[player_norm][stat_norm] = {}
                for line, data in lines_dict.items():
                    prob = data.get('pinnacle_true_prob')
                    if prob is not None:
                        line_float = float(line)
                        if line_float not in self.anchors[player_norm][stat_norm]:
                            self.anchors[player_norm][stat_norm][line_float] = {}
                        self.anchors[player_norm][stat_norm][line_float]['Pinnacle'] = {
                            'prob': prob,
                            'ask_liquidity_usd': None,
                            'bid_liquidity_usd': None
                        }

        # 2. Add Kalshi anchors
        for player, stats in kalshi_data.items():
            player_norm = clean_player_name(player)
            if not player_norm:
                continue
            if player_norm not in self.anchors:
                self.anchors[player_norm] = {}
            for stat, lines_dict in stats.items():
                stat_norm = self._map_stat_type(stat)
                if not stat_norm:
                    continue
                if stat_norm not in self.anchors[player_norm]:
                    self.anchors[player_norm][stat_norm] = {}
                for line, p in lines_dict.items():
                    yes_bid = p.get('yes_bid', 0.0)
                    yes_ask = p.get('yes_ask', 0.0)
                    mid_prob = p.get('prob') or ((yes_bid + yes_ask) / 2.0)
                    if mid_prob > 0.0:
                        line_float = float(line)
                        if line_float not in self.anchors[player_norm][stat_norm]:
                            self.anchors[player_norm][stat_norm][line_float] = {}
                        self.anchors[player_norm][stat_norm][line_float]['Kalshi'] = {
                            'prob': mid_prob,
                            'ask_liquidity_usd': p.get('ask_liquidity_usd', 0.0),
                            'bid_liquidity_usd': p.get('bid_liquidity_usd', 0.0)
                        }

        # 3. Add Polymarket anchors
        for player, stats in polymarket_data.items():
            player_norm = clean_player_name(player)
            if not player_norm:
                continue
            if player_norm not in self.anchors:
                self.anchors[player_norm] = {}
            for stat, lines_dict in stats.items():
                stat_norm = self._map_stat_type(stat)
                if not stat_norm:
                    continue
                if stat_norm not in self.anchors[player_norm]:
                    self.anchors[player_norm][stat_norm] = {}
                for line, p in lines_dict.items():
                    yes_bid = p.get('yes_bid', 0.0)
                    yes_ask = p.get('yes_ask', 0.0)
                    mid_prob = p.get('prob') or ((yes_bid + yes_ask) / 2.0)
                    if mid_prob > 0.0:
                        line_float = float(line)
                        if line_float not in self.anchors[player_norm][stat_norm]:
                            self.anchors[player_norm][stat_norm][line_float] = {}
                        self.anchors[player_norm][stat_norm][line_float]['Polymarket'] = {
                            'prob': mid_prob,
                            'ask_liquidity_usd': p.get('ask_liquidity_usd', 0.0),
                            'bid_liquidity_usd': p.get('bid_liquidity_usd', 0.0)
                        }

    def _flatten_3d_state(self, data: dict) -> dict:
        """
        Flattens a 3D exchange state dictionary player->stat->line->info 
        to a flat dict keyed by player_stat_line.
        """
        flat = {}
        if not data:
            return flat
        for player, stats in data.items():
            for stat, lines in stats.items():
                for line, info in lines.items():
                    prop_id = f"{player}_{stat}_{line}"
                    flat[prop_id] = info
        return flat

    # ══════════════════════════════════════════════════════════
    # MASTER ORCHESTRATION LOOP
    # ══════════════════════════════════════════════════════════

    async def poll_markets(self):
        """
        Master polling cycle:
          1. Load latest sharp lines from cache (Tier 2).
          2. Fetch PrizePicks, Underdog, Kalshi, and Polymarket concurrently.
          3. Run diff engine on all feeds.
        """
        # Step 1: Load sharp cache (non-blocking, reads local file)
        try:
            self.sharp_cache = await self.sharp_provider.get_sharp_lines()
            logger.info(f"Sharp cache loaded: {len(self.sharp_cache)} players")
        except Exception as e:
            logger.warning(f"Sharp cache load failed: {e}")
            self.sharp_cache = {}

        # Step 2: Fetch all books concurrently
        proxy_url = os.getenv("RESIDENTIAL_PROXY_URL")
        proxies = {"http": proxy_url, "https": proxy_url} if proxy_url else None

        async with AsyncSession(impersonate="chrome110", headers=self.headers, proxies=proxies) as session:
            logger.info("Fetching all books concurrently (PrizePicks + Underdog + Kalshi + Polymarket)...")
            
            # Fetch all four sources concurrently
            pp_data, ud_data, kalshi_data, polymarket_data = await asyncio.gather(
                self.fetch_pp(session),
                self.fetch_ud(session),
                self.kalshi_client.fetch_current_lines(session),
                self.polymarket_client.fetch_current_lines(session),
                return_exceptions=True
            )

            # Handle exceptions gracefully
            if isinstance(pp_data, Exception):
                logger.error(f"Error fetching PrizePicks: {pp_data}")
                pp_data = {}
            if isinstance(ud_data, Exception):
                logger.error(f"Error fetching Underdog: {ud_data}")
                ud_data = {}
            if isinstance(kalshi_data, Exception):
                logger.error(f"Error fetching Kalshi: {kalshi_data}")
                kalshi_data = {}
            if isinstance(polymarket_data, Exception):
                logger.error(f"Error fetching Polymarket: {polymarket_data}")
                polymarket_data = {}

            logger.info(
                f"Retrieved PP: {len(pp_data)} | UD: {len(ud_data)} | "
                f"Kalshi: {len(kalshi_data)} | Polymarket: {len(polymarket_data)} props"
            )

            # Build in-memory anchors mapping for the current cycle
            self._build_anchors(kalshi_data, polymarket_data)

            # Step 3: Run diff engine on all feeds
            await self._diff_engine('pp', pp_data)
            await self._diff_engine('ud', ud_data)

            # Flatten 3D exchange states before sending to 1D diff engine
            flat_kalshi = self._flatten_3d_state(kalshi_data)
            flat_polymarket = self._flatten_3d_state(polymarket_data)
            await self._diff_engine('kalshi', flat_kalshi)
            await self._diff_engine('polymarket', flat_polymarket)


async def main():
    poller = UnifiedRestPoller()

    print("\n" + "=" * 60)
    print(" HIGH-FREQUENCY POLLER (HFP) — DUAL-TIER ARCHITECTURE")
    print("=" * 60)
    print(" Tier 1 (Retail): PrizePicks + Underdog Fantasy (30s cycle)")
    print(" Tier 2 (Sharp):  Pinnacle/DK (45-min cache) + Kalshi + Polymarket")
    print("=" * 60 + "\n")

    cycle = 0
    while True:
        cycle += 1
        print(f"\n>>> Cycle {cycle}: Polling markets...")
        await poller.poll_markets()

        print(f"\n{'─'*60}")
        print(f" PrizePicks Props Monitored:  {len(poller.last_state['pp'])}")
        print(f" Underdog Props Monitored:    {len(poller.last_state['ud'])}")
        print(f" Kalshi Props Monitored:      {len(poller.last_state['kalshi'])}")
        print(f" Polymarket Props Monitored:  {len(poller.last_state['polymarket'])}")
        print(f" Sharp Cache Players Loaded:  {len(poller.sharp_cache)}")
        print(f"{'─'*60}")

        print(f"\n>>> Waiting 30 seconds...\n")
        await asyncio.sleep(30)


if __name__ == "__main__":
    asyncio.run(main())
