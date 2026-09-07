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
import random
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from curl_cffi.requests import AsyncSession
import psycopg2
from dotenv import load_dotenv
load_dotenv()

from src.models import ZINBModel, PlayerProjectionModel, ShinDevigger, EVCalculator, BetMaskingEngine
from src.execution import CorrelatedSlipEvaluator
from src.utils import MarketDataLogger
from infrastructure.entity_resolver import EntityResolver, clean_player_name
from infrastructure.redis_client import RedisClient

# Import retail pollers and exchanges
from scrapers.exchanges import UnderdogGhostClient, KalshiClobClient, PolymarketClobClient

# Import sharp line provider
from scrapers.unified_api import SharpLineProvider

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/hfp_poller.log"), # Saves to file
        logging.StreamHandler()                     # Still prints to tmux
    ]
)

import threading
import urllib.request
import json

class DiscordErrorHandler(logging.Handler):
    def __init__(self, webhook_url):
        super().__init__()
        self.webhook_url = webhook_url
        self.error_cache = {}           # Tracks recently emitted errors by fingerprint
        self.cooldown_seconds = 3600    # 1-hour cooldown per unique error signature

    def emit(self, record):
        if record.levelno < logging.ERROR:
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
if discord_webhook:
    handler = DiscordErrorHandler(discord_webhook)
    handler.setLevel(logging.ERROR)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)

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
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
            "Sec-Ch-Ua": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"macOS"',
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-site",
            "Referer": "https://app.prizepicks.com/",
            "Origin": "https://app.prizepicks.com",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
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
        self.last_pp_fetch_time = None
        self.last_ud_fetch_time = None

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
        self.discord_webhook_url = os.getenv("DISCORD_WEBHOOK_URL")

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

        # ── SQLite Market Snapshouter Logger ──────────────────
        self.rate_limited = False
        try:
            self.market_logger = MarketDataLogger()
            logger.info("MarketDataLogger SQLite snapshouter initialized.")
        except Exception as mle:
            logger.error(f"Failed to initialize MarketDataLogger: {mle}")
            self.market_logger = None

        # ── Positional Priors & Position Caching ──────────────
        from src.models import load_all_positional_priors
        self.cached_priors = load_all_positional_priors(self.db_uri)
        self.player_positions = {}
        logger.info("Pre-loaded positional priors for O(1) in-memory lookups.")

        # ── Map Live Market Strings to Specialized XGBoost Models ──
        self.market_model_map = {
            'Points': 'models/global_xpts.json',
            'Rebounds': 'models/global_xreb.json',
            'Assists': 'models/global_xast.json',
            '3-Pointers Made': 'models/global_xfg3m.json',
            '3-PT Made': 'models/global_xfg3m.json',
            '3pt Made': 'models/global_xfg3m.json',
            '3-Pointers': 'models/global_xfg3m.json',
            'Steals': 'models/global_xstl.json',
            'Blocked Shots': 'models/global_xblk.json',
            'Turnovers': 'models/global_xtov.json'
        }
        
        self.loaded_models = {}
        self.models_loaded = False
        self._initialize_xgboost_cluster()

        import redis.asyncio as aioredis
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", 6379))
        redis_password = os.getenv("REDIS_PASSWORD", None)
        self.redis = aioredis.Redis(
            host=redis_host,
            port=redis_port,
            password=redis_password,
            decode_responses=True,
            socket_timeout=5.0
        )
        
        self.locks = {
            'pp': asyncio.Lock(),
            'ud': asyncio.Lock(),
            'kalshi': asyncio.Lock(),
            'polymarket': asyncio.Lock()
        }

        # Initialize an asynchronous alert queue and start the coalescing daemon
        self.alert_queue = asyncio.Queue()
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._coalesced_alert_consumer())
        except RuntimeError:
            pass

        # --- NEW: Initialize the Bayesian Market Blender ---
        from src.models import BayesianMarketBlender
        self.blender = BayesianMarketBlender(confidence_threshold_usd=1500.0)

    def _initialize_xgboost_cluster(self):
        """Pre-loads all target models into memory at daemon startup."""
        import xgboost as xgb
        import os
        
        loaded_count = 0
        for market, path in self.market_model_map.items():
            if os.path.exists(path):
                try:
                    bst = xgb.XGBRegressor()
                    bst.load_model(path)
                    self.loaded_models[market] = bst
                    loaded_count += 1
                    logger.info(f"Successfully loaded model cluster for {market} from {path}.")
                except Exception as e:
                    logger.error(f"Failed to load model for {market} from {path}: {e}")
            else:
                logger.warning(f"Model file missing at {path}. Market execution disabled.")
                
        if loaded_count > 0:
            self.models_loaded = True
            logger.info(f"Successfully loaded model cluster with {loaded_count} models.")
        else:
            self.models_loaded = False
            logger.error("No models loaded in the cluster.")

    def _calculate_dynamic_interval(self) -> float:
        """
        Calculates dynamic polling interval based on time remaining to the next scheduled game.
        - < 30 minutes: 30 seconds
        - 30 minutes to 1 hour: 1 minute (60 seconds)
        - 1 to 5 hours: 2 minutes (120 seconds)
        - 5 to 12 hours: 5 minutes (300 seconds)
        - > 12 hours or no future games: 10 minutes (600 seconds)
        """
        now = datetime.now(timezone.utc)
        min_seconds = None
        
        # Check start times across all loaded PrizePicks projections in last state
        for leg in self.last_state.get('pp', {}).values():
            start_time_str = leg.get('start_time')
            if start_time_str:
                try:
                    # Handles ISO format like Z or +00:00
                    start_dt = datetime.fromisoformat(start_time_str.replace("Z", "+00:00"))
                    diff_seconds = (start_dt - now).total_seconds()
                    # Only consider future games
                    if diff_seconds > 0:
                        if min_seconds is None or diff_seconds < min_seconds:
                            min_seconds = diff_seconds
                except Exception:
                    pass
                    
        if min_seconds is None:
            return 600.0 + random.uniform(-45.0, 45.0)  # Default to 10 minutes if no future games are found
            
        min_minutes = min_seconds / 60.0
        
        if min_minutes < 30.0:
            return 30.0    # < 30 minutes: 30 seconds
        elif min_minutes < 60.0:
            return 60.0    # 30m to 1h: 1 minute
        elif min_minutes < 300.0:
            return 120.0   # 1h to 5h: 2 minutes
        elif min_minutes < 720.0:
            return 300.0   # 5h to 12h: 5 minutes
        else:
            return 600.0 + random.uniform(-45.0, 45.0)   # > 12h: 10 minutes with jitter

    # ══════════════════════════════════════════════════════════
    # UTILITY METHODS
    # ══════════════════════════════════════════════════════════

    def _map_stat_type(self, prop_name: str) -> str:
        """Maps API prop category string to db column name."""
        if not prop_name:
            return None
        prop_lower = prop_name.lower().strip()
        if '3-pt' in prop_lower or '3pt' in prop_lower or '3-pointer' in prop_lower:
            return 'three_pointers_made'
        elif 'point' in prop_lower:
            return 'points'
        elif 'rebound' in prop_lower:
            return 'rebounds'
        elif 'assist' in prop_lower:
            return 'assists'
        elif 'turnover' in prop_lower:
            return 'turnovers'
        elif 'steal' in prop_lower:
            return 'steals'
        elif 'block' in prop_lower:
            return 'blocks'
        return None

    # ══════════════════════════════════════════════════════════
    # DATABASE & MODEL PIPELINE (Preserved from Sprint 10)
    # ══════════════════════════════════════════════════════════

    def get_player_position(self, player_id: str) -> str:
        """Looks up the position of the canonical master_player_id from the database with O(1) caching."""
        if not player_id:
            return "SG"
        if hasattr(self, 'player_positions') and player_id in self.player_positions:
            return self.player_positions[player_id]
        try:
            conn = psycopg2.connect(self.db_uri)
            cursor = conn.cursor()
            cursor.execute("SELECT position FROM players WHERE master_player_id = %s", (player_id,))
            row = cursor.fetchone()
            conn.close()
            if row and row[0]:
                pos = row[0].upper().strip()
                if hasattr(self, 'player_positions'):
                    self.player_positions[player_id] = pos
                return pos
        except Exception as e:
            logger.error(f"Error fetching position for {player_id}: {e}")
        if hasattr(self, 'player_positions'):
            self.player_positions[player_id] = "SG"
        return "SG"

    async def get_player_history_async(self, player_id: str) -> pd.DataFrame:
        """Fetches historical box scores from the Redis In-Memory store in O(1) time."""
        try:
            raw_data = await self.redis.json().get(f"player:history:{player_id}")
            if raw_data:
                df = pd.DataFrame(raw_data)
                
                # Coerce stats columns to numeric
                stat_cols = ['minutes_played', 'points', 'rebounds', 'assists', 'turnovers', 'steals', 'blocks', 'three_pointers_made', 'usage_rate']
                for col in stat_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
                return df
            else:
                logger.warning(f"Cache miss for {player_id} in Redis. Ensure sync_feature_store.py has been run.")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Redis feature store fetch error: {e}")
            return pd.DataFrame()

    def project_player_baseline(self, history_df: pd.DataFrame, stat_col: str, player_id: str = None) -> float:
        """
        Engineers shrunken rolling stat features and contextual features,
        then queries the preloaded model cluster to return the predicted target mean (mu) directly.
        """
        if len(history_df) < 5:
            # Fallback to a simple historical mean or a baseline average
            if not history_df.empty:
                return max(0.001, float(history_df[stat_col].mean()))
            return 15.0

        try:
            from src.features import ContextualFeatureEngine
            fe = ContextualFeatureEngine(db_uri=self.db_uri)
            
            # Enrich history with contextual features first
            df = fe.build_player_feature_matrix(player_id, history_df)
            
            df['usage_rate'] = (
                (df['points'] + 1.5 * df['assists'] + df['turnovers'])
                / (df['minutes_played'] * 2.0 + 1e-5)
            )
            df['usage_rate'] = df['usage_rate'].clip(0.05, 0.45)

            df['roll_min'] = df['minutes_played'].shift(1).ewm(alpha=0.3, adjust=False).mean()
            df['roll_stat'] = df[stat_col].shift(1).ewm(alpha=0.3, adjust=False).mean()
            df['roll_usg'] = df['usage_rate'].shift(1).ewm(alpha=0.3, adjust=False).mean()

            # Apply Bayesian shrinkage to roll_stat based on cumulative minutes and positional prior
            from src.models import get_positional_prior, apply_bayesian_shrinkage
            position = self.get_player_position(player_id) if player_id else "SG"
            prior_per_36 = get_positional_prior(stat_col, position, cached_priors=getattr(self, 'cached_priors', None))

            # Cumulative minutes played prior to the game (shift 1 so current game's minutes are excluded)
            df['cum_min'] = df['minutes_played'].shift(1).cumsum().fillna(0.0)

            # Convert rolling stat per game to rolling stat per 36 mins
            df['roll_stat_per_36'] = (df['roll_stat'] / (df['roll_min'] + 1e-5)) * 36

            # Apply shrinkage to obtain Bayesian rolling stat per 36
            df['bayesian_roll_stat_per_36'] = df.apply(
                lambda row: apply_bayesian_shrinkage(row['roll_stat_per_36'], row['cum_min'], prior_per_36, K=500.0),
                axis=1
            )
            # Scale back to expected per-game value
            df['bayesian_roll_stat'] = (df['bayesian_roll_stat_per_36'] / 36.0) * df['roll_min']

            # Overwrite roll_stat with shrunken bayesian_roll_stat
            df['roll_stat'] = df['bayesian_roll_stat']

            last_3 = df.tail(3)
            if len(last_3) < 3:
                if not history_df.empty:
                    return max(0.001, float(history_df[stat_col].mean()))
                return 15.0

            # Apply shrinkage to prediction features as well
            total_minutes = df['minutes_played'].sum()
            pred_raw_per_36 = (last_3[stat_col].mean() / (last_3['minutes_played'].mean() + 1e-5)) * 36
            pred_bayesian_per_36 = apply_bayesian_shrinkage(pred_raw_per_36, total_minutes, prior_per_36, K=500.0)
            pred_bayesian_stat = (pred_bayesian_per_36 / 36.0) * last_3['minutes_played'].mean()

            # Active/upcoming game contextual features
            active_polled_names = []
            if hasattr(self, 'last_state'):
                for b in self.last_state:
                    if self.last_state[b]:
                        active_polled_names.extend([p.get('player') for p in self.last_state[b].values() if p.get('player')])
            
            active_polled_ids = []
            from unittest.mock import Mock
            if not isinstance(getattr(self.resolver, 'resolve_player', None), Mock):
                for name in set(active_polled_names):
                    resolved_id = self.resolver.resolve_player("PrizePicks", name)
                    if resolved_id:
                        active_polled_ids.append(resolved_id)

            active_feats = fe.build_active_features(player_id, active_polled_player_ids=active_polled_ids)

            pred_features = np.array([[
                last_3['minutes_played'].mean(),
                pred_bayesian_stat,
                last_3['usage_rate'].mean(),
                active_feats["days_rest"],
                active_feats["is_back_to_back"],
                active_feats["opp_def_rating"],
                active_feats["opp_pace_factor"],
                active_feats.get("vacated_usage", active_feats.get("teammate_sidelined", 0.0))
            ]])

            # Map database stat column to model market key
            stat_to_market = {
                'points': 'Points',
                'rebounds': 'Rebounds',
                'assists': 'Assists',
                'three_pointers_made': '3-Pointers Made',
                'steals': 'Steals',
                'blocks': 'Blocked Shots',
                'turnovers': 'Turnovers'
            }
            market_key = stat_to_market.get(stat_col.lower().strip())
            model = self.loaded_models.get(market_key) if market_key else None

            if model is None:
                logger.warning(f"Unmapped or missing model for property target: '{stat_col}' (market key: '{market_key}'). Using historical average fallback.")
                return max(0.001, float(last_3[stat_col].mean()))

            predicted_mu = float(model.predict(pred_features)[0])
            return max(0.001, predicted_mu)
        except Exception as e:
            logger.error(f"XGBoost target projection error: {e}")
            if not history_df.empty:
                return max(0.001, float(history_df[stat_col].mean()))
            return 15.0

    # ══════════════════════════════════════════════════════════
    # RETAIL BOOK FETCHERS
    # ══════════════════════════════════════════════════════════

    async def fetch_pp(self, session: AsyncSession) -> dict | None:
        """Fetch PrizePicks unauthenticated projections."""
        try:
            params = {"league_id": "7", "per_page": "250"}
            resp = await session.get(self.pp_url, params=params, headers=self.headers)
            if resp.status_code == 200:
                raw = resp.json()
                return self._parse_pp(raw)
            else:
                logger.warning(f"PP API returned {resp.status_code}")
                if resp.status_code in [403, 429]:
                    self.rate_limited = True
                return None
        except Exception as e:
            logger.error(f"PP fetch error: {e}")
            return None

    async def fetch_ud(self, session: AsyncSession) -> dict | None:
        """Fetch Underdog Fantasy lines via the ghost client."""
        try:
            return await self.ud_client.fetch_current_lines(session)
        except Exception as e:
            logger.error(f"Underdog fetch error: {e}")
            return None

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

            # 1. Native API combo flag check
            if attrs.get("player_combo", False):
                continue

            # Strict League Filter: Only allow NBA
            league_rel = rels.get("league", {}).get("data", {})
            league_attrs = included_map.get((league_rel.get("type"), league_rel.get("id")), {})
            league_name = str(league_attrs.get("name", "")).strip().upper()
            if league_name != "NBA":
                continue

            player_rel = rels.get("new_player", {}).get("data", {})
            player_attrs = included_map.get((player_rel.get("type"), player_rel.get("id")), {})
            player_name = player_attrs.get("display_name") or player_attrs.get("name", "Unknown")

            # 2. Hard string fallback check for combo props
            if "+" in player_name:
                continue

            if proj_id:
                parsed[proj_id] = {
                    'player': clean_player_name(player_name),
                    'prop': attrs.get("stat_display_name", ""),
                    'line': attrs.get("line_score"),
                    'odds_type': attrs.get("odds_type", "standard"),
                    'flash_sale': attrs.get("flash_sale", False),
                    'is_promo': attrs.get("is_promo", False),
                    'promo': attrs.get("promo", False),
                    'start_time': attrs.get("start_time")
                }
        return parsed

    def apply_pace_elasticity(self, base_mu: float, base_pi: float, opp_pace: float, prop_type: str) -> tuple[float, float]:
        """
        Non-linearly scales expected distribution parameters based on opponent game pace.
        """
        LEAGUE_AVERAGE_PACE = 99.0
        
        # Safe fallback if opponent pace is missing or corrupted
        valid_pace = max(opp_pace if opp_pace and not pd.isna(opp_pace) else LEAGUE_AVERAGE_PACE, 85.0)
        pace_ratio = valid_pace / LEAGUE_AVERAGE_PACE
        
        # Elasticity exponents: Points/Rebounds scale less directly than pure usage stats like Assists
        gamma = 1.0 if prop_type == 'Assists' else 0.85
        
        # 1. Scale Mu Non-Linearly
        adj_mu = base_mu * (pace_ratio ** gamma)
        
        # 2. Inverse Scale Zero-Inflation (Faster pace = fewer true zero outputs)
        adj_pi = base_pi * (1.0 / pace_ratio)
        
        # Boundary clamps to ensure mathematical stability
        adj_pi = max(0.001, min(adj_pi, 0.499))
        
        return adj_mu, adj_pi

    def calculate_theoretical_prob(self, player_name: str, stat_col: str, line: float, master_id: str = None, history_df: pd.DataFrame = None) -> tuple[float, ZINBModel]:
        """
        Fits the Zero-Inflated Negative Binomial distribution to the pre-fetched historical baseline.
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
            # Ensure history_df was passed in successfully
            if history_df is None:
                history_df = pd.DataFrame()
                
            zinb = ZINBModel()

            if not history_df.empty:
                predicted_mu = self.project_player_baseline(history_df, stat_col, master_id)

                stat_series = history_df[stat_col]
                zinb.fit(stat_series)

                # Direct prediction directly maps to the expected mean
                zinb.mu = predicted_mu
                logger.info(f"[{player_name}] ZINB mu set to direct XGBoost target prediction: {zinb.mu:.4f}")
            else:
                logger.warning(f"No history for '{player_name}' (ID: {master_id}). Using fallback.")
                zinb.mu = line
                zinb.pi = 0.03
                zinb.n = 15.0
                predicted_mu = line

            self.zinb_cache[cache_key] = (zinb, predicted_mu)

        # Condition ZINB parameters using active opponent pace factor
        try:
            from src.features import ContextualFeatureEngine
            fe = ContextualFeatureEngine(db_uri=self.db_uri)
            active_polled_names = []
            if hasattr(self, 'last_state'):
                for b in self.last_state:
                    if self.last_state[b]:
                        active_polled_names.extend([p.get('player') for p in self.last_state[b].values() if p.get('player')])
            active_polled_ids = []
            from unittest.mock import Mock
            if not isinstance(getattr(self.resolver, 'resolve_player', None), Mock):
                for name in set(active_polled_names):
                    resolved_id = self.resolver.resolve_player("PrizePicks", name)
                    if resolved_id:
                        active_polled_ids.append(resolved_id)
            active_feats = fe.build_active_features(master_id, active_polled_player_ids=active_polled_ids)
            opp_pace = active_feats.get("opp_pace_factor", 99.0)
        except Exception as e:
            logger.error(f"Failed to fetch active pace factor for ZINB conditioning: {e}")
            opp_pace = 99.0

        # Apply pace elasticity adjustments to zinb.mu and zinb.pi
        prop_type = 'Assists' if stat_col.lower() == 'assists' else 'Other'
        zinb.mu, zinb.pi = self.apply_pace_elasticity(zinb.mu, zinb.pi, opp_pace, prop_type)

        # Call condition_parameters with pace_factor=1.0 since pace scaling is handled by apply_pace_elasticity
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
        async with self.locks[book]:
            redis_key = f"board:state:{book}"
            try:
                redis_state = await self.redis.json().get(redis_key)
                if redis_state is None:
                    redis_state = self.last_state[book]
            except Exception as e:
                logger.error(f"Error fetching state from Redis for {book}: {e}")
                redis_state = self.last_state[book]

            old_state = redis_state
            ticks = []

            if not old_state:
                self.last_state[book] = new_state
                try:
                    await self.redis.json().set(redis_key, "$", new_state)
                except Exception as e:
                    logger.error(f"Error setting initial state in Redis for {book}: {e}")
                logger.info(f"Seeded initial state for {book} ({len(new_state)} props)")
                return ticks

            for prop_id, new_data in new_state.items():
                # Explicitly clear loop context variables to prevent async reuse/leakage
                current_player_raw = None
                master_id = None
                theoretical_prob = None
                zinb = None

                if prop_id in old_state:
                    old_data = old_state[prop_id]

                    # Check for line movement or price change
                    if new_data != old_data:
                        # 1. Force explicit isolation of the player string
                        current_player_raw = str(new_data.get('player', '')).strip()

                        # 2. Strict validation before calling the resolver
                        if not current_player_raw or current_player_raw == 'None' or current_player_raw == '':
                            continue

                        tick = {
                            'timestamp': datetime.now(timezone.utc).isoformat(),
                            'book': book,
                            'player': current_player_raw,
                            'prop': new_data['prop'],
                            'old': old_data,
                            'new': new_data
                        }
                        ticks.append(tick)

                        print(f"\n{'='*60}")
                        print(f" 🚨 SYNTHETIC TICK | {book.upper()} | {current_player_raw} - {new_data['prop']}")
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
                        player_clean = clean_player_name(current_player_raw)
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
                        master_id = self.resolver.resolve_player(mapped_book, current_player_raw)

                        # --- NEW: Fetch from Redis in micro-seconds ---
                        history_df = await self.get_player_history_async(master_id)

                        # Pass the DataFrame directly into the thread to avoid blocking I/O
                        theoretical_prob, zinb = await asyncio.to_thread(
                            self.calculate_theoretical_prob,
                            current_player_raw,
                            stat_col,
                            raw_line,
                            master_id,
                            history_df
                        )

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
                                    f"Thin prediction market line rejected: {current_player_raw} on {abook} "
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
                                
                                # --- NEW: Apply Bayesian Shrinkage for Pinnacle lines ---
                                estimated_limit = self.blender.get_heuristic_limit(new_data['prop'])
                                sharp_prob = self.blender.calculate_blended_probability(
                                    sharp_prior_prob=sharp_prob,
                                    internal_model_prob=theoretical_prob,
                                    market_limit_usd=estimated_limit
                                )
                            else:
                                ask_liq = chosen_anchor_data.get('ask_liquidity_usd') or 0.0
                                bid_liq = chosen_anchor_data.get('bid_liquidity_usd') or 0.0
                                anchor_liquidity_usd = min(ask_liq, bid_liq)
                                
                                # --- NEW: Apply Bayesian Shrinkage for prediction market anchors ---
                                sharp_prob = self.blender.calculate_blended_probability(
                                    sharp_prior_prob=sharp_prob,
                                    internal_model_prob=theoretical_prob,
                                    market_limit_usd=anchor_liquidity_usd
                                )

                        # ── Evaluation 1: YES / Over Contract ─────────────────
                        if target_yes_prob > 0.0:
                            target_ask_liq = new_data.get('ask_liquidity_usd') or 0.0
                            
                            # Calculate both edge metrics
                            absolute_edge = sharp_prob - target_yes_prob
                            relative_edge = (sharp_prob / target_yes_prob) - 1.0
                            
                            # DYNAMIC TAIL PROTECTION
                            # If the contract is cheaper than 15 cents, rely strictly on Absolute Edge >= 5.0%
                            if target_yes_prob < 0.15:
                                is_valid_edge = absolute_edge >= 0.05
                            else:
                                is_valid_edge = relative_edge >= (self.min_ev_threshold / 100.0)
                            
                            if is_valid_edge:
                                # Net odds received
                                b = (1.0 / target_yes_prob) - 1.0
                                kelly_yes = (sharp_prob - (1.0 - sharp_prob) / b) * 0.25 if b > 0 else 0.0
                                kelly_yes = min(max(kelly_yes, 0.0), 0.05)
                                
                                # Calculate raw recommended stake
                                recommended_stake = kelly_yes * 50000.0
                                
                                # --- NEW: Tail-Risk Volatility Scaler ---
                                # If the true probability of the event is low, aggressively penalize the stake
                                if sharp_prob < 0.30:
                                    # Scale the stake down linearly
                                    volatility_penalty = sharp_prob / 0.30
                                    recommended_stake = recommended_stake * volatility_penalty
                                    kelly_yes = kelly_yes * volatility_penalty
                                    logger.info(f"Applied tail-risk penalty of {volatility_penalty:.2f}x to YES stake for {current_player_raw}. New stake: ${recommended_stake:.2f}")
                                
                                # DYNAMIC LIQUIDITY SCALING: Cap stake at available liquidity
                                if book in ('kalshi', 'polymarket'):
                                    executable_stake = min(recommended_stake, target_ask_liq * 0.95)  # Leave 5% buffer
                                    if executable_stake < 5.0:  # Standard prediction market minimum
                                        logger.info(f"Skipping YES on {book}: Scaled stake (${executable_stake:.2f}) is below $5.00 minimum.")
                                        continue
                                    # Override kelly fraction based on scaled stake
                                    kelly_yes = executable_stake / 50000.0
                                    
                                await self._fire_alert_and_save(
                                    book=book,
                                    player_name=current_player_raw,
                                    player_id=master_id,
                                    prop_name=new_data['prop'],
                                    side="Over",
                                    raw_line=raw_line,
                                    sharp_book=chosen_anchor_book,
                                    sharp_line=sharp_line,
                                    sharp_prob=sharp_prob,
                                    target_cost=target_yes_prob,
                                    target_american=target_yes_american,
                                    ev_edge=absolute_edge if target_yes_prob < 0.15 else relative_edge,
                                    kelly_fraction=kelly_yes,
                                    zinb_mu=zinb.mu,
                                    zinb_pi=zinb.pi,
                                    zinb_n=zinb.n,
                                    anchor_liquidity_usd=anchor_liquidity_usd
                                )
                                
                        # ── Evaluation 2: NO / Under Contract ─────────────────
                        if target_no_prob > 0.0:
                            target_bid_liq = new_data.get('bid_liquidity_usd') or 0.0
                            
                            # Calculate both edge metrics
                            absolute_edge = (1.0 - sharp_prob) - target_no_prob
                            relative_edge = ((1.0 - sharp_prob) / target_no_prob) - 1.0
                            
                            # DYNAMIC TAIL PROTECTION
                            # If the contract is cheaper than 15 cents, rely strictly on Absolute Edge >= 5.0%
                            if target_no_prob < 0.15:
                                is_valid_edge = absolute_edge >= 0.05
                            else:
                                is_valid_edge = relative_edge >= (self.min_ev_threshold / 100.0)
                            
                            if is_valid_edge:
                                # Net odds received
                                b = (1.0 / target_no_prob) - 1.0
                                kelly_no = ((1.0 - sharp_prob) - sharp_prob / b) * 0.25 if b > 0 else 0.0
                                kelly_no = min(max(kelly_no, 0.0), 0.05)
                                
                                # Calculate raw recommended stake
                                recommended_stake = kelly_no * 50000.0
                                
                                # --- NEW: Tail-Risk Volatility Scaler ---
                                # If the true probability of the event is low, aggressively penalize the stake
                                event_prob = 1.0 - sharp_prob
                                if event_prob < 0.30:
                                    # Scale the stake down linearly
                                    volatility_penalty = event_prob / 0.30
                                    recommended_stake = recommended_stake * volatility_penalty
                                    kelly_no = kelly_no * volatility_penalty
                                    logger.info(f"Applied tail-risk penalty of {volatility_penalty:.2f}x to NO stake for {current_player_raw}. New stake: ${recommended_stake:.2f}")
                                
                                # DYNAMIC LIQUIDITY SCALING: Cap stake at available liquidity
                                if book in ('kalshi', 'polymarket'):
                                    executable_stake = min(recommended_stake, target_bid_liq * 0.95)  # Leave 5% buffer
                                    if executable_stake < 5.0:  # Standard prediction market minimum
                                        logger.info(f"Skipping NO on {book}: Scaled stake (${executable_stake:.2f}) is below $5.00 minimum.")
                                        continue
                                    # Override kelly fraction based on scaled stake
                                    kelly_no = executable_stake / 50000.0
                                    
                                await self._fire_alert_and_save(
                                    book=book,
                                    player_name=current_player_raw,
                                    player_id=master_id,
                                    prop_name=new_data['prop'],
                                    side="Under",
                                    raw_line=raw_line,
                                    sharp_book=chosen_anchor_book,
                                    sharp_line=sharp_line,
                                    sharp_prob=1.0 - sharp_prob,
                                    target_cost=target_no_prob,
                                    target_american=target_no_american,
                                    ev_edge=absolute_edge if target_no_prob < 0.15 else relative_edge,
                                    kelly_fraction=kelly_no,
                                    zinb_mu=zinb.mu,
                                    zinb_pi=zinb.pi,
                                    zinb_n=zinb.n,
                                    anchor_liquidity_usd=anchor_liquidity_usd
                                )

            self.last_state[book] = new_state
            try:
                await self.redis.json().set(redis_key, "$", new_state)
            except Exception as e:
                logger.error(f"Error saving state to Redis for {book}: {e}")
            return ticks

    def _prob_to_american(self, prob: float) -> int:
        """
        Converts implied probability to bounded American Odds.
        Clamps to MAX_SQL_INT to prevent PostgreSQL integer out of range errors.
        """
        MAX_SQL_INT = 2000000000
        
        # Bound check for extreme edge cases (e.g., 0% or 100% lines on prediction books)
        prob = max(0.00001, min(prob, 0.99999))
        
        if prob >= 0.5:
            odds = -100 * (prob / (1.0 - prob))
        else:
            odds = 100 * ((1.0 - prob) / prob)
            
        # Hard clamp to prevent 32-bit integer overflows on DB writes
        return max(-MAX_SQL_INT, min(int(round(odds)), MAX_SQL_INT))

    async def _fire_alert_and_save(self, book, player_name, player_id, prop_name, side, raw_line,
                                   sharp_book, sharp_line, sharp_prob, target_cost, target_american,
                                   ev_edge, kelly_fraction, zinb_mu, zinb_pi, zinb_n, anchor_liquidity_usd=None):
        """Dispatches alerts, caches signals in Redis, and stores in Postgres database."""
        mapped_book = "PrizePicks" if book == 'pp' else "Underdog" if book == 'ud' else "Kalshi" if book == 'kalshi' else "Polymarket"
        
        # --- FIX: Standardize the prop name for the cache key ---
        standardized_prop = self._map_stat_type(prop_name) or prop_name
        
        # State-Aware Smart Cache check
        import time
        current_time = time.time()
        alert_key = f"{player_name}_{standardized_prop}_{side}_{mapped_book}"
        last_state = getattr(self, 'smart_alert_cache', {}).get(alert_key)
        
        if last_state:
            time_since_last = current_time - last_state["timestamp"]
            edge_diff = abs(ev_edge - last_state["edge"])
            line_changed = raw_line != last_state["line"]
            
            # If within a 2-hour cooldown window:
            if time_since_last < 7200:
                # ONLY break cooldown if the line physically moved OR the edge shifted by > 5.0% (filters CLOB micro-volatility)
                if not line_changed and edge_diff < 0.05:
                    return
                # Otherwise, it's a meaningful update (line moved or edge expanded/shrunk). Proceed to alert.
        
        # Initialize or update the smart cache with the new state
        if not hasattr(self, 'smart_alert_cache'):
            self.smart_alert_cache = {}
            
        self.smart_alert_cache[alert_key] = {
            "timestamp": current_time,
            "edge": ev_edge,
            "line": raw_line
        }

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

        # Compile embed metadata payload and queue it for the coalescing background daemon
        alert_payload = {
            "player": player_name,
            "prop": prop_name,
            "side": side,
            "line": raw_line,
            "edge": ev_edge,
            "book": mapped_book,
            "stake": float(masked_stake),
            "sharp_book": sharp_book,
            "sharp_line": sharp_line,
            "anchor_liquidity_usd": anchor_liquidity_usd,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Non-blocking push to the memory queue
        self.alert_queue.put_nowait(alert_payload)
        logger.info(f"Buffered +EV single-leg alert for {player_name} to internal queue.")


        # 2. Cache in Redis
        try:
            signal_key = f"signal:{book}:{player_name}:{prop_name}:{side}"
            await self.redis.hset(signal_key, mapping={
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
            await self.redis.expire(signal_key, 300)
            logger.info(f"Cached +EV signal in Redis: {signal_key}")
        except Exception as re:
            logger.error(f"Failed to cache +EV signal in Redis: {re}")

        # 3. Buffer to Redis queue for asynchronous batch-insert into PostgreSQL
        sharp_odds = self._prob_to_american(sharp_prob)
        stale_odds = target_american
        
        signal_data = {
            "player_id": player_id,
            "player_name": player_name,
            "sharp_book": sharp_book,
            "stale_book": mapped_book,
            "sharp_line": float(sharp_line),
            "stale_line": float(raw_line),
            "sharp_over_odds": sharp_odds,
            "stale_under_odds": stale_odds,
            "expected_value": ev_edge,
            "kelly_fraction": kelly_fraction,
            "anchor_liquidity_usd": anchor_liquidity_usd
        }
        
        try:
            await self.redis.rpush("queue:ev_signals_buffer", json.dumps(signal_data))
            logger.info(f"Buffered +EV signal to Redis: {player_name} ({sharp_book} vs {mapped_book})")
        except Exception as e:
            logger.error(f"Failed to buffer +EV signal: {e}")

        # 4. Walk Forward Telemetry Log & Slippage Check
        try:
            from src.utils import WalkForwardLogger
            # Try to lookup start_time from the cached poller state
            game_time = None
            for entry in self.last_state.get(book, {}).values():
                if clean_player_name(entry.get('player', '')) == clean_player_name(player_name) and entry.get('prop') == prop_name:
                    game_time = entry.get('start_time')
                    break
                    
            timestamp_str = datetime.now(timezone.utc).isoformat()
            WalkForwardLogger.log_signal(
                player=player_name,
                stat=prop_name,
                line=raw_line,
                zinb_prob=sharp_prob,
                edge=ev_edge,
                timestamp=timestamp_str,
                game_time=game_time
            )
            
            # Spawn non-blocking background check
            asyncio.create_task(self._check_line_slippage(
                book=book,
                player_name=player_name,
                prop_name=prop_name,
                original_line=raw_line,
                timestamp_str=timestamp_str
            ))
        except Exception as te:
            logger.error(f"Failed to execute walk forward logging: {te}")

    def _dispatch_webhook_sync(self, url, payload):
        """
        Synchronously posts a JSON payload to a Discord webhook URL with retries 
        and exponential/dynamic backoff when rate-limited (HTTP 429).
        """
        if not url:
            return 0
            
        import urllib.request
        import urllib.error
        import time
        import json
        
        for attempt in range(5):
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode('utf-8'),
                headers={
                    'Content-Type': 'application/json',
                    'User-Agent': 'DiscordNotifier/1.0'
                },
                method='POST'
            )
            try:
                with urllib.request.urlopen(req, timeout=10) as response:
                    if response.status in (200, 204):
                        return response.status
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    retry_after = 1.0
                    try:
                        body = e.read().decode('utf-8')
                        res_json = json.loads(body)
                        retry_after = float(res_json.get("retry_after", 1.0))
                    except Exception:
                        pass
                    logger.warning(f"Discord rate limit hit in coalesced alert dispatch. Retrying in {retry_after:.2f} seconds...")
                    time.sleep(retry_after)
                    continue
                else:
                    logger.error(f"Failed to post to Discord (HTTP {e.code})")
                    break
            except Exception as e:
                logger.error(f"Error executing webhook URL request: {e}")
                break
        return 0

    async def _coalesced_alert_consumer(self):
        """
        Monitors the internal alert queue, coalescing up to 10 distinct 
        signals into a single multi-embed Discord payload every 1.5 seconds.
        """
        logger.info("Outbound alert coalescing daemon active.")
        while True:
            try:
                if self.alert_queue.empty():
                    await asyncio.sleep(0.5)
                    continue
                    
                payloads_to_batch = []
                # Drain the queue up to Discord's maximum limit of 10 embeds per message
                while not self.alert_queue.empty() and len(payloads_to_batch) < 10:
                    payloads_to_batch.append(await self.alert_queue.get())
                    self.alert_queue.task_done()
                    
                if not payloads_to_batch:
                    continue
                    
                # Construct the consolidated multi-embed payload
                embeds = []
                for payload in payloads_to_batch:
                    color = 65280 if payload["edge"] >= 0.05 else 16753920
                    fields = [
                        {"name": "Model Edge", "value": f"`{payload['edge']:.2%}`", "inline": True},
                        {"name": "Recommended Stake", "value": f"`${float(payload['stake']):.2f}`", "inline": True},
                        {"name": "Sharp Book (Anchor)", "value": f"`{payload['sharp_book']}` (Line: `{payload['sharp_line']}`)", "inline": True},
                        {"name": "Anchor Liquidity", "value": f"`${float(payload['anchor_liquidity_usd']):,.2f}`" if payload['anchor_liquidity_usd'] is not None else "`N/A`", "inline": True}
                    ]
                    embeds.append({
                        "title": f"🚨 +EV Edge Detected | {payload['book'].upper()}",
                        "description": f"**Player**: `{payload['player']}`\n**Prop**: `{payload['prop']}`\n**Selection**: `{payload['side']} {payload['line']}`",
                        "color": color,
                        "fields": fields,
                        "timestamp": payload["timestamp"]
                    })
                    
                combined_payload = {"embeds": embeds}
                
                # Dispatch the single consolidated request via the threadpool executor
                asyncio.create_task(asyncio.to_thread(
                    self._dispatch_webhook_sync, self.discord_webhook_url, combined_payload
                ))
                
                # Enforce a strict 1.5-second pacing interval to safely clear Discord's 5 req/sec limit
                await asyncio.sleep(1.5)
            except Exception as e:
                logger.error(f"Error in coalesced alert consumer: {e}")
                await asyncio.sleep(1.0)

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
        Includes defensive type-checking to survive malformed API payloads.
        """
        flat = {}
        if not data or not isinstance(data, dict):
            return flat
            
        for player, stats in data.items():
            if not isinstance(stats, dict):
                continue
            for stat, lines in stats.items():
                if not isinstance(lines, dict):
                    continue
                for line, info in lines.items():
                    # Only process if 'info' is actually a dictionary of odds
                    if not isinstance(info, dict):
                        continue
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
        fingerprint = random.choice(["chrome110", "chrome116", "chrome120", "edge101", "safari15_5"])

        async with AsyncSession(impersonate=fingerprint, proxies=proxies) as session:
            logger.info("Fetching all books concurrently (PrizePicks + Underdog + Kalshi + Polymarket)...")
            
            now = datetime.now(timezone.utc)
            interval_seconds = self._calculate_dynamic_interval()
            interval_minutes = interval_seconds / 60.0
            
            # Throttled fetches for PP and Underdog (dynamic interval check)
            pp_coro_or_val = None
            if self.last_pp_fetch_time is None or (now - self.last_pp_fetch_time).total_seconds() >= interval_seconds:
                logger.info(f"Scheduling PrizePicks API network fetch (dynamic interval of {interval_minutes:.1f} minutes elapsed)...")
                pp_coro_or_val = self.fetch_pp(session)
            else:
                logger.info(f"Using cached PrizePicks state to respect {interval_minutes:.1f}-minute dynamic IP protection interval.")
                
            ud_coro_or_val = None
            if self.last_ud_fetch_time is None or (now - self.last_ud_fetch_time).total_seconds() >= interval_seconds:
                logger.info(f"Scheduling Underdog API network fetch (dynamic interval of {interval_minutes:.1f} minutes elapsed)...")
                ud_coro_or_val = self.fetch_ud(session)
            else:
                logger.info(f"Using cached Underdog state to respect {interval_minutes:.1f}-minute dynamic IP protection interval.")

            # Prepare tasks or pre-fetched values
            tasks = []
            task_indices = {}
            
            # Kalshi and Polymarket are fetched every cycle
            tasks.append(self.kalshi_client.fetch_current_lines(session))
            task_indices['kalshi'] = len(tasks) - 1
            
            tasks.append(self.polymarket_client.fetch_current_lines(session))
            task_indices['polymarket'] = len(tasks) - 1
            
            if pp_coro_or_val is not None:
                tasks.append(pp_coro_or_val)
                task_indices['pp'] = len(tasks) - 1
            
            if ud_coro_or_val is not None:
                tasks.append(ud_coro_or_val)
                task_indices['ud'] = len(tasks) - 1
                
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Map results back
            kalshi_data = results[task_indices['kalshi']]
            polymarket_data = results[task_indices['polymarket']]
            
            if pp_coro_or_val is not None:
                pp_data = results[task_indices['pp']]
            else:
                pp_data = self.last_state.get('pp', {})
                
            if ud_coro_or_val is not None:
                ud_data = results[task_indices['ud']]
            else:
                ud_data = self.last_state.get('ud', {})

            # Handle exceptions gracefully
            if isinstance(pp_data, Exception) or pp_data is None:
                logger.error(f"Error fetching PrizePicks: {pp_data}")
                pp_data = self.last_state.get('pp', {})
            else:
                if pp_coro_or_val is not None:
                    self.last_pp_fetch_time = now

            if isinstance(ud_data, Exception) or ud_data is None:
                logger.error(f"Error fetching Underdog: {ud_data}")
                ud_data = self.last_state.get('ud', {})
            else:
                if ud_coro_or_val is not None:
                    self.last_ud_fetch_time = now

            if isinstance(kalshi_data, Exception) or kalshi_data is None:
                logger.error(f"Error fetching Kalshi: {kalshi_data}")
                kalshi_data = self.last_state.get('kalshi', {})

            if isinstance(polymarket_data, Exception) or polymarket_data is None:
                logger.error(f"Error fetching Polymarket: {polymarket_data}")
                polymarket_data = self.last_state.get('polymarket', {})

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

    async def start_snapshot_loop(self):
        """Asynchronously triggers MarketDataLogger.log_snapshot() every 5 minutes (300 seconds)."""
        logger.info("Market logger 5-minute snapshot loop started in background.")
        while True:
            await asyncio.sleep(300)
            try:
                if self.market_logger:
                    logger.info("Triggering 5-minute market board snapshot to SQLite database...")
                    await self.market_logger.log_snapshot(self.last_state.get('pp'), self.sharp_cache)
            except Exception as e:
                logger.error(f"Error during scheduled SQLite snapshot: {e}")

    async def _check_line_slippage(self, book: str, player_name: str, prop_name: str, original_line: float, timestamp_str: str):
        """Checks line slippage against cached board state to avoid extra network requests that cause IP bans."""
        await asyncio.sleep(15)
        new_board = self.last_state.get(book, {})
            
        # Re-check if the player + prop still exists with the exact same line
        line_found = False
        for entry in new_board.values():
            if clean_player_name(entry.get('player', '')) == clean_player_name(player_name) and entry.get('prop', '') == prop_name:
                if abs(float(entry.get('line', 0.0) or 0.0) - original_line) < 0.01:
                    line_found = True
                    break
                    
        status = "EXECUTABLE" if line_found else "SLIPPED"
        
        try:
            from src.utils import WalkForwardLogger
            WalkForwardLogger.update_status(player_name, prop_name, original_line, timestamp_str, status)
        except Exception as e:
            logger.error(f"Failed to update walk-forward logger status: {e}")


async def main():
    poller = UnifiedRestPoller()

    print("\n" + "=" * 60)
    print(" HIGH-FREQUENCY POLLER (HFP) — DUAL-TIER ARCHITECTURE")
    print("=" * 60)
    print(" Tier 1 (Retail): PrizePicks + Underdog Fantasy (15s cycle)")
    print(" Tier 2 (Sharp):  Pinnacle/DK (45-min cache) + Kalshi + Polymarket")
    print("=" * 60 + "\n")

    # Start the non-blocking background market snapshot loop
    asyncio.create_task(poller.start_snapshot_loop())

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

        if poller.rate_limited:
            import random
            wait_time = random.uniform(18.5, 26.5)
            print(f"\n>>> Rate limit / WAF signature detected! Falling back to jittered sleep: {wait_time:.2f} seconds...\n")
            await asyncio.sleep(wait_time)
            # Reset rate limit flag to retry normal operation in the next cycle
            poller.rate_limited = False
        else:
            print(f"\n>>> Waiting 15 seconds...\n")
            await asyncio.sleep(15)


if __name__ == "__main__":
    asyncio.run(main())
