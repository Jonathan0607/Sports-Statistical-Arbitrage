import os
import csv
import sqlite3
import logging
import asyncio
from datetime import datetime, timezone, timedelta
import threading
from apscheduler.schedulers.background import BackgroundScheduler
import numpy as np
import sys

# Ensure root directory is on the path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# In flat layout, these will come from src.models and src.risk
try:
    from src.models import MarketMicrostructure
    from src.risk import Backtester
except ImportError:
    # Fallback to package layout imports during transition/testing
    from src.models.models import MarketMicrostructure
    from src.risk.backtesting import Backtester

logger = logging.getLogger("MarketDataLogger")
logger_telemetry = logging.getLogger("WalkForwardTelemetry")


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1: Market Data Logger (SQLite)
# ═══════════════════════════════════════════════════════════════════════════════

class MarketDataLogger:
    DB_PATH = "logs/market_data.db"

    def __init__(self, db_path: str = None):
        """Initializes the database path and builds the schema if needed."""
        self.db_path = db_path or self.DB_PATH
        self._init_db()

    def _init_db(self):
        """Creates the logs directory and sets up the board_snapshots SQLite table."""
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS board_snapshots (
                    Timestamp DATETIME,
                    Book TEXT,
                    Player TEXT,
                    Stat TEXT,
                    Line REAL,
                    Over_Prob REAL,
                    Under_Prob REAL,
                    Status TEXT
                )
            """)
            conn.commit()
            logger.info(f"Initialized SQLite database at {self.db_path} with board_snapshots table.")
        except Exception as e:
            logger.error(f"Failed to initialize SQLite database: {e}")
            raise
        finally:
            conn.close()

    async def log_snapshot(self, prize_picks_data: dict, pinnacle_data: dict) -> None:
        """
        Asynchronously parses and inserts current board state into SQLite logs database.
        Offloads synchronous disk I/O / database operations to a threadpool via asyncio.to_thread.
        """
        if not prize_picks_data and not pinnacle_data:
            logger.debug("log_snapshot called with empty data, skipping.")
            return

        timestamp = datetime.now(timezone.utc).isoformat()
        records = []

        # 1. Parse PrizePicks
        if prize_picks_data:
            for item in prize_picks_data.values():
                player = item.get("player", "")
                stat = item.get("prop", "")
                line = item.get("line")
                try:
                    line_val = float(line) if line is not None else 0.0
                except (ValueError, TypeError):
                    line_val = 0.0

                records.append((
                    timestamp,
                    "PrizePicks",
                    player,
                    stat,
                    line_val,
                    None,
                    None,
                    "ACTIVE"
                ))

        # 2. Parse Pinnacle (sharp cache)
        if pinnacle_data:
            for player, stats in pinnacle_data.items():
                for stat, lines_dict in stats.items():
                    for line, data in lines_dict.items():
                        try:
                            line_val = float(line)
                        except (ValueError, TypeError):
                            line_val = 0.0

                        over_prob = data.get("pinnacle_true_prob")
                        under_prob = (1.0 - over_prob) if over_prob is not None else None

                        records.append((
                            timestamp,
                            "Pinnacle",
                            player,
                            stat,
                            line_val,
                            over_prob,
                            under_prob,
                            "ACTIVE"
                        ))

        if records:
            await asyncio.to_thread(self._bulk_insert, records)

    def _bulk_insert(self, records: list):
        """Performs a synchronous bulk INSERT on a dedicated database connection (called on background thread)."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.executemany("""
                INSERT INTO board_snapshots (Timestamp, Book, Player, Stat, Line, Over_Prob, Under_Prob, Status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, records)
            conn.commit()
            logger.info(f"Bulk-inserted {len(records)} board snapshot entries into SQLite.")
        except Exception as e:
            logger.error(f"SQLite bulk insert failed: {e}")
        finally:
            conn.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2: Walk Forward Telemetry Logger
# ═══════════════════════════════════════════════════════════════════════════════

class WalkForwardLogger:
    CSV_PATH = "logs/telemetry.csv"
    _lock = threading.Lock()

    @classmethod
    def init_csv(cls):
        """Initializes the telemetry CSV file with headers if it does not exist."""
        os.makedirs(os.path.dirname(cls.CSV_PATH), exist_ok=True)
        with cls._lock:
            if not os.path.exists(cls.CSV_PATH):
                with open(cls.CSV_PATH, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "Signal_ID", "Timestamp", "Player", "Stat", "Line", 
                        "ZINB_Prob", "Edge", "Status", "CLV", "Game_Time"
                    ])
                logger_telemetry.info(f"Initialized telemetry log file: {cls.CSV_PATH}")

    @classmethod
    def generate_id(cls, player: str, stat: str, line: float, timestamp: str) -> str:
        return f"{player}_{stat}_{line}_{timestamp}"

    @classmethod
    def log_signal(cls, player: str, stat: str, line: float, zinb_prob: float, edge: float, timestamp: str, game_time: str = "") -> str:
        """Records a new signal entry as PENDING status."""
        cls.init_csv()
        signal_id = cls.generate_id(player, stat, line, timestamp)
        
        # Default game_time if empty: 3 hours from now
        if not game_time:
            game_time = (datetime.now(timezone.utc) + timedelta(hours=3)).isoformat()
            
        with cls._lock:
            with open(cls.CSV_PATH, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    signal_id, timestamp, player, stat, line, 
                    f"{zinb_prob:.6f}", f"{edge:.6f}", "PENDING", "", game_time
                ])
        logger_telemetry.info(f"Logged signal {signal_id} to walk-forward CSV.")
        return signal_id

    @classmethod
    def update_status(cls, player: str, stat: str, line: float, timestamp: str, status: str):
        """Updates the status of a signal (e.g. EXECUTABLE, SLIPPED)."""
        cls.init_csv()
        signal_id = cls.generate_id(player, stat, line, timestamp)
        
        with cls._lock:
            rows = []
            updated = False
            with open(cls.CSV_PATH, 'r') as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames
                for row in reader:
                    if row.get("Signal_ID") == signal_id:
                        row["Status"] = status
                        updated = True
                    rows.append(row)
            if updated:
                with open(cls.CSV_PATH, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
        logger_telemetry.info(f"Updated signal {signal_id} status to: {status}")

    @classmethod
    def update_clv(cls, signal_id: str, clv: float):
        """Updates the CLV value of a signal."""
        cls.init_csv()
        with cls._lock:
            rows = []
            updated = False
            with open(cls.CSV_PATH, 'r') as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames
                for row in reader:
                    if row.get("Signal_ID") == signal_id:
                        row["CLV"] = f"{clv:.6f}"
                        updated = True
                    rows.append(row)
            if updated:
                with open(cls.CSV_PATH, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
        logger_telemetry.info(f"Updated signal {signal_id} CLV to: {clv:.2%}")

def check_clv_job():
    """
    Background job that reads telemetry.csv, checks for EXECUTABLE signals
    approaching game time (5 minutes before), polls the sharp API, and logs CLV.
    """
    WalkForwardLogger.init_csv()
    executable_signals = []
    
    with WalkForwardLogger._lock:
        with open(WalkForwardLogger.CSV_PATH, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("Status") == "EXECUTABLE" and not row.get("CLV"):
                    executable_signals.append(row)
                    
    if not executable_signals:
        return
        
    now_utc = datetime.now(timezone.utc)
    
    # Filter signals that are <= 5 minutes from game start or have already started
    signals_to_check = []
    for sig in executable_signals:
        gt_str = sig.get("Game_Time", "")
        if gt_str:
            try:
                # Sanitize: Handle Z suffix and truncated timezone offsets (+00:0 -> +00:00)
                gt_str = gt_str.replace("Z", "+00:00")
                if len(gt_str) > 3 and gt_str[-3] == '+' and gt_str[-1] != ':':
                    # Matches patterns like +00:0 (missing trailing digit)
                    gt_str += "0"
                elif gt_str.endswith("+00:0"):
                    gt_str += "0"
                gt = datetime.fromisoformat(gt_str)
                if now_utc >= gt - timedelta(minutes=5):
                    signals_to_check.append(sig)
            except Exception as e:
                logger_telemetry.error(f"Failed to parse Game_Time for CLV check: {e}")
                
    if not signals_to_check:
        return
        
    logger_telemetry.info(f"Found {len(signals_to_check)} executable signals approaching game time. Polling sharp API...")
    
    # Import SharpLineProvider lazily to prevent circular dependencies
    from scrapers.unified_api import SharpLineProvider
    provider = SharpLineProvider()
    
    # Run async fetch in sync scheduler thread
    def run_async(coro):
        try:
            return asyncio.run(coro)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro)
            
    try:
        # Fetch fresh sharp lines from ParlayAPI directly
        sharp_lines = run_async(provider.fetch_sharp_lines(sport="basketball_nba"))
    except Exception as e:
        logger_telemetry.error(f"Telemetry CLV check failed to fetch sharp lines: {e}")
        return
        
    for sig in signals_to_check:
        player = sig.get("Player", "")
        stat = sig.get("Stat", "")
        line_val = float(sig.get("Line", 0.0))
        placed_prob = float(sig.get("ZINB_Prob", 0.50))
        signal_id = sig.get("Signal_ID", "")
        
        # Resolve clean names for key lookups
        from infrastructure.entity_resolver import clean_player_name
        player_clean = clean_player_name(player)
        
        # Map stat string to db column name
        # A simple mapper
        def map_stat(s):
            s_lower = s.lower()
            if 'point' in s_lower: return 'points'
            if 'rebound' in s_lower: return 'rebounds'
            if 'assist' in s_lower: return 'assists'
            if 'turnover' in s_lower: return 'turnovers'
            return None
            
        mapped_stat = map_stat(stat)
        
        closing_prob = None
        if sharp_lines and player_clean in sharp_lines:
            player_stats = sharp_lines[player_clean]
            if mapped_stat in player_stats:
                lines_dict = player_stats[mapped_stat]
                # Find matching line
                for line_key, data in lines_dict.items():
                    if abs(float(line_key) - line_val) < 0.01:
                        closing_prob = data.get("pinnacle_true_prob")
                        break
                        
        if closing_prob is None:
            # If line was pulled or not found, fallback to placed_prob (CLV = 0.0)
            logger_telemetry.warning(f"Closing line not found for {player_clean} {stat} at {line_val}. Defaulting CLV to 0.0.")
            closing_prob = placed_prob
            
        clv = Backtester.calculate_clv(placed_prob, closing_prob)
        WalkForwardLogger.update_clv(signal_id, clv)

# Initialize and start APScheduler background job (with daemon threads)
scheduler = BackgroundScheduler(daemon=True)
scheduler.add_job(check_clv_job, 'interval', minutes=1, id="check_clv_job")
scheduler.start()
logger_telemetry.info("APScheduler telemetry daemon background job started (runs every 1 minute).")
