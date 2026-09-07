import os
import sys
import time
import json
import signal
import logging
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

# Load env variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SignalBufferWorker")

# Import RedisClient
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from infrastructure.redis_client import RedisClient

DB_URI = os.getenv("POSTGRES_URI", "postgresql://postgres:postgres@localhost:5432/sports_db")
if DB_URI.startswith("postgres://"):
    DB_URI = DB_URI.replace("postgres://", "postgresql+psycopg2://", 1)

class SignalBufferWorker:
    def __init__(self):
        self.running = True
        self.redis_client = RedisClient()
        self.db_conn = None
        self._init_db()

    def _init_db(self):
        try:
            self.db_conn = psycopg2.connect(DB_URI)
            logger.info("Successfully connected to PostgreSQL database.")
        except Exception as e:
            logger.critical(f"Failed to connect to PostgreSQL: {e}")
            sys.exit(1)

    def stop(self):
        logger.info("Stopping Signal Buffer Worker...")
        self.running = False

    def flush_signals(self):
        """Fetches signals from Redis list and writes them in bulk to Postgres."""
        r = self.redis_client.get_client()
        try:
            # Atomic fetch and clear using pipeline
            pipe = r.pipeline()
            pipe.lrange("queue:ev_signals_buffer", 0, -1)
            pipe.ltrim("queue:ev_signals_buffer", 1, 0)
            raw_signals, _ = pipe.execute()
        except Exception as e:
            logger.error(f"Redis pipeline error during flush: {e}")
            return

        if not raw_signals:
            return

        logger.info(f"Retrieved {len(raw_signals)} signals from Redis queue. Processing bulk insert...")

        # Parse signals
        parsed_signals = []
        for raw in raw_signals:
            try:
                sig = json.loads(raw)
                parsed_signals.append((
                    sig.get("player_id"),
                    sig.get("player_name"),
                    sig.get("sharp_book"),
                    sig.get("stale_book"),
                    sig.get("sharp_line"),
                    sig.get("stale_line"),
                    sig.get("sharp_over_odds"),
                    sig.get("stale_under_odds"),
                    sig.get("expected_value"),
                    sig.get("z_score"),
                    sig.get("kelly_fraction"),
                    sig.get("anchor_liquidity_usd")
                ))
            except Exception as pe:
                logger.error(f"Failed to parse buffered signal raw JSON: {raw}. Error: {pe}")

        if not parsed_signals:
            return

        # Bulk insert to Postgres
        insert_query = """
            INSERT INTO arb_signals (
                player_id, player_name, sharp_book, stale_book,
                sharp_line, stale_line, sharp_over_odds, stale_under_odds,
                expected_value, z_score, kelly_fraction, anchor_liquidity_usd
            ) VALUES %s;
        """
        try:
            # Ensure DB connection is open
            if self.db_conn.closed:
                logger.info("Database connection closed. Reconnecting...")
                self._init_db()
                
            cursor = self.db_conn.cursor()
            execute_values(cursor, insert_query, parsed_signals)
            self.db_conn.commit()
            cursor.close()
            logger.info(f"Successfully batch-inserted {len(parsed_signals)} signals into PostgreSQL.")
        except Exception as dbe:
            logger.error(f"Failed to batch-insert signals to database: {dbe}")
            # Try to roll back transaction
            try:
                self.db_conn.rollback()
            except Exception:
                pass
            # Push them back to Redis list so they are not lost!
            logger.warning("Re-buffering signals back to Redis list due to DB failure...")
            pipe = r.pipeline()
            for raw in raw_signals:
                pipe.rpush("queue:ev_signals_buffer", raw)
            pipe.execute()

    def run(self):
        logger.info("Signal Buffer Worker Daemon is active and polling Redis...")
        
        # Setup signal handlers for graceful shutdown
        def handle_signal(sig_num, frame):
            self.stop()
        
        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

        while self.running:
            try:
                self.flush_signals()
            except Exception as e:
                logger.error(f"Error in signal buffer loop: {e}")
            
            # Sleep 60 seconds, check running state periodically
            for _ in range(60):
                if not self.running:
                    break
                time.sleep(1)

        # Final flush on exit
        logger.info("Running final flush on exit...")
        try:
            self.flush_signals()
        except Exception as e:
            logger.error(f"Error during final flush: {e}")
            
        if self.db_conn:
            self.db_conn.close()
            logger.info("Database connection closed.")
        logger.info("Signal Buffer Worker stopped.")

if __name__ == "__main__":
    worker = SignalBufferWorker()
    worker.run()
