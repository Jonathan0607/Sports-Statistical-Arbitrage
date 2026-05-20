import os
import logging
import redis
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RedisClient:
    """
    A thread-safe, singleton-pattern Redis client with connection pooling
    tailored for high-frequency low-latency sports arbitrage data storage.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(RedisClient, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.host = os.getenv("REDIS_HOST", "localhost")
        self.port = int(os.getenv("REDIS_PORT", 6379))
        self.password = os.getenv("REDIS_PASSWORD", None)
        
        try:
            logger.info(f"Initializing Redis Connection Pool to {self.host}:{self.port}...")
            self.pool = redis.ConnectionPool(
                host=self.host,
                port=self.port,
                password=self.password,
                decode_responses=True,
                socket_timeout=5.0,
                retry_on_timeout=True
            )
            self.client = redis.Redis(connection_pool=self.pool)
            self._initialized = True
            logger.info("Redis Connection Pool initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Redis client: {e}")
            raise e

    def ping(self) -> bool:
        """Verify connection viability."""
        try:
            return self.client.ping()
        except redis.ConnectionError as e:
            logger.error(f"Redis ping failed: {e}")
            return False

    def process_incoming_tick(self, player_id: str, book: str, line: float, over_odds: int, under_odds: int):
        """
        O(1) execution pipeline for incoming WebSocket odds ticks.
        """
        import time
        timestamp = int(time.time() * 1000) # Millisecond precision
        
        # 1. State Snapshot (Hashes) - Overwrites the current line instantly
        state_key = f"state:odds:nba:points:{player_id}"
        self.client.hset(state_key, mapping={
            f"{book}_line": line,
            f"{book}_over": over_odds,
            f"{book}_under": under_odds,
            f"{book}_last_updated": timestamp
        })

        # 2. Lookback Window (Sorted Sets) - Used by Jump-Diffusion models
        history_key = f"history:nba:points:{player_id}:{book}"
        import json
        tick_payload = json.dumps({"line": line, "over": over_odds, "under": under_odds})
        self.client.zadd(history_key, {tick_payload: timestamp})
        
        # Trim to last 5 minutes (300,000 ms) to prevent RAM bloat
        cutoff = timestamp - 300000
        self.client.zremrangebyscore(history_key, 0, cutoff)

        # 3. Message Brokering (Streams) - Push to XGBoost/ZINB models downstream
        stream_payload = {
            "player_id": player_id,
            "book": book,
            "line": str(line),
            "over_odds": str(over_odds),
            "under_odds": str(under_odds),
            "ts": str(timestamp)
        }
        self.client.xadd("stream:ticks:nba:points", stream_payload)
        self.client.xtrim("stream:ticks:nba:points", maxlen=10000)

    def get_client(self) -> redis.Redis:
        """Retrieve direct redis-py client instance."""
        return self.client

if __name__ == "__main__":
    # Self-test script when run directly
    print("Testing Redis client connection...")
    try:
        r_client = RedisClient()
        if r_client.ping():
            print("Successfully connected to Redis instance.")
        else:
            print("Could not connect to Redis instance.")
    except Exception as err:
        print(f"Exception during self-test: {err}")
