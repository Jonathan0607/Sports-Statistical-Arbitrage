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
