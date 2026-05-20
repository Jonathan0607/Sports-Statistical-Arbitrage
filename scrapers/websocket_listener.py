import os
import asyncio
import json
import logging
import websockets
from python_socks.async_.asyncio import Proxy
from dotenv import load_dotenv
from infrastructure.redis_client import RedisClient
from infrastructure.entity_resolver import EntityResolver

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SharpBookWebsocketListener:
    """
    High-frequency real-time WebSocket listener designed to consume feeds
    from sharp bookmakers (e.g., Pinnacle, Bookmaker.eu) or odd consolidators.
    """
    def __init__(self):
        self.api_key = os.getenv("SHARP_BOOK_API_KEY", "")
        self.proxy_url = os.getenv("PROXY_NETWORK_URL", None)
        # Default mock URI for setup
        self.uri = "wss://api.sharpbookmaker.com/v1/nba/ticks"
        self.is_running = False
        self.redis = RedisClient()
        self.resolver = EntityResolver()

    async def connect_and_listen(self):
        """
        Manages the persistent connection to the websocket server with 
        exponential backoff reconnection logic.
        """
        self.is_running = True
        retry_delay = 1.0  # seconds
        
        while self.is_running:
            try:
                headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
                logger.info(f"Connecting to websocket feed: {self.uri}")
                
                # Configure proxy tunnel if PROXY_NETWORK_URL is set
                connect_kwargs = {"extra_headers": headers}
                if self.proxy_url:
                    logger.info(f"Routing through proxy: {self.proxy_url}")
                    proxy = Proxy.from_url(self.proxy_url)
                    sock = await proxy.connect(dest_host=self.uri.split('//')[1].split('/')[0], dest_port=443)
                    connect_kwargs["sock"] = sock
                else:
                    logger.info("No proxy configured, connecting directly.")
                
                async with websockets.connect(self.uri, **connect_kwargs) as websocket:
                    retry_delay = 1.0  # reset backoff on success
                    logger.info("Successfully connected to websocket stream.")
                    
                    # Send subscription payload
                    sub_payload = {
                        "action": "subscribe",
                        "channel": "nba_player_points",
                        "market": "player_props"
                    }
                    await websocket.send(json.dumps(sub_payload))
                    logger.info("Subscription message transmitted.")

                    while self.is_running:
                        try:
                            message = await websocket.recv()
                            await self.process_message(message)
                        except websockets.ConnectionClosed:
                            logger.warning("Websocket connection closed by host.")
                            break
                            
            except Exception as e:
                logger.error(f"Error in websocket connection: {e}")
                logger.info(f"Reconnecting in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 60.0) # exponential backoff capped at 60s

    async def process_message(self, message: str):
        """
        Process incoming ticker updates, resolve player identities, 
        and push harmonized payloads to Redis.
        """
        try:
            data = json.loads(message)
            logger.debug(f"Received tick: {data}")
            
            # 1. Parse raw incoming payload fields
            raw_player_name = data.get("player_name", "unknown_player")
            bookmaker = data.get("bookmaker", "sharp_book")
            line = float(data.get("line", 0.0))
            over_odds = int(data.get("over_odds", -110))
            under_odds = int(data.get("under_odds", -110))

            # 2. Hardened Entity Resolution step (Sprint 2 Core)
            master_player_id = self.resolver.resolve_player(
                book_name=bookmaker, 
                remote_name=raw_player_name
            )

            # 3. Push harmonized data to the Redis O(1) storage pipeline
            self.redis.process_incoming_tick(
                player_id=master_player_id,
                book=bookmaker,
                line=line,
                over_odds=over_odds,
                under_odds=under_odds
            )
            logger.info(f"[TICK SUCCESS] Harmonized {raw_player_name} -> {master_player_id} | Line: {line}")
            
        except json.JSONDecodeError as err:
            logger.error(f"Failed to parse incoming payload as JSON: {err}")
        except Exception as err:
            logger.error(f"Error handling tick data in pipeline loop: {err}")

    def stop(self):
        """Gracefully stop the listener loop."""
        logger.info("Shutting down websocket listener...")
        self.is_running = False

if __name__ == "__main__":
    listener = SharpBookWebsocketListener()
    try:
        asyncio.run(listener.connect_and_listen())
    except KeyboardInterrupt:
        logger.info("Process terminated by user.")
