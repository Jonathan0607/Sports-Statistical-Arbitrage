import os
import asyncio
import json
import logging
import websockets
from dotenv import load_dotenv

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
                
                async with websockets.connect(self.uri, extra_headers=headers) as websocket:
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
        Process incoming ticker updates. This should decode the json payload,
        extract key fields, and push to Redis/database.
        """
        try:
            data = json.loads(message)
            # Log tick events at debug level to avoid cluttering stdout at high frequency
            logger.debug(f"Received tick: {data}")
            
            # TODO: Add real time pipeline processing:
            # 1. Parse JSON update to extract bookmaker, player, line, odds
            # 2. Push raw data to Redis pub/sub or queue
            # 3. Trigger state updater/arbitrage calculator
            
        except json.JSONDecodeError as err:
            logger.error(f"Failed to parse incoming payload as JSON: {err}")
        except Exception as err:
            logger.error(f"Error handling tick data: {err}")

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
