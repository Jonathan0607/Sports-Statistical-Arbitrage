import os
import sys
import json
import time
import hashlib
import asyncio
import logging
import urllib.request
import urllib.error
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from infrastructure.redis_client import RedisClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DiscordNotifier")

QUEUE_NAME = "execution_queue:prizepicks"
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")
MAX_ALERT_AGE_SECONDS = 20  # Drop alerts older than this to avoid stale line drift

# Liquidity Filter Settings
MAX_ANCHOR_SPREAD = 0.10    # Max 10% bid-ask spread on anchors
MIN_ANCHOR_VOLUME = 500.0   # Min $500 daily volume/open interest on anchors
DEDUP_TTL_SECONDS = 300     # Suppress duplicate alerts for 5 minutes

def get_semantic_hash(legs: list) -> str:
    """
    Generates a unique, order-invariant hash based on leg properties to deduplicate slips.
    """
    normalized_legs = []
    for leg in legs:
        player = str(leg.get("player", "")).strip().lower()
        stat = str(leg.get("stat", "")).strip().lower()
        side = str(leg.get("side", "")).strip().lower()
        line = str(leg.get("target_execution_line", 0.0))
        normalized_legs.append(f"{player}:{stat}:{side}:{line}")
    
    # Sort to make order-invariant
    normalized_legs.sort()
    legs_string = "|".join(normalized_legs)
    return hashlib.md5(legs_string.encode('utf-8')).hexdigest()

def passes_liquidity_filter(legs: list) -> bool:
    """
    Filters out phantom liquidity edges from thin prediction or sharp markets.
    """
    for leg in legs:
        spread = leg.get("anchor_spread", 0.0)
        volume = leg.get("anchor_volume", 1000.0)  # Default high if not populated
        
        if spread > MAX_ANCHOR_SPREAD:
            logger.warning(f"Skipping slip: Anchor spread ({spread:.2%}) exceeds limit ({MAX_ANCHOR_SPREAD:.2%}) for leg {leg.get('player')}")
            return False
            
        if volume < MIN_ANCHOR_VOLUME:
            logger.warning(f"Skipping slip: Anchor volume (${volume:.2f}) below threshold (${MIN_ANCHOR_VOLUME:.2f}) for leg {leg.get('player')}")
            return False
    return True

def format_discord_embed(payload: dict) -> dict:
    """
    Unpacks payload and maps details into a clean Discord Embed payload structure.
    """
    ev_edge = payload.get("ev_edge", 0.0)
    kelly = payload.get("kelly_fraction", 0.0)
    stake = payload.get("risk_adjusted_stake", 5.0)
    slip_id = payload.get("slip_id", "unknown")
    
    # Color coding: Green for premium edges, Orange for regular edges
    color = 65280 if ev_edge >= 0.05 else 16753920
    
    embed = {
        "title": "🚨 +EV Arbitrage Opportunity Detected",
        "description": f"**Slip ID**: `{slip_id}`\n**Model Edge**: `{ev_edge:.2%}` | **Recommended Stake**: `${stake:.2f}` (Kelly: `{kelly:.2%}`)",
        "color": color,
        "fields": [],
        "timestamp": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(payload.get("timestamp", time.time()))),
        "footer": {
            "text": "Antigravity StatArb Alert System"
        }
    }
    
    for idx, leg in enumerate(payload.get("legs", [])):
        player = leg.get("player")
        stat = leg.get("stat")
        side = str(leg.get("side", "")).upper()
        line = leg.get("target_execution_line")
        
        embed["fields"].append({
            "name": f"Leg {idx+1}: {player} ({stat})",
            "value": f"**Selection**: {side} {line}\n[Open PrizePicks Board](https://app.prizepicks.com/)",
            "inline": False
        })
        
    return {"embeds": [embed]}

def send_discord_webhook_sync(url: str, data: dict) -> tuple[int, str]:
    """
    Sends the embed payload synchronously. Returns (status_code, response_body).
    """
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode('utf-8'),
        headers={
            'Content-Type': 'application/json',
            'User-Agent': 'DiscordNotifier/1.0'
        },
        method='POST'
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.status, response.read().decode('utf-8')
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode('utf-8')
        except Exception:
            body = str(e)
        return e.code, body
    except Exception as e:
        return 0, str(e)

async def send_discord_webhook(loop, url: str, data: dict):
    """
    Sends the embed payload with built-in rate-limiting retry handling.
    """
    for attempt in range(3):
        status, body = await loop.run_in_executor(None, send_discord_webhook_sync, url, data)
        if status in (200, 204):
            logger.info("Alert successfully dispatched to Discord.")
            return
        elif status == 429:
            retry_after = 1.0
            try:
                res_json = json.loads(body)
                retry_after = float(res_json.get("retry_after", 1.0))
            except Exception:
                pass
            logger.warning(f"Discord rate limit hit. Retrying in {retry_after} seconds...")
            await asyncio.sleep(retry_after)
        else:
            logger.error(f"Failed to post to Discord (HTTP {status}): {body}")
            return

async def run_notifier():
    if not DISCORD_WEBHOOK_URL:
        logger.critical("DISCORD_WEBHOOK_URL environment variable is missing! Exiting notifier...")
        return
        
    redis_client = RedisClient().get_client()
    logger.info("Discord Notifier Service live. Monitoring Redis queue...")
    
    while True:
        try:
            # Poll Redis non-blocking pop
            loop = asyncio.get_event_loop()
            res = await loop.run_in_executor(None, lambda: redis_client.rpop(QUEUE_NAME))
            
            if res:
                payload = json.loads(res)
                slip_id = payload.get("slip_id", "unknown")
                timestamp = payload.get("timestamp", 0)
                legs = payload.get("legs", [])
                
                # 1. Latency check (Drift Mitigation)
                age = time.time() - timestamp
                if age > MAX_ALERT_AGE_SECONDS:
                    logger.warning(f"Skipping stale slip {slip_id} (age: {age:.1f}s > max: {MAX_ALERT_AGE_SECONDS}s)")
                    continue
                    
                # 2. Phantom Liquidity check
                if not passes_liquidity_filter(legs):
                    continue
                    
                # 3. Deduplication check
                sem_hash = get_semantic_hash(legs)
                dedup_key = f"dedup:slip:{sem_hash}"
                
                # Attempt to set key with EX TTL to lock it
                is_unique = await loop.run_in_executor(
                    None, 
                    lambda: redis_client.set(dedup_key, "1", nx=True, ex=DEDUP_TTL_SECONDS)
                )
                
                if not is_unique:
                    logger.info(f"Skipping duplicate slip alert for {slip_id} (semantically identical alert sent recently).")
                    continue
                
                # 4. Format & Send alert
                embed_payload = format_discord_embed(payload)
                await send_discord_webhook(loop, DISCORD_WEBHOOK_URL, embed_payload)
                
        except Exception as e:
            logger.error(f"Error in Notifier loop: {e}", exc_info=True)
            
        await asyncio.sleep(0.1)

if __name__ == "__main__":
    try:
        asyncio.run(run_notifier())
    except KeyboardInterrupt:
        logger.info("Notifier shutting down...")
