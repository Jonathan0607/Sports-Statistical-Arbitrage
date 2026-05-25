import os
import sys
import json
import asyncio
import psycopg2
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our pipeline components
from infrastructure.redis_client import RedisClient
from infrastructure.entity_resolver import EntityResolver
# WebSocket Listener and Features modules have been consolidated/purged.

load_dotenv()

async def run_diagnostics():
    print("==================================================")
    print("🚀 STARTING E2E PIPELINE DIAGNOSTICS")
    print("==================================================")
    
    # STEP 1: Seed PostgreSQL
    print("\n[1/5] Injecting test mapping into PostgreSQL...")
    conn_uri = os.getenv("POSTGRES_URI", "postgresql://postgres:password@localhost:5432/quant_engine")
    try:
        conn = psycopg2.connect(conn_uri)
        cur = conn.cursor()
        # Insert a master player
        cur.execute("INSERT INTO players (master_player_id, full_name) VALUES ('pj_washington_master', 'PJ Washington') ON CONFLICT DO NOTHING;")
        # Insert the sportsbook-specific mapping
        cur.execute("INSERT INTO player_mappings (master_player_id, book_name, remote_player_name) VALUES ('pj_washington_master', 'DraftKings', 'p.j. washington') ON CONFLICT DO NOTHING;")
        conn.commit()
        cur.close()
        conn.close()
        print("✅ PostgreSQL seeding successful.")
    except Exception as e:
        print(f"❌ PostgreSQL Error: {e}")
        return

    # STEP 2 & 3: WebSocket Listener is deprecated and purged. Skipping simulation...
    pass

    # STEP 4: Verify Redis Output
    print("\n[4/5] Inspecting Redis In-Memory State...")
    r = RedisClient().get_client()
    
    hash_state = r.hgetall("state:odds:nba:points:pj_washington_master")
    print(f"   🔹 Hash State: {hash_state}")
    
    zset_state = r.zrange("history:nba:points:pj_washington_master:draftkings", 0, -1, withscores=True)
    print(f"   🔹 Lookback Window (Sorted Set): {zset_state}")
    
    stream_state = r.xrange("stream:ticks:nba:points", min='-', max='+')
    latest_stream = stream_state[-1] if stream_state else "Empty"
    print(f"   🔹 Latest Stream Event: {latest_stream}")

    # STEP 5: Feature Engineering is deprecated and purged. Skipping...
    pass

    print("\n==================================================")
    print("✅ DIAGNOSTICS COMPLETE")
    print("==================================================")

if __name__ == "__main__":
    asyncio.run(run_diagnostics())
