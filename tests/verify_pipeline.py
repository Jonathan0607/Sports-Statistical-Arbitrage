import os
import json
import asyncio
import psycopg2
from dotenv import load_dotenv

# Import our pipeline components
from infrastructure.redis_client import RedisClient
from infrastructure.entity_resolver import EntityResolver
from scrapers.websocket_listener import SharpBookWebsocketListener
from features.physiological_features import PhysiologicalFeatureProcessor

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

    # STEP 2: Initialize Pipeline
    print("\n[2/5] Initializing WebSocket Listener & Entity Resolver...")
    listener = SharpBookWebsocketListener()
    print("✅ Listener initialized with Redis and Resolver attached.")

    # STEP 3: Simulate Tick
    print("\n[3/5] Simulating incoming WebSocket JSON tick for 'p.j. washington'...")
    mock_payload = json.dumps({
        "player_name": "p.j. washington",
        "bookmaker": "DraftKings",
        "line": 15.5,
        "over_odds": -115,
        "under_odds": -105
    })
    await listener.process_message(mock_payload)

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

    # STEP 5: Verify Feature Engineering
    print("\n[5/5] Testing Physiological Feature Matrix...")
    feat = PhysiologicalFeatureProcessor()
    matrix = feat.generate_fatigue_matrix(1, 3, "DEN", ["DAL", "LAL", "DEN"])
    print(f"   🔹 Matrix Output: {matrix}")

    print("\n==================================================")
    print("✅ DIAGNOSTICS COMPLETE")
    print("==================================================")

if __name__ == "__main__":
    asyncio.run(run_diagnostics())
