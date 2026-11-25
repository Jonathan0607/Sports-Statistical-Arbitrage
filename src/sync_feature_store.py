import os
import sys
import asyncio
import json
import pandas as pd
import sqlalchemy
import redis.asyncio as aioredis
import logging
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import LEAGUE_CONFIG

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FeatureStoreSync")

async def sync_postgres_to_redis(league: str = "NBA"):
    league_key = league.upper() if league else "NBA"
    reg_minutes = LEAGUE_CONFIG.get(league_key, LEAGUE_CONFIG["NBA"])["regulation_minutes"]

    db_uri = os.getenv("POSTGRES_URI", "postgresql://postgres:password@localhost:5432/sports_db")
    if db_uri.startswith("postgres://"):
        db_uri = db_uri.replace("postgres://", "postgresql+psycopg2://", 1)

    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", 6379))
    redis_client = aioredis.Redis(host=redis_host, port=redis_port, decode_responses=True)

    logger.info(f"Querying PostgreSQL for historical player matrices ({league_key}, regulation={reg_minutes:.1f}m)...")
    engine = sqlalchemy.create_engine(db_uri)

    # Optimized Window Function: Grabs the last 100 games per player in a single pass
    query = """
        WITH RankedGames AS (
            SELECT g.player_id, gm.start_time, gm.home_team_id, gm.away_team_id,
                   g.minutes_played, g.points, g.rebounds, g.assists, g.turnovers,
                   g.steals, g.blocks, g.three_pointers_made, g.usage_rate,
                   ROW_NUMBER() OVER(PARTITION BY g.player_id ORDER BY gm.start_time DESC) as rn
            FROM player_game_logs g
            JOIN games gm ON g.game_id = gm.game_id
            WHERE (:league IS NULL OR gm.sport = :league)
        )
        SELECT * FROM RankedGames WHERE rn <= 100 ORDER BY player_id, start_time ASC;
    """

    with engine.connect() as conn:
        df = pd.read_sql(sqlalchemy.text(query), conn, params={"league": league_key})

    # Dynamic time-normalization logic using league regulation minutes
    if not df.empty and 'minutes_played' in df.columns:
        df['norm_reb_rate'] = (df['rebounds'] / (df['minutes_played'] + 1e-5)) * reg_minutes
        df['norm_ast_rate'] = (df['assists'] / (df['minutes_played'] + 1e-5)) * reg_minutes

    # Convert start_time to ISO strings for JSON serialization
    if 'start_time' in df.columns:
        df['start_time'] = df['start_time'].astype(str)

    logger.info(f"Loaded {len(df)} rows. Pipelining to RedisJSON...")
    
    grouped = df.groupby("player_id")
    pipe = redis_client.pipeline()

    for player_id, group in grouped:
        records = group.drop(columns=['rn']).to_dict(orient='records')
        pipe.set(f"player:history:{player_id}", json.dumps(records))

    await pipe.execute()
    logger.info(f"Successfully synced {len(grouped)} player matrices to Redis for {league_key}.")

if __name__ == "__main__":
    target_league = sys.argv[1] if len(sys.argv) > 1 else "NBA"
    asyncio.run(sync_postgres_to_redis(target_league))

