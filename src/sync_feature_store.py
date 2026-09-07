import os
import asyncio
import pandas as pd
import sqlalchemy
import redis.asyncio as aioredis
import logging
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FeatureStoreSync")

async def sync_postgres_to_redis():
    db_uri = os.getenv("POSTGRES_URI", "postgresql://postgres:password@localhost:5432/sports_db")
    if db_uri.startswith("postgres://"):
        db_uri = db_uri.replace("postgres://", "postgresql+psycopg2://", 1)

    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", 6379))
    redis_client = aioredis.Redis(host=redis_host, port=redis_port, decode_responses=True)

    logger.info("Querying PostgreSQL for historical player matrices...")
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
        )
        SELECT * FROM RankedGames WHERE rn <= 100 ORDER BY player_id, start_time ASC;
    """

    with engine.connect() as conn:
        df = pd.read_sql(sqlalchemy.text(query), conn)

    # Convert start_time to ISO strings for JSON serialization
    if 'start_time' in df.columns:
        df['start_time'] = df['start_time'].astype(str)

    logger.info(f"Loaded {len(df)} rows. Pipelining to RedisJSON...")
    
    grouped = df.groupby("player_id")
    pipe = redis_client.pipeline()

    for player_id, group in grouped:
        records = group.drop(columns=['rn']).to_dict(orient='records')
        # Store as native JSON array at the player's dedicated key
        pipe.json().set(f"player:history:{player_id}", "$", records)

    await pipe.execute()
    logger.info(f"Successfully synced {len(grouped)} player matrices to Redis.")

if __name__ == "__main__":
    asyncio.run(sync_postgres_to_redis())
