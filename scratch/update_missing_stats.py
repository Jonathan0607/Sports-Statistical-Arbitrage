#!/usr/bin/env python3
import os
import sys
import logging
import pandas as pd
import sqlalchemy
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("UpdateMissingStats")

try:
    from curl_cffi import requests as cffi_requests
    import nba_api.stats.library.http as nba_http
    from nba_api.stats.endpoints import leaguegamelog
except ImportError as e:
    logger.critical(f"Missing dependencies: {e}")
    sys.exit(1)

class MockRequests:
    @staticmethod
    def get(*args, **kwargs):
        session = cffi_requests.Session(impersonate="chrome120")
        if 'timeout' not in kwargs:
            kwargs['timeout'] = 30
        return session.get(*args, **kwargs)

nba_http.requests = MockRequests

DB_URI = os.getenv("POSTGRES_URI", "postgresql+psycopg2://postgres:password@localhost:5432/sports_db")
if DB_URI.startswith("postgres://"):
    DB_URI = DB_URI.replace("postgres://", "postgresql+psycopg2://", 1)

engine = sqlalchemy.create_engine(DB_URI)

def update_season_type(season_type: str):
    logger.info(f"Querying stats.nba.com for 2025-26 {season_type} data to update missing columns...")
    try:
        raw_log = leaguegamelog.LeagueGameLog(
            season='2025-26',
            player_or_team_abbreviation='P',
            season_type_all_star=season_type
        )
        df = raw_log.get_data_frames()[0]
    except Exception as e:
        logger.error(f"Failed pulling data from NBA API: {e}")
        return

    if df.empty:
        logger.warning(f"No records found for {season_type}")
        return

    logger.info(f"Found {len(df)} records. Performing database update...")
    
    # We will do updates in batches to be memory and transaction friendly
    batch_size = 1000
    records = []
    
    for idx, log in df.iterrows():
        try:
            game_id = int(log['GAME_ID'])
            player_id = str(log['PLAYER_ID'])
            fg3m = int(log['FG3M']) if not pd.isna(log['FG3M']) else 0
            stl = int(log['STL']) if not pd.isna(log['STL']) else 0
            blk = int(log['BLK']) if not pd.isna(log['BLK']) else 0
            records.append({
                "game_id": game_id,
                "player_id": player_id,
                "fg3m": fg3m,
                "stl": stl,
                "blk": blk
            })
        except Exception as e:
            logger.error(f"Error parsing row: {e}")

    logger.info(f"Batched {len(records)} updates. Executing...")
    
    update_stmt = sqlalchemy.text("""
        UPDATE player_game_logs
        SET three_pointers_made = :fg3m,
            steals = :stl,
            blocks = :blk
        WHERE game_id = :game_id AND player_id = :player_id;
    """)
    
    total_updated = 0
    with engine.begin() as conn:
        for i in range(0, len(records), batch_size):
            batch = records[i:i+batch_size]
            # Execute batch update
            conn.execute(update_stmt, batch)
            total_updated += len(batch)
            logger.info(f"Updated {total_updated}/{len(records)} records...")
            
    logger.info(f"Successfully completed update for {season_type}. Total rows updated: {total_updated}")

def main():
    update_season_type('Regular Season')
    update_season_type('Playoffs')
    logger.info("Update complete.")

if __name__ == "__main__":
    main()
