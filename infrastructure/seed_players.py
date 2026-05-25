import os
import sys
import logging
import pandas as pd
import psycopg2
from dotenv import load_dotenv

# Ensure root directory is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import curl_cffi and patch nba_api requests to bypass TLS fingerprinting blocks
from curl_cffi import requests as cffi_requests
import nba_api.stats.library.http as nba_http

class MockRequests:
    @staticmethod
    def get(*args, **kwargs):
        session = cffi_requests.Session(impersonate="chrome120")
        return session.get(*args, **kwargs)

nba_http.requests = MockRequests

from nba_api.stats.endpoints import playergamelogs

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SeedPlayers")

def seed():
    db_uri = os.getenv("POSTGRES_URI", "postgresql://postgres:password@localhost:5432/sports_db")
    
    # 1. Fetch player game logs to extract player name to ID mappings
    logger.info("Fetching game logs from nba_api to extract player mappings...")
    try:
        logs = playergamelogs.PlayerGameLogs(season_nullable="2024-25")
        df = logs.get_data_frames()[0]
    except Exception as e:
        logger.error(f"Failed to fetch game logs from API: {e}")
        return
        
    unique_players = df[['PLAYER_ID', 'PLAYER_NAME']].drop_duplicates()
    logger.info(f"Found {len(unique_players)} unique players in 2024-25 season.")
    
    # 2. Insert into PostgreSQL
    try:
        conn = psycopg2.connect(db_uri)
        cursor = conn.cursor()
        
        # Insert into players and player_mappings
        players_seeded = 0
        mappings_seeded = 0
        
        for _, row in unique_players.iterrows():
            player_id = str(row['PLAYER_ID'])
            player_name = str(row['PLAYER_NAME'])
            
            # Insert into players
            cursor.execute("""
                INSERT INTO players (master_player_id, full_name, sport, is_active)
                VALUES (%s, %s, 'NBA', TRUE)
                ON CONFLICT (master_player_id) DO UPDATE SET full_name = EXCLUDED.full_name;
            """, (player_id, player_name))
            players_seeded += 1
            
            # Insert mappings for various books (normalized names handled by resolver fallback, but explicit mappings help)
            for book in ['DraftKings', 'PrizePicks', 'Pinnacle']:
                cursor.execute("""
                    INSERT INTO player_mappings (master_player_id, book_name, remote_player_name)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (book_name, remote_player_name) DO NOTHING;
                """, (player_id, book, player_name))
                mappings_seeded += 1
                
                # Also seed lowercased version as explicit mapping
                cursor.execute("""
                    INSERT INTO player_mappings (master_player_id, book_name, remote_player_name)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (book_name, remote_player_name) DO NOTHING;
                """, (player_id, book, player_name.lower()))
                
        conn.commit()
        cursor.close()
        conn.close()
        logger.info(f"Seeding complete. Seeded {players_seeded} players and {mappings_seeded} base book mappings.")
        
    except Exception as e:
        logger.error(f"Postgres insertion failed: {e}")

if __name__ == "__main__":
    seed()
