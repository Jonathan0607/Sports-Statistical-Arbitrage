#!/usr/bin/env python3
import os
import sys
import logging
import pandas as pd
import sqlalchemy
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize explicit logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DataBackfiller")

# Import curl_cffi and patch nba_api requests to bypass TLS fingerprinting blocks
try:
    from curl_cffi import requests as cffi_requests
    import nba_api.stats.library.http as nba_http
    from nba_api.stats.endpoints import leaguegamelog
except ImportError as e:
    logger.critical(f"Missing dependencies: {e}. Run: venv/bin/pip install curl_cffi nba_api")
    sys.exit(1)

class MockRequests:
    @staticmethod
    def get(*args, **kwargs):
        session = cffi_requests.Session(impersonate="chrome120")
        if 'timeout' not in kwargs:
            kwargs['timeout'] = 30
        return session.get(*args, **kwargs)

# Apply TLS fingerprinting patch
nba_http.requests = MockRequests

# Normalizing connection string for SQLAlchemy context managers
DB_URI = os.getenv("POSTGRES_URI", "postgresql+psycopg2://postgres:postgres@localhost:5432/sports_db")
if DB_URI.startswith("postgres://"):
    DB_URI = DB_URI.replace("postgres://", "postgresql+psycopg2://", 1)

engine = sqlalchemy.create_engine(DB_URI)

def extract_minutes(min_str) -> float:
    """Parses string 'MM:SS' or raw float values from NBA API logs safely."""
    if pd.isna(min_str) or not min_str:
        return 0.0
    if isinstance(min_str, (int, float)):
        return float(min_str)
    try:
        if ":" in str(min_str):
            parts = str(min_str).split(":")
            return float(parts[0]) + (float(parts[1]) / 60.0)
        return float(min_str)
    except Exception:
        return 0.0

def backfill_season_type(season_type: str):
    """Fetches full box score logs from API and commits non-existent data to Postgres."""
    logger.info(f"Querying stats.nba.com for 2025-26 {season_type} dataset...")
    
    try:
        raw_log = leaguegamelog.LeagueGameLog(
            season='2025-26',
            player_or_team_abbreviation='P',
            season_type_all_star=season_type
        )
        df = raw_log.get_data_frames()[0]
    except Exception as e:
        logger.error(f"Failed pulling data from NBA API endpoint: {e}")
        return

    if df.empty:
        logger.warning(f"No records returned for season type: {season_type}")
        return

    logger.info(f"Ingesting {len(df)} prospective logs. Processing unique roster mapping allocations...")
    
    with engine.begin() as conn:
        # Step 1: Ensure missing entities are registered in 'players' table
        unique_players = df[['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ABBREVIATION']].drop_duplicates()
        for _, row in unique_players.iterrows():
            pid, pname, team = str(row['PLAYER_ID']), row['PLAYER_NAME'], row['TEAM_ABBREVIATION']
            
            # Insert/update canonical player
            conn.execute(
                sqlalchemy.text("""
                    INSERT INTO players (master_player_id, full_name, team_abbrev, position)
                    VALUES (:pid, :name, :team, 'G-F')
                    ON CONFLICT (master_player_id) DO UPDATE 
                    SET team_abbrev = EXCLUDED.team_abbrev;
                """),
                {"pid": pid, "name": pname, "team": team}
            )
            
            # Insert mappings for various books (normalized names handled by resolver fallback, but explicit mappings help)
            for book in ['DraftKings', 'PrizePicks', 'Pinnacle']:
                conn.execute(
                    sqlalchemy.text("""
                        INSERT INTO player_mappings (book_name, remote_player_name, master_player_id)
                        VALUES (:book, :remote_name, :pid)
                        ON CONFLICT (book_name, remote_player_name) DO NOTHING;
                    """),
                    {"book": book, "remote_name": pname, "pid": pid}
                )
                
                # Also seed lowercased version as explicit mapping
                conn.execute(
                    sqlalchemy.text("""
                        INSERT INTO player_mappings (book_name, remote_player_name, master_player_id)
                        VALUES (:book, :remote_name, :pid)
                        ON CONFLICT (book_name, remote_player_name) DO NOTHING;
                    """),
                    {"book": book, "remote_name": pname.lower(), "pid": pid}
                )

        # Step 2: Batch process and insert clean performance records using explicit anti-duplication queries
        success_count = 0
        for _, log in df.iterrows():
            game_id = int(log['GAME_ID'])  # cast to integer format
            player_id = str(log['PLAYER_ID'])
            mp = extract_minutes(log['MIN'])
            
            # Formulating the explicit NOT EXISTS parameterization boundary
            stmt = sqlalchemy.text("""
                INSERT INTO player_game_logs (game_id, player_id, minutes_played, points, rebounds, assists, turnovers, steals, blocks, three_pointers_made)
                SELECT :game_id, :player_id, :mp, :pts, :reb, :ast, :tov, :stl, :blk, :fg3m
                WHERE NOT EXISTS (
                    SELECT 1 FROM player_game_logs 
                    WHERE game_id = :game_id AND player_id = :player_id
                );
            """)
            
            res = conn.execute(stmt, {
                "game_id": game_id,
                "player_id": player_id,
                "mp": mp,
                "pts": int(log['PTS']) if not pd.isna(log['PTS']) else 0,
                "reb": int(log['REB']) if not pd.isna(log['REB']) else 0,
                "ast": int(log['AST']) if not pd.isna(log['AST']) else 0,
                "tov": int(log['TOV']) if not pd.isna(log['TOV']) else 0,
                "stl": int(log['STL']) if not pd.isna(log['STL']) else 0,
                "blk": int(log['BLK']) if not pd.isna(log['BLK']) else 0,
                "fg3m": int(log['FG3M']) if not pd.isna(log['FG3M']) else 0
            })
            if res.rowcount > 0:
                success_count += 1
                
        logger.info(f"Committed {success_count} unique new game logs into player_game_logs.")

def main():
    # Execute full backfill sweep across regular season and playoffs
    backfill_season_type('Regular Season')
    backfill_season_type('Playoffs')
    logger.info("Database ingestion sync execution complete.")

if __name__ == "__main__":
    main()
