#!/usr/bin/env python3
import os
import sys
import logging
import pandas as pd
import numpy as np
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

def backfill_season_logs(season: str, season_type: str):
    """Fetches full box score logs from API for a specific season and upserts to Postgres."""
    logger.info(f"Querying stats.nba.com for {season} {season_type} dataset...")
    
    try:
        raw_log = leaguegamelog.LeagueGameLog(
            season=season,
            player_or_team_abbreviation='P',
            season_type_all_star=season_type
        )
        df = raw_log.get_data_frames()[0]
    except Exception as e:
        logger.error(f"Failed pulling data from NBA API endpoint for {season} {season_type}: {e}")
        return

    if df.empty:
        logger.warning(f"No records returned for {season} {season_type}")
        return

    logger.info(f"Ingesting {len(df)} prospective logs for {season} {season_type}...")
    
    with engine.begin() as conn:
        # Step 1: Ensure missing entities are registered in 'players' table
        unique_players = df[['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ABBREVIATION']].drop_duplicates()
        for _, row in unique_players.iterrows():
            pid, pname, team = str(row['PLAYER_ID']), str(row['PLAYER_NAME']), str(row['TEAM_ABBREVIATION'])
            
            conn.execute(
                sqlalchemy.text("""
                    INSERT INTO players (master_player_id, full_name, team_abbrev, position)
                    VALUES (:pid, :name, :team, 'G-F')
                    ON CONFLICT (master_player_id) DO UPDATE 
                    SET team_abbrev = EXCLUDED.team_abbrev;
                """),
                {"pid": pid, "name": pname, "team": team}
            )
            
            for book in ['DraftKings', 'PrizePicks', 'Pinnacle']:
                conn.execute(
                    sqlalchemy.text("""
                        INSERT INTO player_mappings (book_name, remote_player_name, master_player_id)
                        VALUES (:book, :remote_name, :pid)
                        ON CONFLICT (book_name, remote_player_name) DO NOTHING;
                    """),
                    {"book": book, "remote_name": pname, "pid": pid}
                )
                conn.execute(
                    sqlalchemy.text("""
                        INSERT INTO player_mappings (book_name, remote_player_name, master_player_id)
                        VALUES (:book, :remote_name, :pid)
                        ON CONFLICT (book_name, remote_player_name) DO NOTHING;
                    """),
                    {"book": book, "remote_name": pname.lower(), "pid": pid}
                )

        # Step 2: Batch upsert player_game_logs with ON CONFLICT (game_id, player_id)
        records = []
        for _, log in df.iterrows():
            game_id = int(log['GAME_ID'])
            player_id = str(log['PLAYER_ID'])
            mp = extract_minutes(log['MIN'])
            pts = int(log['PTS']) if not pd.isna(log['PTS']) else 0
            reb = int(log['REB']) if not pd.isna(log['REB']) else 0
            ast = int(log['AST']) if not pd.isna(log['AST']) else 0
            tov = int(log['TOV']) if not pd.isna(log['TOV']) else 0
            stl = int(log['STL']) if not pd.isna(log['STL']) else 0
            blk = int(log['BLK']) if not pd.isna(log['BLK']) else 0
            fgm = int(log['FGM']) if not pd.isna(log['FGM']) else 0
            fga = int(log['FGA']) if not pd.isna(log['FGA']) else 0
            fg3m = int(log['FG3M']) if not pd.isna(log['FG3M']) else 0
            ftm = int(log['FTM']) if not pd.isna(log['FTM']) else 0
            fta = int(log['FTA']) if not pd.isna(log['FTA']) else 0
            pm = int(log['PLUS_MINUS']) if not pd.isna(log['PLUS_MINUS']) else 0

            # Compute usage_rate
            usg = float(np.clip((pts + 1.5 * ast + tov) / (mp * 2.0 + 1e-5), 0.05, 0.45)) if mp > 0 else 0.05

            records.append({
                "game_id": game_id,
                "player_id": player_id,
                "mp": mp,
                "pts": pts,
                "reb": reb,
                "ast": ast,
                "tov": tov,
                "stl": stl,
                "blk": blk,
                "fgm": fgm,
                "fga": fga,
                "fg3m": fg3m,
                "ftm": ftm,
                "fta": fta,
                "pm": pm,
                "usg": usg
            })

        upsert_stmt = sqlalchemy.text("""
            INSERT INTO player_game_logs (
                game_id, player_id, minutes_played, points, rebounds, assists,
                turnovers, steals, blocks, field_goals_made, field_goals_attempted,
                three_pointers_made, free_throws_made, free_throws_attempted, plus_minus, usage_rate
            ) VALUES (
                :game_id, :player_id, :mp, :pts, :reb, :ast,
                :tov, :stl, :blk, :fgm, :fga,
                :fg3m, :ftm, :fta, :pm, :usg
            )
            ON CONFLICT (game_id, player_id) DO UPDATE SET
                minutes_played = EXCLUDED.minutes_played,
                points = EXCLUDED.points,
                rebounds = EXCLUDED.rebounds,
                assists = EXCLUDED.assists,
                turnovers = EXCLUDED.turnovers,
                steals = EXCLUDED.steals,
                blocks = EXCLUDED.blocks,
                field_goals_made = EXCLUDED.field_goals_made,
                field_goals_attempted = EXCLUDED.field_goals_attempted,
                three_pointers_made = EXCLUDED.three_pointers_made,
                free_throws_made = EXCLUDED.free_throws_made,
                free_throws_attempted = EXCLUDED.free_throws_attempted,
                plus_minus = EXCLUDED.plus_minus,
                usage_rate = EXCLUDED.usage_rate;
        """)

        batch_size = 2000
        for i in range(0, len(records), batch_size):
            conn.execute(upsert_stmt, records[i:i+batch_size])

        logger.info(f"Successfully upserted {len(records)} game logs for {season} {season_type}.")

def backfill_all_seasons(seasons: list = None):
    """Executes full backfill sweep across requested seasons for Regular Season and Playoffs."""
    if seasons is None:
        seasons = ['2023-24', '2024-25', '2025-26']

    for season in seasons:
        backfill_season_logs(season, 'Regular Season')
        backfill_season_logs(season, 'Playoffs')
    logger.info("Multi-season player game logs ingestion sync complete.")

if __name__ == "__main__":
    backfill_all_seasons()
