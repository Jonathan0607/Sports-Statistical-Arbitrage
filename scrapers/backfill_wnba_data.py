#!/usr/bin/env python3
import os
import sys
import logging
import pandas as pd
import numpy as np
import sqlalchemy
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("WNBABackfiller")

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

nba_http.requests = MockRequests

DB_URI = os.getenv("POSTGRES_URI", "postgresql+psycopg2://postgres:postgres@localhost:5432/sports_db")
if DB_URI.startswith("postgres://"):
    DB_URI = DB_URI.replace("postgres://", "postgresql+psycopg2://", 1)
engine = sqlalchemy.create_engine(DB_URI)

def extract_minutes(min_str) -> float:
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

def backfill_wnba():
    seasons = ['2021', '2022', '2023', '2024']
    all_dfs = []

    for season in seasons:
        for season_type in ['Regular Season', 'Playoffs']:
            try:
                raw_log = leaguegamelog.LeagueGameLog(
                    season=season,
                    league_id='10',
                    player_or_team_abbreviation='P',
                    season_type_all_star=season_type
                )
                df = raw_log.get_data_frames()[0]
                if not df.empty:
                    df['season'] = season
                    all_dfs.append(df)
                    logger.info(f"Loaded {len(df)} records for {season} {season_type}.")
            except Exception as e:
                logger.error(f"Failed fetching {season} {season_type}: {e}")

    if not all_dfs:
        logger.error("No WNBA data fetched.")
        return

    df = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Total WNBA records: {len(df)}")

    with engine.begin() as conn:
        # 1. Upsert Teams
        unique_teams = df[['TEAM_ID', 'TEAM_ABBREVIATION', 'TEAM_NAME']].drop_duplicates()
        for _, row in unique_teams.iterrows():
            conn.execute(
                sqlalchemy.text("""
                    INSERT INTO teams (team_id, sport, abbreviation, full_name)
                    VALUES (:tid, 'WNBA', :abbr, :name)
                    ON CONFLICT (team_id) DO NOTHING;
                """),
                {"tid": int(row['TEAM_ID']), "abbr": str(row['TEAM_ABBREVIATION']), "name": str(row['TEAM_NAME'])}
            )
        logger.info(f"Inserted {len(unique_teams)} WNBA teams.")

        # 2. Upsert Players
        unique_players = df[['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ABBREVIATION']].drop_duplicates(subset=['PLAYER_ID'])
        for _, row in unique_players.iterrows():
            pid = str(row['PLAYER_ID'])
            conn.execute(
                sqlalchemy.text("""
                    INSERT INTO players (master_player_id, full_name, team_abbrev, position, sport)
                    VALUES (:pid, :name, :team, 'G-F', 'WNBA')
                    ON CONFLICT (master_player_id) DO UPDATE 
                    SET team_abbrev = EXCLUDED.team_abbrev;
                """),
                {"pid": pid, "name": str(row['PLAYER_NAME']), "team": str(row['TEAM_ABBREVIATION'])}
            )
        logger.info(f"Inserted {len(unique_players)} WNBA players.")

        # 3. Upsert Games
        unique_games = df[['GAME_ID', 'GAME_DATE', 'MATCHUP', 'season']].drop_duplicates(subset=['GAME_ID'])
        
        # Build team ID map
        res = conn.execute(sqlalchemy.text("SELECT team_id, abbreviation FROM teams WHERE sport='WNBA'"))
        team_map = {row[1]: row[0] for row in res}

        for _, row in unique_games.iterrows():
            matchup = str(row['MATCHUP'])
            if '@' in matchup:
                teams = matchup.split(' @ ')
                home_team, away_team = teams[1], teams[0]
            elif 'vs.' in matchup:
                teams = matchup.split(' vs. ')
                home_team, away_team = teams[0], teams[1]
            else:
                continue
                
            home_id = team_map.get(home_team)
            away_id = team_map.get(away_team)
            
            if not home_id or not away_id:
                continue

            conn.execute(
                sqlalchemy.text("""
                    INSERT INTO games (game_id, sport, home_team_id, away_team_id, game_status, start_time, season)
                    VALUES (:gid, 'WNBA', :hid, :aid, 'FINAL', :stime, :season)
                    ON CONFLICT (game_id) DO UPDATE SET
                        season = EXCLUDED.season, start_time = EXCLUDED.start_time;
                """),
                {"gid": int(row['GAME_ID']), "hid": home_id, "aid": away_id, 
                 "stime": pd.to_datetime(row['GAME_DATE']), "season": row['season']}
            )
        logger.info(f"Inserted {len(unique_games)} WNBA games.")

        # 4. Upsert Player Game Logs
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

            usg = float(np.clip((pts + 1.5 * ast + tov) / (mp * 2.0 + 1e-5), 0.05, 0.45)) if mp > 0 else 0.05

            conn.execute(
                sqlalchemy.text("""
                    INSERT INTO player_game_logs (
                        game_id, player_id, minutes_played, points, rebounds, assists,
                        steals, blocks, turnovers, field_goals_made, field_goals_attempted,
                        three_pointers_made, free_throws_made, free_throws_attempted,
                        plus_minus, usage_rate
                    ) VALUES (
                        :game_id, :player_id, :mp, :pts, :reb, :ast,
                        :stl, :blk, :tov, :fgm, :fga,
                        :fg3m, :ftm, :fta,
                        :pm, :usg
                    ) ON CONFLICT (game_id, player_id) DO UPDATE SET
                        minutes_played = EXCLUDED.minutes_played,
                        points = EXCLUDED.points, rebounds = EXCLUDED.rebounds,
                        assists = EXCLUDED.assists, usage_rate = EXCLUDED.usage_rate;
                """),
                {"game_id": game_id, "player_id": player_id, "mp": mp, "pts": pts, "reb": reb, 
                 "ast": ast, "stl": stl, "blk": blk, "tov": tov, "fgm": fgm, "fga": fga, 
                 "fg3m": fg3m, "ftm": ftm, "fta": fta, "pm": pm, "usg": usg}
            )
        logger.info("Inserted WNBA player game logs.")
        
    # Generate rolling pace and defense ratings
    from scrapers.backfill_games_and_metrics import backfill_matchup_metrics
    backfill_matchup_metrics(df)

if __name__ == "__main__":
    backfill_wnba()
