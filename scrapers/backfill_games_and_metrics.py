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
logger = logging.getLogger("MetricsBackfiller")

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

def backfill_games_and_fetch_df(seasons: list = None) -> pd.DataFrame:
    if seasons is None:
        seasons = ['2023-24', '2024-25', '2025-26']

    logger.info(f"Querying stats.nba.com to backfill games table for seasons: {seasons}...")
    
    # Load team abbreviations map
    with engine.connect() as conn:
        res = conn.execute(sqlalchemy.text("SELECT team_id, abbreviation FROM teams"))
        team_map = {row[1]: row[0] for row in res}

    all_dfs = []

    for season in seasons:
        logger.info(f"Fetching game logs for season: {season}...")
        for season_type in ['Regular Season', 'Playoffs']:
            try:
                raw_log = leaguegamelog.LeagueGameLog(
                    season=season,
                    player_or_team_abbreviation='P',
                    season_type_all_star=season_type
                )
                sub_df = raw_log.get_data_frames()[0]
                if not sub_df.empty:
                    sub_df['season'] = season
                    all_dfs.append(sub_df)
                    logger.info(f"Loaded {len(sub_df)} records for {season} {season_type}.")
            except Exception as e:
                logger.error(f"Failed to fetch {season} {season_type}: {e}")

    if not all_dfs:
        logger.warning("No games found to backfill.")
        return pd.DataFrame()

    df = pd.concat(all_dfs, ignore_index=True)
    unique_games = df[['GAME_ID', 'GAME_DATE', 'MATCHUP', 'season']].drop_duplicates(subset=['GAME_ID'])
    logger.info(f"Found {len(unique_games)} unique games across {seasons}.")

    success_count = 0
    with engine.begin() as conn:
        for _, row in unique_games.iterrows():
            game_id = int(row['GAME_ID'])
            matchup = row['MATCHUP']
            season_val = row['season']
            
            # Parse home/away
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
                    VALUES (:game_id, 'NBA', :home_id, :away_id, 'FINAL', :start_time, :season)
                    ON CONFLICT (game_id) DO UPDATE SET
                        season = EXCLUDED.season,
                        start_time = EXCLUDED.start_time,
                        home_team_id = EXCLUDED.home_team_id,
                        away_team_id = EXCLUDED.away_team_id;
                """),
                {
                    "game_id": game_id,
                    "home_id": home_id,
                    "away_id": away_id,
                    "start_time": pd.to_datetime(row['GAME_DATE']),
                    "season": season_val
                }
            )
            success_count += 1
            
    logger.info(f"Successfully backfilled {success_count} games into games table.")
    return df

def backfill_matchup_metrics(df: pd.DataFrame):
    if df.empty:
        logger.warning("Dataframe is empty. Skipping metrics calculation.")
        return
        
    logger.info("Computing and backfilling team matchup metrics from API logs...")
    
    with engine.begin() as conn:
        # Classify player positions based on average stats in db
        logger.info("Classifying player positions based on stats...")
        conn.execute(
            sqlalchemy.text("""
                WITH player_stats AS (
                    SELECT player_id, AVG(rebounds) as avg_reb, AVG(assists) as avg_ast, AVG(points) as avg_pts
                    FROM player_game_logs
                    GROUP BY player_id
                ), classified AS (
                    SELECT player_id,
                           CASE WHEN avg_ast > 4.5 THEN 'PG'
                                WHEN avg_reb > 7.0 THEN 'C'
                                WHEN avg_reb > 5.0 THEN 'PF'
                                WHEN avg_pts > 12.0 THEN 'SF'
                                ELSE 'SG' END as new_pos
                    FROM player_stats
                )
                UPDATE players
                SET position = classified.new_pos
                FROM classified
                WHERE players.master_player_id = classified.player_id;
            """)
        )
        
    # Load player positions from DB
    with engine.connect() as conn:
        res = conn.execute(sqlalchemy.text("SELECT master_player_id, position FROM players")).fetchall()
        player_positions = {row[0]: row[1] for row in res}
        
        # Load team abbreviations mapping
        res_teams = conn.execute(sqlalchemy.text("SELECT team_id, abbreviation FROM teams")).fetchall()
        team_map = {row[1]: row[0] for row in res_teams}
        team_id_to_abbrev = {row[0]: row[1] for row in res_teams}

    df = df.copy()
    df['PLAYER_ID'] = df['PLAYER_ID'].astype(str)
    df['position'] = df['PLAYER_ID'].map(player_positions).fillna('SG')
    
    df['MIN_val'] = df['MIN'].apply(extract_minutes)
    df['PTS_val'] = pd.to_numeric(df['PTS'], errors='coerce').fillna(0).astype(int)
    df['REB_val'] = pd.to_numeric(df['REB'], errors='coerce').fillna(0).astype(int)
    df['AST_val'] = pd.to_numeric(df['AST'], errors='coerce').fillna(0).astype(int)
    df['TOV_val'] = pd.to_numeric(df['TOV'], errors='coerce').fillna(0).astype(int)
    df['FGA_val'] = pd.to_numeric(df['FGA'], errors='coerce').fillna(0).astype(int)
    df['FTA_val'] = pd.to_numeric(df['FTA'], errors='coerce').fillna(0).astype(int)
    
    game_team_stats = {}
    grouped = df.groupby(['GAME_ID', 'TEAM_ABBREVIATION'])
    for (game_id_str, team_abbrev), group in grouped:
        game_id = int(game_id_str)
        points = group['PTS_val'].sum()
        fga = group['FGA_val'].sum()
        fta = group['FTA_val'].sum()
        tov = group['TOV_val'].sum()
        
        pts_pg = group[group['position'] == 'PG']['PTS_val'].sum()
        pts_sg = group[group['position'] == 'SG']['PTS_val'].sum()
        pts_sf = group[group['position'] == 'SF']['PTS_val'].sum()
        pts_pf = group[group['position'] == 'PF']['PTS_val'].sum()
        pts_c = group[group['position'] == 'C']['PTS_val'].sum()
        
        game_team_stats[(game_id, team_abbrev)] = {
            'points': points,
            'possessions': fga + 0.44 * fta + tov,
            'pts_pg': pts_pg,
            'pts_sg': pts_sg,
            'pts_sf': pts_sf,
            'pts_pf': pts_pf,
            'pts_c': pts_c
        }
        
    with engine.connect() as conn:
        games_query = "SELECT game_id, home_team_id, away_team_id, start_time, season FROM games ORDER BY start_time ASC"
        games_df = pd.read_sql(games_query, conn)
        
    team_allowed_history = {abbrev: [] for abbrev in team_map}
    current_season = None
    rows_to_insert = []
    
    for _, game_row in games_df.iterrows():
        game_id = int(game_row['game_id'])
        season_val = game_row.get('season')
        
        # Reset team allowed history when season boundaries cross
        if current_season is not None and season_val != current_season:
            logger.info(f"Season transition detected ({current_season} -> {season_val}). Resetting rolling matchup metrics.")
            team_allowed_history = {abbrev: [] for abbrev in team_map}
        current_season = season_val

        if pd.isna(game_row['home_team_id']) or pd.isna(game_row['away_team_id']):
            continue
        home_team_id = int(game_row['home_team_id'])
        away_team_id = int(game_row['away_team_id'])
        home_abbrev = team_id_to_abbrev.get(home_team_id)
        away_abbrev = team_id_to_abbrev.get(away_team_id)
        
        if not home_abbrev or not away_abbrev:
            continue
            
        home_stats = game_team_stats.get((game_id, home_abbrev))
        away_stats = game_team_stats.get((game_id, away_abbrev))
        
        if not home_stats or not away_stats:
            continue
            
        home_allowed = {
            'points': away_stats['points'],
            'possessions': away_stats['possessions'],
            'pts_pg': away_stats['pts_pg'],
            'pts_sg': away_stats['pts_sg'],
            'pts_sf': away_stats['pts_sf'],
            'pts_pf': away_stats['pts_pf'],
            'pts_c': away_stats['pts_c']
        }
        
        away_allowed = {
            'points': home_stats['points'],
            'possessions': home_stats['possessions'],
            'pts_pg': home_stats['pts_pg'],
            'pts_sg': home_stats['pts_sg'],
            'pts_sf': home_stats['pts_sf'],
            'pts_pf': home_stats['pts_pf'],
            'pts_c': home_stats['pts_c']
        }
        
        home_history = team_allowed_history[home_abbrev]
        if not home_history:
            rolling_def = 115.0
            rolling_pace = 99.0
            rolling_pg = 24.0
            rolling_sg = 22.0
            rolling_sf = 21.0
            rolling_pf = 22.0
            rolling_c = 23.0
        else:
            last_10 = home_history[-10:]
            rolling_def = np.mean([h['points'] for h in last_10])
            rolling_pace = np.mean([h['possessions'] for h in last_10])
            rolling_pg = np.mean([h['pts_pg'] for h in last_10])
            rolling_sg = np.mean([h['pts_sg'] for h in last_10])
            rolling_sf = np.mean([h['pts_sf'] for h in last_10])
            rolling_pf = np.mean([h['pts_pf'] for h in last_10])
            rolling_c = np.mean([h['pts_c'] for h in last_10])
            
        rows_to_insert.append({
            'team_id': int(game_row['home_team_id']),
            'game_id': game_id,
            'rolling_def_rating': float(rolling_def),
            'rolling_pace_factor': float(rolling_pace),
            'points_allowed_vs_pg': float(rolling_pg),
            'points_allowed_vs_sg': float(rolling_sg),
            'points_allowed_vs_sf': float(rolling_sf),
            'points_allowed_vs_pf': float(rolling_pf),
            'points_allowed_vs_c': float(rolling_c)
        })
        
        away_history = team_allowed_history[away_abbrev]
        if not away_history:
            rolling_def = 115.0
            rolling_pace = 99.0
            rolling_pg = 24.0
            rolling_sg = 22.0
            rolling_sf = 21.0
            rolling_pf = 22.0
            rolling_c = 23.0
        else:
            last_10 = away_history[-10:]
            rolling_def = np.mean([h['points'] for h in last_10])
            rolling_pace = np.mean([h['possessions'] for h in last_10])
            rolling_pg = np.mean([h['pts_pg'] for h in last_10])
            rolling_sg = np.mean([h['pts_sg'] for h in last_10])
            rolling_sf = np.mean([h['pts_sf'] for h in last_10])
            rolling_pf = np.mean([h['pts_pf'] for h in last_10])
            rolling_c = np.mean([h['pts_c'] for h in last_10])
            
        rows_to_insert.append({
            'team_id': int(game_row['away_team_id']),
            'game_id': game_id,
            'rolling_def_rating': float(rolling_def),
            'rolling_pace_factor': float(rolling_pace),
            'points_allowed_vs_pg': float(rolling_pg),
            'points_allowed_vs_sg': float(rolling_sg),
            'points_allowed_vs_sf': float(rolling_sf),
            'points_allowed_vs_pf': float(rolling_pf),
            'points_allowed_vs_c': float(rolling_c)
        })
        
        team_allowed_history[home_abbrev].append(home_allowed)
        team_allowed_history[away_abbrev].append(away_allowed)
        
    logger.info(f"Saving {len(rows_to_insert)} records to team_matchup_metrics table...")
    batch_size = 2000
    upsert_stmt = sqlalchemy.text("""
        INSERT INTO team_matchup_metrics (
            team_id, game_id, rolling_def_rating, rolling_pace_factor,
            points_allowed_vs_pg, points_allowed_vs_sg, points_allowed_vs_sf,
            points_allowed_vs_pf, points_allowed_vs_c
        ) VALUES (
            :team_id, :game_id, :rolling_def_rating, :rolling_pace_factor,
            :points_allowed_vs_pg, :points_allowed_vs_sg, :points_allowed_vs_sf,
            :points_allowed_vs_pf, :points_allowed_vs_c
        ) ON CONFLICT (team_id, game_id) DO UPDATE SET
            rolling_def_rating = EXCLUDED.rolling_def_rating,
            rolling_pace_factor = EXCLUDED.rolling_pace_factor,
            points_allowed_vs_pg = EXCLUDED.points_allowed_vs_pg,
            points_allowed_vs_sg = EXCLUDED.points_allowed_vs_sg,
            points_allowed_vs_sf = EXCLUDED.points_allowed_vs_sf,
            points_allowed_vs_pf = EXCLUDED.points_allowed_vs_pf,
            points_allowed_vs_c = EXCLUDED.points_allowed_vs_c;
    """)
    with engine.begin() as conn:
        for i in range(0, len(rows_to_insert), batch_size):
            conn.execute(upsert_stmt, rows_to_insert[i:i+batch_size])
            
    logger.info("Successfully backfilled team_matchup_metrics table.")

def main(seasons: list = None):
    if seasons is None:
        seasons = ['2023-24', '2024-25', '2025-26']
    df = backfill_games_and_fetch_df(seasons=seasons)
    backfill_matchup_metrics(df)
    logger.info("Backfill of games and matchup metrics completed successfully.")

if __name__ == "__main__":
    main()
