import os
import time
import pandas as pd
import sqlalchemy
from dotenv import load_dotenv
from nba_api.stats.endpoints import playbyplayv2
from src.utils import get_db_engine

load_dotenv()

def backfill_play_by_play():
    engine = get_db_engine()
    
    # Get all distinct game_ids from the database
    games_query = "SELECT DISTINCT game_id FROM games ORDER BY game_id DESC"
    with engine.connect() as conn:
        all_games = pd.read_sql(games_query, conn)["game_id"].tolist()
        
    # Get all game_ids we already have pbp for
    existing_pbp_query = "SELECT DISTINCT game_id FROM play_by_play_events"
    with engine.connect() as conn:
        existing_games = pd.read_sql(existing_pbp_query, conn)["game_id"].tolist()
        
    missing_games = [g for g in all_games if g not in existing_games]
    print(f"Found {len(missing_games)} games missing play-by-play data.")
    
    for i, game_id in enumerate(missing_games):
        # The game_id in nba_api must be 10 characters zero-padded, e.g. '0022300001'
        nba_game_id = str(game_id).zfill(10)
        print(f"[{i+1}/{len(missing_games)}] Fetching PBP for game {nba_game_id}...")
        
        try:
            pbp = playbyplayv2.PlayByPlayV2(game_id=nba_game_id)
            df = pbp.get_data_frames()[0]
            
            if df.empty:
                print(f"  Warning: No data for {nba_game_id}")
                continue
                
            # Rename columns to match database schema
            rename_map = {
                'GAME_ID': 'game_id',
                'EVENTNUM': 'event_num',
                'EVENTMSGTYPE': 'event_msg_type',
                'EVENTMSGACTIONTYPE': 'event_msg_action_type',
                'PERIOD': 'period',
                'WCTIMESTRING': 'wc_timestring',
                'PCTIMESTRING': 'pctimestring',
                'HOMEDESCRIPTION': 'homedescription',
                'NEUTRALDESCRIPTION': 'neutraldescription',
                'VISITORDESCRIPTION': 'visitordescription',
                'SCORE': 'score',
                'SCOREMARGIN': 'score_margin',
                'PERSON1TYPE': 'person1type',
                'PLAYER1_ID': 'player1_id',
                'PLAYER1_NAME': 'player1_name',
                'PLAYER1_TEAM_ID': 'player1_team_id',
                'PLAYER2_ID': 'player2_id',
                'PLAYER2_NAME': 'player2_name',
                'PLAYER2_TEAM_ID': 'player2_team_id',
                'PLAYER3_ID': 'player3_id',
                'PLAYER3_NAME': 'player3_name',
                'PLAYER3_TEAM_ID': 'player3_team_id'
            }
            df = df.rename(columns=rename_map)
            df = df[[c for c in rename_map.values() if c in df.columns]]
            
            # Cast game_id to match our database varchar(20) format
            df['game_id'] = df['game_id'].astype(str)
            
            # Write to database
            with engine.begin() as conn:
                df.to_sql('play_by_play_events', conn, if_exists='append', index=False)
            
            print(f"  Inserted {len(df)} events.")
            time.sleep(0.5)  # Rate limiting
            
        except Exception as e:
            print(f"  Error fetching game {nba_game_id}: {e}")
            time.sleep(2)  # Longer wait on error

if __name__ == "__main__":
    backfill_play_by_play()
