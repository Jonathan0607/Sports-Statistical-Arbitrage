import os
import sys
import sqlalchemy
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

# Ensure root directory is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scrapers.backfill_games_and_metrics import backfill_games_and_fetch_df, extract_minutes

engine = sqlalchemy.create_engine(os.getenv("POSTGRES_URI"))

df = backfill_games_and_fetch_df()
print("DF length:", len(df))

# Check team abbreviations
with engine.connect() as conn:
    res = conn.execute(sqlalchemy.text("SELECT team_id, abbreviation FROM teams"))
    team_map = {row[1]: row[0] for row in res}
    team_id_to_abbrev = {row[0]: row[1] for row in res}

print("Team map:", team_map)

# Group by GAME_ID and TEAM_ABBREVIATION
game_team_stats = {}
grouped = df.groupby(['GAME_ID', 'TEAM_ABBREVIATION'])
for (game_id_str, team_abbrev), group in grouped:
    game_team_stats[(int(game_id_str), team_abbrev)] = True

print("game_team_stats size:", len(game_team_stats))

# Read games from database
with engine.connect() as conn:
    games_df = pd.read_sql("SELECT game_id, home_team_id, away_team_id FROM games ORDER BY start_time ASC", conn)

print("Games in database:", len(games_df))

stats_found = 0
for idx, game_row in games_df.iterrows():
    game_id = int(game_row['game_id'])
    home_abbrev = team_id_to_abbrev.get(int(game_row['home_team_id']))
    away_abbrev = team_id_to_abbrev.get(int(game_row['away_team_id']))
    
    home_exists = (game_id, home_abbrev) in game_team_stats
    away_exists = (game_id, away_abbrev) in game_team_stats
    
    if home_exists and away_exists:
        stats_found += 1
    
    if idx < 5:
        print(f"Game {game_id}: home={home_abbrev} (exists={home_exists}), away={away_abbrev} (exists={away_exists})")

print(f"Stats found for {stats_found} / {len(games_df)} games")
