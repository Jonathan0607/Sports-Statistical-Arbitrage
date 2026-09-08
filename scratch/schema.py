import pandas as pd
import sqlalchemy
engine = sqlalchemy.create_engine("postgresql+psycopg2://postgres:password@localhost:5432/sports_db")
with engine.connect() as conn:
    print(pd.read_sql("SELECT column_name FROM information_schema.columns WHERE table_name = 'player_game_logs'", conn))
    print(pd.read_sql("SELECT column_name FROM information_schema.columns WHERE table_name = 'player_tracking_logs'", conn))
    print(pd.read_sql("SELECT column_name FROM information_schema.columns WHERE table_name = 'team_matchup_metrics'", conn))
