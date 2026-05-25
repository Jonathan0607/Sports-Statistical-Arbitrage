import psycopg2
import pandas as pd
from sqlalchemy import create_engine
import os

DB_URI = os.getenv("POSTGRES_URI", "postgresql://postgres:password@localhost:5432/sports_db")

try:
    engine = create_engine(DB_URI)
    query = """
        SELECT p.full_name, gl.points, gl.created_at
        FROM player_game_logs gl
        JOIN players p ON gl.player_id = p.master_player_id
        WHERE p.full_name IN (
            'Pascal Siakam', 'T.J. McConnell', 'Shai Gilgeous-Alexander',
            'Obi Toppin', 'Aaron Wiggins', 'Mikal Bridges'
        )
        ORDER BY gl.created_at ASC;
    """
    df = pd.read_sql(query, con=engine)
    print("Fetched logs rows:", len(df))
    if not df.empty:
        df['game_seq'] = df.groupby('full_name').cumcount()
        pivoted = df.pivot(index='game_seq', columns='full_name', values='points')
        pivoted = pivoted.dropna()
        print("Pivoted and aligned rows count:", len(pivoted))
        print("Kendall correlation matrix:\n", pivoted.corr(method='kendall'))
except Exception as e:
    print("Error:", e)
