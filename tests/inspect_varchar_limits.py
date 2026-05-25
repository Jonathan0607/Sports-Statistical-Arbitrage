import psycopg2
import os

DB_URI = os.getenv("POSTGRES_URI", "postgresql://postgres:password@localhost:5432/sports_db")

try:
    conn = psycopg2.connect(DB_URI)
    cur = conn.cursor()
    cur.execute("""
        SELECT column_name, character_maximum_length 
        FROM information_schema.columns 
        WHERE table_name = 'trade_execution' AND data_type = 'character varying';
    """)
    res = cur.fetchall()
    print("Varchar constraints:", res)
    cur.close()
    conn.close()
except Exception as e:
    print("Error:", e)
