import psycopg2
import os

DB_URI = os.getenv("POSTGRES_URI", "postgresql://postgres:password@localhost:5432/sports_db")

try:
    conn = psycopg2.connect(DB_URI)
    cur = conn.cursor()
    cur.execute("SELECT COLUMN_NAME, DATA_TYPE FROM information_schema.columns WHERE table_name = 'trade_execution';")
    cols = cur.fetchall()
    print("trade_execution columns:", cols)
    cur.close()
    conn.close()
except Exception as e:
    print("Error:", e)
