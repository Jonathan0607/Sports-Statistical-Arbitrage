import sqlite3

db_path = "logs/market_data.db"
print(f"Connecting to SQLite database: {db_path}...")
try:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cur.fetchall()
    print("Tables found in SQLite database:")
    for t in tables:
        print(f"  - {t[0]}")
        # Count rows
        cur.execute(f"SELECT COUNT(*) FROM {t[0]};")
        cnt = cur.fetchone()[0]
        print(f"    Count: {cnt} rows")
        # Columns
        cur.execute(f"PRAGMA table_info({t[0]});")
        cols = cur.fetchall()
        print("    Columns:")
        for c in cols:
            print(f"      {c[1]} ({c[2]})")
            
        # Sample row
        if cnt > 0:
            cur.execute(f"SELECT * FROM {t[0]} LIMIT 1;")
            print("    Sample row:", cur.fetchone())
    conn.close()
except Exception as e:
    print(f"Error checking SQLite: {e}")
