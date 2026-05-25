import psycopg2
import os

DB_URI = os.getenv("POSTGRES_URI", "postgresql://postgres:password@localhost:5432/sports_db")

players = [
    "Pascal Siakam",
    "T.J. McConnell",
    "Shai Gilgeous-Alexander",
    "Obi Toppin",
    "Aaron Wiggins",
    "Mikal Bridges"
]

try:
    conn = psycopg2.connect(DB_URI)
    cur = conn.cursor()
    
    # Check player details
    cur.execute("""
        SELECT master_player_id, full_name 
        FROM players 
        WHERE full_name IN %s;
    """, (tuple(players),))
    found_players = cur.fetchall()
    print("Found players in players table:", found_players)
    
    # Query game logs counts for each found player
    for pid, name in found_players:
        cur.execute("""
            SELECT COUNT(*) 
            FROM player_game_logs 
            WHERE player_id = %s;
        """, (pid,))
        cnt = cur.fetchone()[0]
        print(f"Player '{name}' (ID: {pid}) has {cnt} game logs.")
        
    cur.close()
    conn.close()
except Exception as e:
    print("Error:", e)
