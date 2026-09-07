import os
import sys
import psycopg2
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

def main():
    db_uri = os.getenv("POSTGRES_URI", "postgresql://postgres:password@localhost:5432/sports_db")
    print(f"Connecting to database at {db_uri}...")
    try:
        conn = psycopg2.connect(db_uri)
        cursor = conn.cursor()
        
        # Search for McBride
        cursor.execute("SELECT master_player_id, full_name FROM players WHERE full_name ILIKE '%mcbride%';")
        rows = cursor.fetchall()
        print(f"Search results for 'mcbride': {rows}")
        
        if rows:
            master_player_id = rows[0][0]
            names_to_map = [
                "Deuce McBride",
                "Miles McBride",
                "deucemcbride",
                "milesmcbride",
                "deuce.mcbride",
                "m.mcbride"
            ]
            books = ["PrizePicks", "Underdog", "Kalshi", "Polymarket", "DraftKings", "Pinnacle"]
            
            inserted = 0
            for book in books:
                for remote_name in names_to_map:
                    cursor.execute("""
                        INSERT INTO player_mappings (master_player_id, book_name, remote_player_name)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (book_name, remote_player_name) DO NOTHING;
                    """, (master_player_id, book, remote_name))
                    inserted += cursor.rowcount
                    
            conn.commit()
            print(f"Successfully mapped {rows[0][1]} variations! Rows inserted: {inserted}")
            
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
