import psycopg2
import uuid
import datetime
from datetime import timezone
import os

DB_URI = os.getenv("POSTGRES_URI", "postgresql://postgres:password@localhost:5432/sports_db")

def seed_historical_trades():
    players = ["LeBron James", "Luka Doncic", "Nikola Jokic", "P.J. Washington", "Anthony Davis",
               "Kyrie Irving", "Kevin Durant", "Stephen Curry", "Jayson Tatum", "Jaylen Brown"] * 2
    
    # American odds corresponding to the decimal odds in the mock data
    american_odds = [
        110, -105, 105, -102, 115, -111, 102, -104, 108, -108,
        112, -106, 106, -103, 116, -114, 104, -105, 109, -109
    ]
    
    closing_decimal_odds = [
        2.01, 1.88, 1.96, 1.91, 2.04, 1.83, 1.94, 1.89, 1.99, 1.86,
        2.03, 1.87, 1.97, 1.90, 2.05, 1.81, 1.96, 1.88, 2.00, 1.85
    ]
    
    try:
        conn = psycopg2.connect(DB_URI)
        cur = conn.cursor()
        
        # Check if table is empty
        cur.execute("SELECT COUNT(*) FROM trade_execution;")
        cnt = cur.fetchone()[0]
        if cnt > 0:
            print("trade_execution table already has data. Count:", cnt)
            cur.close()
            conn.close()
            return
            
        print("Seeding 20 trades...")
        for i in range(20):
            tid = str(uuid.uuid4())
            now_ts = datetime.datetime.now(timezone.utc)
            
            cur.execute("""
                INSERT INTO trade_execution (
                    trade_id, timestamp, player_id, player_name, bookmaker, 
                    bet_type, line_value, odds, stake_amount, expected_value, 
                    status, settlement_outcome, pnl, closing_line_value, settled_at
                ) VALUES (
                    %s, %s, %s, %s, %s, 
                    %s, %s, %s, %s, %s, 
                    %s, %s, %s, %s, %s
                );
            """, (
                tid, now_ts, f"PID-{1000+i}", players[i], "PrizePicks" if i % 2 == 0 else "DraftKings",
                "OVER", 24.5, american_odds[i], 100.0, 0.05,
                "SETTLED", "WIN" if i % 2 == 0 else "LOSS", 110.0 if i % 2 == 0 else -100.0,
                closing_decimal_odds[i], now_ts
            ))
            
        conn.commit()
        print("Successfully seeded 20 trades.")
        cur.close()
        conn.close()
    except Exception as e:
        print("Seeding error:", e)

if __name__ == "__main__":
    seed_historical_trades()
