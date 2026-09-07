import os
import sqlite3
import pytest
import asyncio
from datetime import datetime, timezone
from src.utils import MarketDataLogger

TEST_DB_PATH = "logs/test_market_data.db"

@pytest.fixture
def clean_db():
    # Remove database if exists before test
    if os.path.exists(TEST_DB_PATH):
        try:
            os.remove(TEST_DB_PATH)
        except OSError:
            pass

    yield

    # Cleanup after test
    if os.path.exists(TEST_DB_PATH):
        try:
            os.remove(TEST_DB_PATH)
        except OSError:
            pass

@pytest.mark.anyio
async def test_market_logger_init(clean_db):
    logger = MarketDataLogger(db_path=TEST_DB_PATH)
    assert os.path.exists(TEST_DB_PATH)

    conn = sqlite3.connect(TEST_DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(board_snapshots)")
        columns = cursor.fetchall()
        # Columns format: (cid, name, type, notnull, dflt_value, pk)
        col_names = [col[1] for col in columns]
        col_types = [col[2] for col in columns]
        
        expected_cols = ["Timestamp", "Book", "Player", "Stat", "Line", "Over_Prob", "Under_Prob", "Status"]
        assert col_names == expected_cols
        assert "DATETIME" in col_types[0] or "TEXT" in col_types[0] or col_types[0] == "DATETIME"
    finally:
        conn.close()

@pytest.mark.anyio
async def test_log_snapshot_inserts(clean_db):
    logger = MarketDataLogger(db_path=TEST_DB_PATH)

    # Sample data
    prize_picks_data = {
        "proj_1": {
            "player": "LeBron James",
            "prop": "Points",
            "line": 25.5,
            "odds_type": "standard",
            "start_time": "2026-05-26T20:00:00Z"
        },
        "proj_2": {
            "player": "Anthony Davis",
            "prop": "Rebounds",
            "line": 12.5,
            "odds_type": "standard",
            "start_time": "2026-05-26T20:00:00Z"
        }
    }

    pinnacle_data = {
        "lebronjames": {
            "points": {
                "25.5": {
                    "pinnacle_true_prob": 0.525,
                    "pinnacle_line": 25.5,
                    "dk_line": 25.5
                }
            }
        }
    }

    # Log snapshot asynchronously
    await logger.log_snapshot(prize_picks_data, pinnacle_data)

    # Verify insertions
    conn = sqlite3.connect(TEST_DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM board_snapshots")
        rows = cursor.fetchall()
        
        assert len(rows) == 3 # 2 PrizePicks + 1 Pinnacle
        
        # Verify PrizePicks row
        pp_rows = [r for r in rows if r[1] == "PrizePicks"]
        assert len(pp_rows) == 2
        players = [r[2] for r in pp_rows]
        assert "LeBron James" in players
        assert "Anthony Davis" in players
        
        # Verify Pinnacle row
        pin_rows = [r for r in rows if r[1] == "Pinnacle"]
        assert len(pin_rows) == 1
        assert pin_rows[0][2] == "lebronjames"
        assert pin_rows[0][3] == "points"
        assert pin_rows[0][4] == 25.5
        assert abs(pin_rows[0][5] - 0.525) < 1e-6
        assert abs(pin_rows[0][6] - 0.475) < 1e-6
        assert pin_rows[0][7] == "ACTIVE"
    finally:
        conn.close()

@pytest.mark.anyio
async def test_log_snapshot_empty(clean_db):
    logger = MarketDataLogger(db_path=TEST_DB_PATH)
    
    # Try logging empty data
    await logger.log_snapshot({}, {})
    
    conn = sqlite3.connect(TEST_DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM board_snapshots")
        count = cursor.fetchone()[0]
        assert count == 0
    finally:
        conn.close()
