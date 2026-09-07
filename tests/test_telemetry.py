import pytest
import os
import csv
from datetime import datetime, timezone, timedelta
from src.utils import WalkForwardLogger, check_clv_job

@pytest.fixture(autouse=True)
def clean_telemetry_csv():
    """Fixture that cleans telemetry CSV before/after tests."""
    original_path = WalkForwardLogger.CSV_PATH
    temp_path = "logs/test_telemetry.csv"
    WalkForwardLogger.CSV_PATH = temp_path
    
    # Remove file if exists
    if os.path.exists(temp_path):
        os.remove(temp_path)
        
    yield
    
    # Cleanup after test
    if os.path.exists(temp_path):
        os.remove(temp_path)
    # Restore original path
    WalkForwardLogger.CSV_PATH = original_path

def test_telemetry_logger_flow():
    """Validates that WalkForwardLogger successfully creates, updates, and structures rows."""
    timestamp = datetime.now(timezone.utc).isoformat()
    game_time = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()
    
    # 1. Log a signal
    sig_id = WalkForwardLogger.log_signal(
        player="Donovan Mitchell",
        stat="Points",
        line=28.5,
        zinb_prob=0.582,
        edge=0.071,
        timestamp=timestamp,
        game_time=game_time
    )
    
    assert os.path.exists(WalkForwardLogger.CSV_PATH)
    
    # Verify rows in CSV
    with open(WalkForwardLogger.CSV_PATH, 'r') as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 1
        assert reader[0]["Signal_ID"] == sig_id
        assert reader[0]["Player"] == "Donovan Mitchell"
        assert reader[0]["Status"] == "PENDING"
        assert reader[0]["CLV"] == ""
        
    # 2. Update status
    WalkForwardLogger.update_status(
        player="Donovan Mitchell",
        stat="Points",
        line=28.5,
        timestamp=timestamp,
        status="EXECUTABLE"
    )
    
    with open(WalkForwardLogger.CSV_PATH, 'r') as f:
        reader = list(csv.DictReader(f))
        assert reader[0]["Status"] == "EXECUTABLE"
        
    # 3. Update CLV
    WalkForwardLogger.update_clv(sig_id, 0.085)
    
    with open(WalkForwardLogger.CSV_PATH, 'r') as f:
        reader = list(csv.DictReader(f))
        assert reader[0]["CLV"] == "0.085000"
