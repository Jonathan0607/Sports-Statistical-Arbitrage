import pytest
import json
from src.execution.sgp_worker import GlobalExposureTracker, SlipQueueManager
from infrastructure.entity_resolver import clean_player_name
from infrastructure.redis_client import RedisClient

@pytest.fixture
def redis_conn():
    return RedisClient().get_client()

def test_clean_player_name():
    assert clean_player_name("Luka Dončić") == "luka doncic"
    assert clean_player_name("T.J. McConnell") == "tj mcconnell"
    assert clean_player_name("Robert Williams III") == "robert williams"
    assert clean_player_name("J.R. Smith Jr.") == "smith"

def test_exposure_tracker_keys():
    tracker = GlobalExposureTracker()
    date_str = tracker.get_current_date_est()
    key = tracker._get_key("Pascal Siakam", date_str)
    assert key == f"exposure:player:pascal siakam:{date_str}"

def test_exposure_tracker_scaling(redis_conn):
    tracker = GlobalExposureTracker(max_exposure_per_player=500.00)
    test_date = "2026-99-99"
    player1 = "Test Player One"
    player2 = "Test Player Two"
    
    key1 = tracker._get_key(player1, test_date)
    key2 = tracker._get_key(player2, test_date)
    redis_conn.delete(key1, key2)
    
    try:
        # Initial check when exposure is zeroed
        stake, allowed = tracker.check_and_scale_stake([player1, player2], 100.0, date_str=test_date)
        assert allowed is True
        assert stake == 100.0
        
        # Commit partial exposure to player 1
        tracker.commit_exposure([player1], 300.0, date_str=test_date)
        
        # Check scale down:
        # Player 1 exposure: $300.00 (remaining: $200.00)
        # Player 2 exposure: $0.00 (remaining: $500.00)
        # Recommended: $300.00 -> should scale to min(300, 200, 500) = $200.00
        stake, allowed = tracker.check_and_scale_stake([player1, player2], 300.0, date_str=test_date)
        assert allowed is True
        assert stake == 200.0
        
        # Commit partial exposure to player 2
        tracker.commit_exposure([player2], 350.0, date_str=test_date)
        
        # Recommended: $250.00:
        # Player 1 remaining: $200.00
        # Player 2 remaining: $150.00
        # Should scale to min(250, 200, 150) = $150.00
        stake, allowed = tracker.check_and_scale_stake([player1, player2], 250.0, date_str=test_date)
        assert allowed is True
        assert stake == 150.0
        
        # Hit cap on player 2
        tracker.commit_exposure([player2], 150.0, date_str=test_date)
        
        # Try checking stake -> should reject completely since Player 2 is at cap
        stake, allowed = tracker.check_and_scale_stake([player1, player2], 50.0, date_str=test_date)
        assert allowed is False
        assert stake == 0.0
        
    finally:
        redis_conn.delete(key1, key2)

def test_slip_queue_manager(redis_conn):
    queue_name = "execution_queue:prizepicks_test_suite"
    redis_conn.delete(queue_name)
    
    try:
        manager = SlipQueueManager(queue_name=queue_name)
        leg1 = {"player": "Player X", "stat": "points", "side": "Over", "line": 20.5}
        leg2 = {"player": "Player Y", "stat": "assists", "side": "Under", "line": 5.5}
        
        payload = manager.formulate_execution_payload(leg1, leg2, 75.0, 0.05)
        
        assert "slip_id" in payload
        assert "timestamp" in payload
        assert payload["risk_adjusted_stake"] == 75.0
        assert payload["ev_edge"] == 0.05
        assert len(payload["legs"]) == 2
        assert payload["legs"][0]["player"] == "Player X"
        assert payload["legs"][1]["side"] == "Under"
        
        # Enqueue
        success = manager.enqueue_slip(payload)
        assert success is True
        
        # Verify Redis LPUSH
        length = redis_conn.llen(queue_name)
        assert length == 1
        
        raw_slip = redis_conn.rpop(queue_name)
        data = json.loads(raw_slip)
        assert data["slip_id"] == payload["slip_id"]
        assert data["risk_adjusted_stake"] == 75.0
    finally:
        redis_conn.delete(queue_name)
