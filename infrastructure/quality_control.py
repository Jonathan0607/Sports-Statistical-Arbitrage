import logging
from datetime import datetime, timezone

logger = logging.getLogger("QualityControl")

class DataQualityAuditor:
    @staticmethod
    def validate_timestamp(incoming_ts: str) -> bool:
        """
        Ensures incoming tick timestamps are strictly localized to UTC and 
        rejects any data from the future (look-ahead bias prevention).
        """
        try:
            # Assume incoming_ts is an ISO8601 string or Unix epoch
            if isinstance(incoming_ts, (int, float)) or incoming_ts.isdigit():
                dt = datetime.fromtimestamp(int(incoming_ts)/1000, tz=timezone.utc)
            else:
                dt = datetime.fromisoformat(incoming_ts.replace("Z", "+00:00"))
                
            now = datetime.now(timezone.utc)
            
            if dt > now:
                logger.error(f"[LOOK-AHEAD BIAS] Rejected future timestamp: {dt}")
                return False
            return True
        except Exception as e:
            logger.error(f"[FORMAT ERROR] Invalid timestamp payload: {incoming_ts} - {e}")
            return False

    @staticmethod
    def validate_box_score(box_score: dict) -> bool:
        """
        Validates structural integrity of historical data.
        Drops logs with missing critical fields or impossible values.
        """
        required_keys = ["player_id", "pts", "min"]
        
        # Check for missing keys
        if not all(key in box_score for key in required_keys):
            logger.warning(f"[DATA QUALITY DROP] Missing required keys in log: {box_score.get('game_id', 'Unknown')}")
            return False
            
        # Check for nulls or impossible values
        try:
            minutes = float(box_score["min"])
            if minutes < 0 or minutes > 60: # Max NBA game with multi-OT is ~60 mins
                logger.warning(f"[DATA QUALITY DROP] Impossible minutes value ({minutes}) for player {box_score['player_id']}")
                return False
        except (ValueError, TypeError):
            logger.warning(f"[DATA QUALITY DROP] Non-numeric minutes value for player {box_score.get('player_id')}")
            return False
            
        return True

if __name__ == "__main__":
    import time
    logging.basicConfig(level=logging.INFO)
    auditor = DataQualityAuditor()
    
    # Test 1: Valid current timestamp (ms epoch)
    current_ms = str(int(time.time() * 1000))
    print(f"Valid timestamp ({current_ms}): {auditor.validate_timestamp(current_ms)}")
    
    # Test 2: Future timestamp (should be rejected)
    future_ms = str(int(time.time() * 1000) + 86400000)  # +24 hours
    print(f"Future timestamp ({future_ms}): {auditor.validate_timestamp(future_ms)}")
    
    # Test 3: ISO8601 string
    iso_ts = "2024-11-15T20:30:00Z"
    print(f"ISO timestamp ({iso_ts}): {auditor.validate_timestamp(iso_ts)}")
