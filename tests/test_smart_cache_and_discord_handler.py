import pytest
import time
import logging
import json
from unittest.mock import AsyncMock, MagicMock, patch
from scrapers.hfp_poller import UnifiedRestPoller, DiscordErrorHandler

def test_state_aware_smart_cache():
    """
    Tests the State-Aware Smart Cache deduplication logic.
    - Initial alert: should fire.
    - Duplicate alert within 2 hours (same line, < 5% edge change): should NOT fire.
    - Same book & player, but line changed: should fire.
    - Same book & player, line same, but edge changed by >= 5%: should fire.
    - Cooldown window expired (> 2 hours): should fire.
    """
    poller = UnifiedRestPoller()

    # Mock the async Redis client and queue dependencies
    poller.redis = AsyncMock()
    poller.redis.hset = AsyncMock(return_value=1)
    poller.redis.expire = AsyncMock(return_value=True)
    poller.redis.rpush = AsyncMock(return_value=1)
    poller.alert_queue = MagicMock()

    async def run_test():
        if hasattr(poller, 'smart_alert_cache'):
            delattr(poller, 'smart_alert_cache')

        async def call_alert(player_name, prop_name, side, book, line, edge):
            with patch("random.lognormvariate", return_value=0.0):
                await poller._fire_alert_and_save(
                    book=book,
                    player_name=player_name,
                    player_id="123",
                    prop_name=prop_name,
                    side=side,
                    raw_line=line,
                    sharp_book="Pinnacle",
                    sharp_line=line,
                    sharp_prob=0.55,
                    target_cost=0.50,
                    target_american=100,
                    ev_edge=edge,
                    kelly_fraction=0.05,
                    zinb_mu=line,
                    zinb_pi=0.05,
                    zinb_n=10.0,
                )
            await asyncio.sleep(0)

        # 1. Initial Alert (should fire and cache)
        await call_alert("LeBron James", "Points", "Over", "pp", 25.5, 0.05)
        assert poller.alert_queue.put_nowait.call_count == 1

        # Verify cached state (key is lowercase 'points' due to _map_stat_type mapping)
        alert_key = "LeBron James_points_Over_PrizePicks"
        assert alert_key in poller.smart_alert_cache
        assert poller.smart_alert_cache[alert_key]["line"] == 25.5
        assert poller.smart_alert_cache[alert_key]["edge"] == 0.05

        poller.alert_queue.put_nowait.reset_mock()

        # 2. Duplicate Alert within cooldown with same line and minor edge change (< 5% absolute, e.g., 0.05 to 0.08)
        await call_alert("LeBron James", "Points", "Over", "pp", 25.5, 0.08)
        assert poller.alert_queue.put_nowait.call_count == 0

        # 3. Alert with line change (25.5 to 26.5)
        await call_alert("LeBron James", "Points", "Over", "pp", 26.5, 0.06)
        assert poller.alert_queue.put_nowait.call_count == 1
        poller.alert_queue.put_nowait.reset_mock()

        # 4. Alert with edge change >= 5% (0.06 to 0.12, diff = 6.0%)
        await call_alert("LeBron James", "Points", "Over", "pp", 26.5, 0.12)
        assert poller.alert_queue.put_nowait.call_count == 1
        poller.alert_queue.put_nowait.reset_mock()

        # 5. Alert outside cooldown window (set timestamp to 3 hours ago)
        poller.smart_alert_cache[alert_key]["timestamp"] = time.time() - 10800
        await call_alert("LeBron James", "Points", "Over", "pp", 26.5, 0.12)
        assert poller.alert_queue.put_nowait.call_count == 1

    import asyncio
    asyncio.run(run_test())

def test_discord_error_handler():
    """
    Tests that the DiscordErrorHandler formats and sends ERROR and CRITICAL logs to Discord.
    """
    webhook_url = "https://discord.com/api/webhooks/mock"
    handler = DiscordErrorHandler(webhook_url)
    
    # Create a record
    record_info = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="test.py",
        lineno=10,
        msg="Info message",
        args=(),
        exc_info=None
    )
    
    record_error = logging.LogRecord(
        name="test_logger",
        level=logging.ERROR,
        pathname="test.py",
        lineno=20,
        msg="Error message",
        args=(),
        exc_info=None
    )
    
    try:
        raise ValueError("Something went wrong")
    except ValueError:
        import sys
        exc_info = sys.exc_info()
        
    record_exception = logging.LogRecord(
        name="test_logger",
        level=logging.CRITICAL,
        pathname="test.py",
        lineno=30,
        msg="Exception occurred",
        args=(),
        exc_info=exc_info
    )
    
    # Patch urllib.request.urlopen to check payload and prevent actual calls
    with patch("urllib.request.urlopen") as mock_urlopen, \
         patch("threading.Thread") as mock_thread:
         
        # Emitting INFO should return immediately (level < logging.ERROR)
        handler.emit(record_info)
        mock_thread.assert_not_called()
        
        # Emitting ERROR should start a thread to send webhook
        handler.emit(record_error)
        assert mock_thread.call_count == 1
        
        # Verify thread was started
        thread_kwargs = mock_thread.call_args[1]
        assert thread_kwargs.get("daemon") is True
        
        # Let's inspect the target function of the thread
        target_fn = mock_thread.call_args[1]['target']
        
        # Now let's call the target function directly to test urlopen behavior
        mock_thread.reset_mock()
        mock_urlopen.reset_mock()
        
        # Emit exception record
        handler.emit(record_exception)
        assert mock_thread.call_count == 1
        
        # Run target function
        target_fn = mock_thread.call_args[1]['target']
        
        # We need to capture the req object passed to urlopen
        # We can run target_fn and verify it calls urlopen with a Request object
        target_fn()
        
        mock_urlopen.assert_called_once()
        req_arg = mock_urlopen.call_args[0][0]
        
        import urllib.request
        assert isinstance(req_arg, urllib.request.Request)
        assert req_arg.full_url == webhook_url
        
        # Verify payload structure
        data = json.loads(req_arg.data.decode('utf-8'))
        assert "embeds" in data
        embed = data["embeds"][0]
        assert "⚠️ System CRITICAL Detected" in embed["title"]
        assert "ValueError: Something went wrong" in embed["description"]
        assert any(f["name"] == "File" and "test.py:30" in f["value"] for f in embed["fields"])


def test_discord_alerts_suppressed_in_pytest_when_unmocked():
    """Unmocked Discord delivery must be a strict no-op under pytest / DISABLE_DISCORD_ALERTS=1."""
    import os
    from scrapers.hfp_poller import _discord_alerts_suppressed
    from src.drift_monitor import send_discord_drift_alert
    from src.execution import send_discord_webhook_sync

    assert os.getenv("DISABLE_DISCORD_ALERTS") == "1"
    assert _discord_alerts_suppressed() is True
    assert not any(type(h).__name__ == "DiscordErrorHandler" for h in logging.getLogger().handlers)

    poller = UnifiedRestPoller()
    assert poller._dispatch_webhook_sync("https://discord.com/api/webhooks/mock", {"content": "noop"}) == 0
    assert send_discord_webhook_sync("https://discord.com/api/webhooks/mock", {"content": "noop"}) == (
        0,
        "suppressed_in_tests",
    )
    assert (
        send_discord_drift_alert(
            edge_pct=-5.0,
            meta_mae=4.0,
            naive_mae=3.5,
            sample_count=50,
            webhook_url="https://discord.com/api/webhooks/mock",
        )
        is False
    )
