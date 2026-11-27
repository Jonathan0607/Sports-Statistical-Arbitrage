"""Global pytest safety fixtures preventing live external webhook leaks during tests."""

from __future__ import annotations

import logging
import os

import pytest

# Ensure alert suppression is active before test modules import app/poller modules.
os.environ.setdefault("DISABLE_DISCORD_ALERTS", "1")


def _is_discord_handler(handler: logging.Handler) -> bool:
    cls_name = type(handler).__name__
    return cls_name in ("DiscordErrorHandler", "DiscordWebhookHandler") or hasattr(
        handler, "webhook_url"
    )


def _strip_discord_handlers() -> None:
    root = logging.getLogger()
    root.handlers[:] = [h for h in root.handlers if not _is_discord_handler(h)]
    for logger_obj in logging.Logger.manager.loggerDict.values():
        if isinstance(logger_obj, logging.Logger):
            logger_obj.handlers[:] = [
                h for h in logger_obj.handlers if not _is_discord_handler(h)
            ]


@pytest.fixture(autouse=True, scope="session")
def disable_live_discord_alerts():
    """Sets DISABLE_DISCORD_ALERTS=1 and strips attached Discord handlers from loggers."""
    prev = os.environ.get("DISABLE_DISCORD_ALERTS")
    os.environ["DISABLE_DISCORD_ALERTS"] = "1"
    _strip_discord_handlers()
    try:
        yield
    finally:
        _strip_discord_handlers()
        if prev is None:
            os.environ.pop("DISABLE_DISCORD_ALERTS", None)
        else:
            os.environ["DISABLE_DISCORD_ALERTS"] = prev
