"""Centralized League Configuration for Multi-League Quantitative Engine.

Defines league-specific constants, timing, retail bet limits, and exchange mappings.
"""

LEAGUE_CONFIG = {
    "NBA": {
        "nba_api_id": "00",
        "regulation_minutes": 48.0,
        "max_retail_bet_limit": 100.00,
        "min_anchor_liquidity": 100.00,
        "prizepicks_league_name": "NBA"
    },
    "WNBA": {
        "nba_api_id": "10",
        "regulation_minutes": 40.0,
        "max_retail_bet_limit": 25.00,
        "min_anchor_liquidity": 25.00,
        "prizepicks_league_name": "WNBA"
    }
}


def get_league_config(league: str = "NBA") -> dict:
    """Safely retrieves league configuration with fallback to NBA."""
    if not league:
        return LEAGUE_CONFIG["NBA"]
    return LEAGUE_CONFIG.get(league.upper(), LEAGUE_CONFIG["NBA"])
