import os
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger("BayesianPrior")

def get_default_priors() -> dict:
    """Returns reasonable fallback positional priors (per 36 minutes)."""
    return {
        'points': {'PG': 18.0, 'SG': 17.0, 'SF': 16.0, 'PF': 15.5, 'C': 16.0},
        'rebounds': {'PG': 4.0, 'SG': 4.5, 'SF': 6.0, 'PF': 8.5, 'C': 10.5},
        'assists': {'PG': 6.5, 'SG': 4.0, 'SF': 3.5, 'PF': 2.5, 'C': 2.0},
        'turnovers': {'PG': 2.5, 'SG': 2.0, 'SF': 1.8, 'PF': 1.5, 'C': 1.6}
    }

def calculate_positional_priors(box_scores_df: pd.DataFrame) -> dict:
    """
    Calculates the league-average stats per 36 minutes for each position.
    Supports both standard box score CSV headers and database-style column names.
    """
    df = box_scores_df.copy()
    
    # Normalize column names to standard internal names
    col_map = {
        'Minutes': 'minutes', 'minutes_played': 'minutes',
        'Points': 'points',
        'Rebounds': 'rebounds',
        'Assists': 'assists',
        'Turnovers': 'turnovers',
        'Position': 'position', 'position_name': 'position'
    }
    
    # Rename matching columns
    df = df.rename(columns={c: col_map[c] for c in df.columns if c in col_map})
    
    required = ['minutes', 'points', 'rebounds', 'assists', 'turnovers', 'position']
    for req in required:
        if req not in df.columns:
            df[req] = 0.0 if req != 'position' else 'PG'

    # Filter out garbage time (minutes >= 15) to keep priors clean
    core_players = df[df['minutes'] >= 15.0].copy()
    if core_players.empty:
        # Fallback to whole dataset if no players played >= 15 mins to avoid empty groupby
        core_players = df.copy()

    priors = {}
    stats = ['points', 'rebounds', 'assists', 'turnovers']
    
    for stat in stats:
        # Normalize to Per-36 to standardize volume
        col_name = f"{stat}_per_36"
        core_players[col_name] = (core_players[stat] / (core_players['minutes'] + 1e-5)) * 36
        
        # Group by position and compute the average
        grouped = core_players.groupby('position')[col_name].mean().to_dict()
        priors[stat] = {str(k).upper(): float(v) for k, v in grouped.items()}
        
    return priors

def apply_bayesian_shrinkage(player_stat_per_36: float, player_total_minutes: float, positional_prior: float, K: float = 500.0) -> float:
    """
    Shrinks a player's stat toward the positional prior based on sample size.
    K is the hyperparameter determining how quickly we trust the player's own data.
    """
    # Calculate weight (w approaches 1 as total_minutes grows)
    w = player_total_minutes / (player_total_minutes + K)
    
    # Calculate Posterior
    posterior_stat = (w * player_stat_per_36) + ((1 - w) * positional_prior)
    
    return posterior_stat

def get_positional_prior(stat_col: str, position: str, db_uri: str = None) -> float:
    """
    Retrieves the positional prior for a specific stat and position.
    Queries PostgreSQL database if db_uri is provided, otherwise falls back to defaults.
    """
    stat_clean = stat_col.lower().strip()
    # Map synonyms if needed
    if 'point' in stat_clean: stat_clean = 'points'
    elif 'rebound' in stat_clean: stat_clean = 'rebounds'
    elif 'assist' in stat_clean: stat_clean = 'assists'
    elif 'turnover' in stat_clean: stat_clean = 'turnovers'
    
    pos_clean = (position or 'SG').upper().strip()
    
    priors = None
    if db_uri:
        try:
            import psycopg2
            conn = psycopg2.connect(db_uri)
            query = """
                SELECT g.minutes_played, g.points, g.rebounds, g.assists, g.turnovers, p.position
                FROM player_game_logs g
                JOIN players p ON g.player_id = p.master_player_id
                WHERE g.minutes_played IS NOT NULL AND p.position IS NOT NULL;
            """
            df = pd.read_sql(query, conn)
            conn.close()
            if not df.empty:
                priors = calculate_positional_priors(df)
        except Exception as e:
            logger.warning(f"Failed to load positional priors from database: {e}. Using defaults.")

    if not priors:
        priors = get_default_priors()
        
    stat_priors = priors.get(stat_clean, {})
    # Fallback to general average or standard position if not found
    return stat_priors.get(pos_clean, stat_priors.get('SG', 15.0))
