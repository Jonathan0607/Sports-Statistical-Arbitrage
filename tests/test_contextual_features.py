import os
import sys
import pandas as pd
import numpy as np
import sqlalchemy

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

# Load env variables
load_dotenv()

from src.features import ContextualFeatureEngine

def test_contextual_feature_engine():
    print("Testing ContextualFeatureEngine...")
    
    # Instantiate engine
    fe = ContextualFeatureEngine()
    
    # Fetch a player ID from the players table who actually has game logs
    query_player = """
        SELECT DISTINCT g.player_id, p.full_name, p.current_team_id, p.position 
        FROM player_game_logs g
        JOIN players p ON g.player_id = p.master_player_id
        LIMIT 5;
    """
    with fe.engine.connect() as conn:
        players_df = pd.read_sql(sqlalchemy.text(query_player), conn)
        
    if players_df.empty:
        print("Warning: players table is empty. Cannot test feature engineering with real data.")
        return
        
    print("Found players in database:")
    print(players_df)
    
    for _, row in players_df.iterrows():
        player_id = row['player_id']
        name = row['full_name']
        print(f"\nBuilding feature matrix for player: {name} (ID: {player_id})")
        
        # Get history from DB
        query_history = """
            SELECT g.game_id, gm.start_time, gm.home_team_id, gm.away_team_id,
                   g.minutes_played, g.points, g.rebounds, g.assists, g.turnovers,
                   g.steals, g.blocks, g.three_pointers_made, g.usage_rate
            FROM player_game_logs g
            JOIN games gm ON g.game_id = gm.game_id
            WHERE g.player_id = :player_id
            ORDER BY gm.start_time ASC;
        """
        with fe.engine.connect() as conn:
            history_df = pd.read_sql(sqlalchemy.text(query_history), conn, params={"player_id": player_id})
            
        print(f"History records found: {len(history_df)}")
        if history_df.empty:
            continue
            
        feature_df = fe.build_player_feature_matrix(player_id, history_df)
        print("Feature matrix columns:", feature_df.columns.tolist())
        print(feature_df[['days_rest', 'is_back_to_back', 'opp_def_rating', 'opp_pace_factor', 'opp_pos_pts_allowed', 'vacated_usage']].head())
        
        # Verify columns exist
        expected_cols = ['days_rest', 'is_back_to_back', 'opp_def_rating', 'opp_pace_factor', 'opp_pos_pts_allowed', 'vacated_usage']
        for col in expected_cols:
            assert col in feature_df.columns, f"Missing feature column: {col}"
            
        # Test active features
        active_feats = fe.build_active_features(player_id)
        print(f"Active Features: {active_feats}")
        for col in expected_cols:
            assert col in active_feats, f"Missing active feature: {col}"
            
        print(f"✅ Contextual features test PASSED for player {name}.")
        break

if __name__ == "__main__":
    test_contextual_feature_engine()
