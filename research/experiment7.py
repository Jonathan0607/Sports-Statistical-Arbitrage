import os
import sys

# Script to inject the new features into src/features.py and src/train_global_model.py

with open("src/features.py", "r") as f:
    features_code = f.read()

# 1. Update SQL Query
query_replacement = """                SELECT g.game_id, gm.start_time, gm.season, gm.home_team_id, gm.away_team_id,
                       g.minutes_played, g.points, g.rebounds, g.assists, g.turnovers,
                       g.steals, g.blocks, g.three_pointers_made, g.usage_rate,
                       g.field_goals_attempted, g.free_throws_attempted, g.plus_minus,
                       COALESCE(t.potential_assists, 0.0) AS potential_assists,
                       COALESCE(t.touches, 0.0) AS touches,
                       COALESCE(t.passes_made, 0.0) AS passes_made,
                       COALESCE(t.time_of_possession_sec, 0.0) AS time_of_possession_sec,
                       COALESCE(t.avg_defender_distance_ft, 4.5) AS avg_defender_distance_ft,
                       COALESCE(t.shorthanded_points, 0.0) AS shorthanded_points"""
                       
features_code = features_code.replace("""                SELECT g.game_id, gm.start_time, gm.season, gm.home_team_id, gm.away_team_id,
                       g.minutes_played, g.points, g.rebounds, g.assists, g.turnovers,
                       g.steals, g.blocks, g.three_pointers_made, g.usage_rate,
                       COALESCE(t.potential_assists, 0.0) AS potential_assists,
                       COALESCE(t.touches, 0.0) AS touches,
                       COALESCE(t.passes_made, 0.0) AS passes_made,
                       COALESCE(t.time_of_possession_sec, 0.0) AS time_of_possession_sec,
                       COALESCE(t.avg_defender_distance_ft, 4.5) AS avg_defender_distance_ft""", query_replacement)


# 2. Add rolling features computation
rolling_features_replacement = """        for col, default in [('potential_assists', 0.0), ('touches', 0.0), ('passes_made', 0.0), ('time_of_possession_sec', 0.0), ('avg_defender_distance_ft', 4.5), ('shorthanded_points', 0.0), ('plus_minus', 0.0), ('field_goals_attempted', 0.0), ('free_throws_attempted', 0.0)]:
            if col not in df.columns:
                df[col] = default
            else:
                df[col] = df[col].fillna(default)

        # Advanced derived columns before rolling
        df['ts_pct'] = df['points'] / (2 * (df['field_goals_attempted'] + 0.44 * df['free_throws_attempted']) + 1e-5)
        df['reb_rate'] = (df['rebounds'] / (df['minutes_played'] + 1e-5)) * 36.0
        df['ast_rate'] = (df['assists'] / (df['minutes_played'] + 1e-5)) * 36.0
        df['usg_efficiency'] = df['points'] / ((df['usage_rate'] * df['minutes_played']) + 1e-5)

        df['roll_potential_assists'] = df.groupby('season')['potential_assists'].transform(lambda s: s.shift(1).ewm(alpha=0.3, adjust=False).mean()).fillna(0.0)
        df['roll_touches'] = df.groupby('season')['touches'].transform(lambda s: s.shift(1).ewm(alpha=0.3, adjust=False).mean()).fillna(0.0)
        df['roll_passes_made'] = df.groupby('season')['passes_made'].transform(lambda s: s.shift(1).ewm(alpha=0.3, adjust=False).mean()).fillna(0.0)
        df['roll_time_of_possession_sec'] = df.groupby('season')['time_of_possession_sec'].transform(lambda s: s.shift(1).ewm(alpha=0.3, adjust=False).mean()).fillna(0.0)
        df['roll_defender_distance'] = df.groupby('season')['avg_defender_distance_ft'].transform(lambda s: s.shift(1).ewm(alpha=0.3, adjust=False).mean()).fillna(4.5)
        df['roll_shorthanded_points'] = df.groupby('season')['shorthanded_points'].transform(lambda s: s.shift(1).ewm(alpha=0.3, adjust=False).mean()).fillna(0.0)
        df['roll_plus_minus'] = df.groupby('season')['plus_minus'].transform(lambda s: s.shift(1).ewm(alpha=0.3, adjust=False).mean()).fillna(0.0)
        
        df['roll_ts_pct'] = df.groupby('season')['ts_pct'].transform(lambda s: s.shift(1).ewm(alpha=0.3, adjust=False).mean()).fillna(0.0)
        df['roll_reb_rate'] = df.groupby('season')['reb_rate'].transform(lambda s: s.shift(1).ewm(alpha=0.3, adjust=False).mean()).fillna(0.0)
        df['roll_ast_rate'] = df.groupby('season')['ast_rate'].transform(lambda s: s.shift(1).ewm(alpha=0.3, adjust=False).mean()).fillna(0.0)
        df['roll_usg_efficiency'] = df.groupby('season')['usg_efficiency'].transform(lambda s: s.shift(1).ewm(alpha=0.3, adjust=False).mean()).fillna(0.0)"""

old_rolling = """        for col, default in [('potential_assists', 0.0), ('touches', 0.0), ('passes_made', 0.0), ('time_of_possession_sec', 0.0), ('avg_defender_distance_ft', 4.5)]:
            if col not in df.columns:
                df[col] = default
            else:
                df[col] = df[col].fillna(default)

        df['roll_potential_assists'] = df.groupby('season')['potential_assists'].transform(
            lambda s: s.shift(1).ewm(alpha=0.3, adjust=False).mean()
        ).fillna(0.0)
        df['roll_touches'] = df.groupby('season')['touches'].transform(
            lambda s: s.shift(1).ewm(alpha=0.3, adjust=False).mean()
        ).fillna(0.0)
        df['roll_passes_made'] = df.groupby('season')['passes_made'].transform(
            lambda s: s.shift(1).ewm(alpha=0.3, adjust=False).mean()
        ).fillna(0.0)
        df['roll_time_of_possession_sec'] = df.groupby('season')['time_of_possession_sec'].transform(
            lambda s: s.shift(1).ewm(alpha=0.3, adjust=False).mean()
        ).fillna(0.0)
        df['roll_defender_distance'] = df.groupby('season')['avg_defender_distance_ft'].transform(
            lambda s: s.shift(1).ewm(alpha=0.3, adjust=False).mean()
        ).fillna(4.5)"""

features_code = features_code.replace(old_rolling, rolling_features_replacement)

with open("src/features.py", "w") as f:
    f.write(features_code)


with open("src/train_global_model.py", "r") as f:
    train_code = f.read()

# 3. Add to FEATURE_COLUMNS in train_global_model.py
old_features = """FEATURE_COLUMNS = [
    'roll_min', 'roll_stat', 'roll_usg',
    'days_rest', 'is_back_to_back', 'opp_def_rating', 'opp_pace_factor',
    'opp_pos_pts_allowed', 'vacated_usage',
    'roll_potential_assists', 'roll_touches', 'roll_passes_made',
    'roll_time_of_possession_sec', 'roll_defender_distance',
    'is_home_game', 'pace_differential', 'roll_stat_var', 'roll_residual'
]"""
new_features = """FEATURE_COLUMNS = [
    'roll_min', 'roll_stat', 'roll_usg',
    'days_rest', 'is_back_to_back', 'opp_def_rating', 'opp_pace_factor',
    'opp_pos_pts_allowed', 'vacated_usage',
    'roll_potential_assists', 'roll_touches', 'roll_passes_made',
    'roll_time_of_possession_sec', 'roll_defender_distance',
    'is_home_game', 'pace_differential', 'roll_stat_var', 'roll_residual',
    'roll_ts_pct', 'roll_reb_rate', 'roll_ast_rate', 'roll_shorthanded_points',
    'roll_plus_minus', 'roll_usg_efficiency'
]"""
train_code = train_code.replace(old_features, new_features)

# 4. Modify Ridge
old_ridge = """        meta_alpha = 1.0
        if target_key in ['ast', 'reb']:
            meta_alpha = 1000.0  # Extreme shrinkage for low-variance targets
        
        meta_learner = Ridge(alpha=meta_alpha)"""
new_ridge = """        from sklearn.linear_model import RidgeCV
        meta_learner = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0, 500.0, 1000.0])"""
train_code = train_code.replace(old_ridge, new_ridge)

# 5. Fix fallback columns in src/features.py just to be safe
with open("src/train_global_model.py", "w") as f:
    f.write(train_code)

print("Modifications applied successfully")
