import os
import logging
import pandas as pd
import numpy as np
import sqlalchemy

logger = logging.getLogger("ContextualFeatureEngine")

class ContextualFeatureEngine:
    def __init__(self, db_uri: str = None):
        self.db_uri = db_uri or os.getenv("POSTGRES_URI", "postgresql+psycopg2://postgres:password@localhost:5432/sports_db")
        if self.db_uri.startswith("postgres://"):
            self.db_uri = self.db_uri.replace("postgres://", "postgresql+psycopg2://", 1)
        self.engine = sqlalchemy.create_engine(self.db_uri)
        self._load_player_info()

    def _load_player_info(self):
        """Loads and caches player team and position maps."""
        query = "SELECT master_player_id, current_team_id as team_id, position FROM players"
        try:
            with self.engine.connect() as conn:
                df = pd.read_sql(query, conn)
            self.player_teams = dict(zip(df['master_player_id'], df['team_id']))
            self.player_positions = dict(zip(df['master_player_id'], df['position']))
        except Exception as e:
            logger.error(f"Error loading player info: {e}")
            self.player_teams = {}
            self.player_positions = {}

    def get_player_team_and_pos(self, player_id: str):
        team_id = self.player_teams.get(player_id)
        pos = self.player_positions.get(player_id)
        if not pos:
            pos = "SG"
        return team_id, pos

    def get_co_stars(self, team_id: int, exclude_player_id: str = None) -> list:
        """
        Retrieves co-stars for a team.
        Co-stars are teammates (same team_id) with average minutes > 25 and usage > 25%.
        Returns a list of dicts containing player_id and their historical avg_usg.
        """
        if not team_id:
            return []
        query = """
            SELECT g.player_id, AVG(g.minutes_played) as avg_min, AVG(g.usage_rate) as avg_usg
            FROM player_game_logs g
            JOIN players p ON g.player_id = p.master_player_id
            WHERE p.current_team_id = :team_id
            GROUP BY g.player_id
            HAVING AVG(g.minutes_played) > 25.0 AND AVG(g.usage_rate) > 0.25;
        """
        try:
            with self.engine.connect() as conn:
                res = conn.execute(sqlalchemy.text(query), {"team_id": team_id}).fetchall()
            costars = [
                {"player_id": str(row[0]), "avg_usg": float(row[2])}
                for row in res
                if str(row[0]) != str(exclude_player_id)
            ]
            return costars
        except Exception as e:
            logger.error(f"Error getting co-stars: {e}")
            return []

    def get_sidelined_teammates_for_game(self, team_id: int, game_id: int, co_stars: list) -> float:
        """
        Calculates the total vacated usage of co-stars who were sidelined in a historical game.
        """
        if not co_stars:
            return 0.0
            
        co_star_map = {}
        if isinstance(co_stars, dict):
            co_star_map = co_stars
        elif isinstance(co_stars, list):
            for c in co_stars:
                if isinstance(c, dict):
                    co_star_map[str(c["player_id"])] = float(c.get("avg_usg", 0.25))
                elif isinstance(c, (list, tuple)) and len(c) >= 2:
                    co_star_map[str(c[0])] = float(c[1])
                elif isinstance(c, str):
                    co_star_map[c] = 0.25

        co_star_ids = list(co_star_map.keys())
        if not co_star_ids:
            return 0.0

        query = """
            SELECT player_id, minutes_played
            FROM player_game_logs
            WHERE game_id = :game_id AND player_id IN :co_stars;
        """
        try:
            with self.engine.connect() as conn:
                res = conn.execute(sqlalchemy.text(query), {"game_id": game_id, "co_stars": tuple(co_star_ids)}).fetchall()
            played_players = {str(row[0]): float(row[1]) for row in res}
            
            vacated_usg = 0.0
            for costar_id, costar_usg in co_star_map.items():
                mins = played_players.get(costar_id, 0.0)
                if mins < 5.0:
                    vacated_usg += costar_usg
            return float(vacated_usg)
        except Exception as e:
            logger.error(f"Error checking sidelined teammates vacated usage: {e}")
            return 0.0

    def build_player_feature_matrix(self, player_id: str, history_df: pd.DataFrame, target_date: str = None) -> pd.DataFrame:
        """
        Takes a player history DataFrame and enriches it with contextual features.
        The input history_df must contain at least 'game_id' and 'start_time'.
        If it doesn't, we will fetch the complete history from DB.
        """
        if 'game_id' not in history_df.columns or 'start_time' not in history_df.columns:
            # Fetch complete history with game dates and season tag
            query = """
                SELECT g.game_id, gm.start_time, gm.season, gm.home_team_id, gm.away_team_id,
                       g.minutes_played, g.points, g.rebounds, g.assists, g.turnovers,
                       g.steals, g.blocks, g.three_pointers_made, g.usage_rate,
                       g.field_goals_attempted, g.free_throws_attempted, g.plus_minus,
                       COALESCE(t.potential_assists, 0.0) AS potential_assists,
                       COALESCE(t.touches, 0.0) AS touches,
                       COALESCE(t.passes_made, 0.0) AS passes_made,
                       COALESCE(t.time_of_possession_sec, 0.0) AS time_of_possession_sec,
                       COALESCE(t.avg_defender_distance_ft, 4.5) AS avg_defender_distance_ft,
                       COALESCE(t.shorthanded_points, 0.0) AS shorthanded_points
                FROM player_game_logs g
                JOIN games gm ON g.game_id = gm.game_id
                LEFT JOIN player_tracking_logs t ON g.game_id = t.game_id AND g.player_id = t.player_id
                WHERE g.player_id = :player_id
                  AND (:target_date IS NULL OR gm.start_time < :target_date)
                ORDER BY gm.start_time ASC;
            """
            try:
                with self.engine.connect() as conn:
                    df = pd.read_sql(sqlalchemy.text(query), conn, params={"player_id": player_id, "target_date": target_date})
                if df.empty:
                    # Fallback to copy and add default columns
                    df = history_df.copy()
                    df['days_rest'] = 3.0
                    df['is_back_to_back'] = 0.0
                    df['opp_def_rating'] = 115.0
                    df['opp_pace_factor'] = 99.0
                    df['opp_pos_pts_allowed'] = 22.0
                    df['vacated_usage'] = 0.0
                    df['roll_potential_assists'] = 0.0
                    df['roll_touches'] = 0.0
                    df['roll_passes_made'] = 0.0
                    df['roll_time_of_possession_sec'] = 0.0
                    df['roll_defender_distance'] = 4.5
                    df['is_home_game'] = 0.0
                    df['pace_differential'] = 0.0
                    return df
                history_df = df
            except Exception as e:
                logger.error(f"Failed to fetch full history: {e}")
                # Fallback to copy and add default columns
                df = history_df.copy()
                df['days_rest'] = 3.0
                df['is_back_to_back'] = 0.0
                df['opp_def_rating'] = 115.0
                df['opp_pace_factor'] = 99.0
                df['opp_pos_pts_allowed'] = 22.0
                df['vacated_usage'] = 0.0
                df['roll_potential_assists'] = 0.0
                df['roll_touches'] = 0.0
                df['roll_passes_made'] = 0.0
                df['roll_time_of_possession_sec'] = 0.0
                df['roll_defender_distance'] = 4.5
                df['is_home_game'] = 0.0
                df['pace_differential'] = 0.0
                return df
        else:
            if target_date:
                history_df = history_df[pd.to_datetime(history_df['start_time'], utc=True) < pd.to_datetime(target_date, utc=True)].copy()

        df = history_df.copy()
        df['start_time'] = pd.to_datetime(df['start_time'], utc=True)
        df = df.sort_values('start_time').reset_index(drop=True)

        if 'season' not in df.columns:
            def infer_season(dt):
                if pd.isna(dt):
                    return '2025-26'
                y, m = dt.year, dt.month
                return f"{y}-{str(y+1)[-2:]}" if m >= 9 else f"{y-1}-{str(y)[-2:]}"
            df['season'] = df['start_time'].apply(infer_season)

        team_id, position = self.get_player_team_and_pos(player_id)
        if not team_id:
            if not df.empty and 'home_team_id' in df.columns:
                team_id = int(df.iloc[0]['home_team_id'])
            else:
                team_id = 0
                
        df['team_id'] = team_id
        df['player_id'] = player_id

        # 1. Rest & Back-to-Back features (reset at season boundaries to prevent summer bleedover)
        if 'home_team_id' in df.columns:
            df['is_home_game'] = (df['home_team_id'].astype(float) == float(team_id)).astype(float)
        else:
            df['is_home_game'] = 0.0

        days_rest_list = []
        is_b2b_list = []
        for i in range(len(df)):
            if i == 0 or (df.loc[i, 'season'] != df.loc[i-1, 'season']):
                days_rest = 3.0
            else:
                raw_days = (df.loc[i, 'start_time'] - df.loc[i-1, 'start_time']).days
                days_rest = float(raw_days)
            days_rest_list.append(float(days_rest))
            is_b2b_list.append(1.0 if days_rest == 1 else 0.0)

        df['days_rest'] = days_rest_list
        df['is_back_to_back'] = is_b2b_list

        # 2. Opponent-based features (Bulk query team_matchup_metrics)
        game_ids = df['game_id'].tolist()
        df['opponent_team_id'] = df.apply(
            lambda r: int(r['away_team_id']) if int(r['home_team_id']) == team_id else int(r['home_team_id']),
            axis=1
        )
        
        opp_metrics = {}
        if game_ids:
            pos_col = f"points_allowed_vs_{position.lower()}"
            if pos_col not in [
                "points_allowed_vs_pg", "points_allowed_vs_sg", "points_allowed_vs_sf",
                "points_allowed_vs_pf", "points_allowed_vs_c"
            ]:
                pos_col = "points_allowed_vs_sg"

            query_metrics = f"""
                SELECT team_id, game_id, rolling_def_rating, rolling_pace_factor, {pos_col}
                FROM team_matchup_metrics
                WHERE game_id IN :game_ids;
            """
            try:
                with self.engine.connect() as conn:
                    res = conn.execute(sqlalchemy.text(query_metrics), {"game_ids": tuple(game_ids)}).fetchall()
                for row in res:
                    opp_metrics[(int(row[0]), int(row[1]))] = {
                        "opp_def_rating": float(row[2]),
                        "opp_pace_factor": float(row[3]),
                        "opp_pos_pts_allowed": float(row[4])
                    }
            except Exception as e:
                logger.error(f"Failed to query bulk matchup metrics: {e}")

        # Map opponent metrics to df
        opp_def_rating_list = []
        opp_pace_factor_list = []
        opp_pos_pts_allowed_list = []
        pace_differential_list = []
        fallback_pos_allowed = {"PG": 24.0, "SG": 22.0, "SF": 21.0, "PF": 22.0, "C": 23.0}

        for _, row in df.iterrows():
            opp_id = int(row['opponent_team_id'])
            g_id = int(row['game_id'])
            metrics = opp_metrics.get((opp_id, g_id))
            own_metrics = opp_metrics.get((team_id, g_id))
            
            if metrics:
                opp_def_rating_list.append(metrics["opp_def_rating"])
                opp_pace_factor_list.append(metrics["opp_pace_factor"])
                opp_pos_pts_allowed_list.append(metrics["opp_pos_pts_allowed"])
            else:
                opp_def_rating_list.append(115.0)
                opp_pace_factor_list.append(99.0)
                opp_pos_pts_allowed_list.append(fallback_pos_allowed.get(position.upper(), 22.0))
            
            if own_metrics and metrics:
                pace_differential_list.append(own_metrics["opp_pace_factor"] - metrics["opp_pace_factor"])
            else:
                pace_differential_list.append(0.0)

        df['opp_def_rating'] = opp_def_rating_list
        df['opp_pace_factor'] = opp_pace_factor_list
        df['opp_pos_pts_allowed'] = opp_pos_pts_allowed_list
        df['pace_differential'] = pace_differential_list

        # 3. Vacated usage feature (Bulk query)
        co_stars = self.get_co_stars(team_id, exclude_player_id=player_id)
        vacated_usage_list = []
        
        co_star_map = {}
        if isinstance(co_stars, dict):
            co_star_map = co_stars
        elif isinstance(co_stars, list):
            for c in co_stars:
                if isinstance(c, dict):
                    co_star_map[str(c["player_id"])] = float(c.get("avg_usg", 0.25))
                elif isinstance(c, (list, tuple)) and len(c) >= 2:
                    co_star_map[str(c[0])] = float(c[1])
                elif isinstance(c, str):
                    co_star_map[c] = 0.25

        co_star_ids = list(co_star_map.keys())
        vacated_by_game = {}
        if co_star_ids and game_ids:
            query_sidelined = """
                SELECT game_id, player_id, minutes_played
                FROM player_game_logs
                WHERE game_id IN :game_ids AND player_id IN :co_stars;
            """
            try:
                with self.engine.connect() as conn:
                    res = conn.execute(sqlalchemy.text(query_sidelined), {
                        "game_ids": tuple(game_ids),
                        "co_stars": tuple(co_star_ids)
                    }).fetchall()
                
                played = {}
                for row in res:
                    g_id, p_id, mins = int(row[0]), str(row[1]), float(row[2])
                    if g_id not in played:
                        played[g_id] = {}
                    played[g_id][p_id] = mins
                
                for g_id in game_ids:
                    game_played = played.get(g_id, {})
                    vacated_usg = 0.0
                    for costar_id, costar_usg in co_star_map.items():
                        mins = game_played.get(costar_id, 0.0)
                        if mins < 5.0:
                            vacated_usg += costar_usg
                    vacated_by_game[g_id] = float(vacated_usg)
            except Exception as e:
                logger.error(f"Failed to query teammate vacated usage: {e}")

        for _, row in df.iterrows():
            g_id = int(row['game_id'])
            vacated_usage_list.append(vacated_by_game.get(g_id, 0.0))

        df['vacated_usage'] = vacated_usage_list

        # 4. Rolling tracking features (from player_tracking_logs) - partitioned strictly by season
        for col, default in [('potential_assists', 0.0), ('touches', 0.0), ('passes_made', 0.0), ('time_of_possession_sec', 0.0), ('avg_defender_distance_ft', 4.5), ('shorthanded_points', 0.0), ('plus_minus', 0.0), ('field_goals_attempted', 0.0), ('free_throws_attempted', 0.0)]:
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
        df['roll_usg_efficiency'] = df.groupby('season')['usg_efficiency'].transform(lambda s: s.shift(1).ewm(alpha=0.3, adjust=False).mean()).fillna(0.0)

        return df

    def build_active_features(self, player_id: str, active_polled_player_ids: list = None, target_date: str = None) -> dict:
        """
        Builds the contextual features for the upcoming game.
        """
        team_id, position = self.get_player_team_and_pos(player_id)
        if not team_id:
            return {
                "days_rest": 3.0,
                "is_back_to_back": 0.0,
                "opp_def_rating": 115.0,
                "opp_pace_factor": 99.0,
                "opp_pos_pts_allowed": 22.0,
                "vacated_usage": 0.0,
                "is_home_game": 0.0,
                "pace_differential": 0.0
            }

        # Query upcoming game
        query_game = """
            SELECT game_id, home_team_id, away_team_id, start_time, season
            FROM games
            WHERE (home_team_id = :team_id OR away_team_id = :team_id)
              AND (
                (:target_date IS NOT NULL AND start_time >= :target_date)
                OR (:target_date IS NULL AND start_time >= NOW() - INTERVAL '12 hours')
              )
            ORDER BY start_time ASC
            LIMIT 1;
        """
        
        # Query last game date and season to compute days_rest strictly within the same season
        query_last_game = """
            SELECT gm.start_time, gm.season
            FROM player_game_logs g
            JOIN games gm ON g.game_id = gm.game_id
            WHERE g.player_id = :player_id
              AND (:target_date IS NULL OR gm.start_time < :target_date)
            ORDER BY gm.start_time DESC
            LIMIT 1;
        """

        try:
            with self.engine.connect() as conn:
                game_row = conn.execute(sqlalchemy.text(query_game), {"team_id": team_id, "target_date": target_date}).fetchone()
                last_game_row = conn.execute(sqlalchemy.text(query_last_game), {"player_id": player_id, "target_date": target_date}).fetchone()
        except Exception as e:
            logger.error(f"Error querying active game/last game: {e}")
            game_row = None
            last_game_row = None

        # 1. Rest days (if new season, reset to 3.0)
        if game_row and last_game_row:
            game_time = pd.to_datetime(game_row[3])
            last_game_time = pd.to_datetime(last_game_row[0])
            game_season = game_row[4] if len(game_row) > 4 else None
            last_game_season = last_game_row[1] if len(last_game_row) > 1 else None
            if game_season and last_game_season and game_season != last_game_season:
                days_rest = 3.0
            else:
                days_rest = float((game_time - last_game_time).days)
        else:
            days_rest = 3.0
        
        is_b2b = 1.0 if days_rest == 1 else 0.0

        # 2. Opponent metrics
        if game_row:
            opp_id = int(game_row[2]) if int(game_row[1]) == team_id else int(game_row[1])
            pos_col = f"points_allowed_vs_{position.lower()}"
            if pos_col not in [
                "points_allowed_vs_pg", "points_allowed_vs_sg", "points_allowed_vs_sf",
                "points_allowed_vs_pf", "points_allowed_vs_c"
            ]:
                pos_col = "points_allowed_vs_sg"

            query_metrics = f"""
                SELECT m.rolling_def_rating, m.rolling_pace_factor, m.{pos_col}
                FROM team_matchup_metrics m
                JOIN games g ON m.game_id = g.game_id
                WHERE m.team_id = :opp_id
                  AND g.start_time < :game_time
                ORDER BY g.start_time DESC
                LIMIT 1;
            """
            try:
                with self.engine.connect() as conn:
                    metric_row = conn.execute(sqlalchemy.text(query_metrics), {"opp_id": opp_id, "game_time": game_row[3]}).fetchone()
                    own_metric_row = conn.execute(sqlalchemy.text(query_metrics), {"opp_id": team_id, "game_time": game_row[3]}).fetchone()
                if metric_row:
                    opp_def = float(metric_row[0])
                    opp_pace = float(metric_row[1])
                    opp_pos_pts = float(metric_row[2])
                else:
                    opp_def = 115.0
                    opp_pace = 99.0
                    fallback_pos_allowed = {"PG": 24.0, "SG": 22.0, "SF": 21.0, "PF": 22.0, "C": 23.0}
                    opp_pos_pts = fallback_pos_allowed.get(position.upper(), 22.0)
                    
                if own_metric_row and metric_row:
                    pace_diff = float(own_metric_row[1]) - opp_pace
                else:
                    pace_diff = 0.0
                    
            except Exception as e:
                logger.error(f"Error querying active matchup metrics: {e}")
                opp_def = 115.0
                opp_pace = 99.0
                opp_pos_pts = 22.0
                pace_diff = 0.0
        else:
            opp_def = 115.0
            opp_pace = 99.0
            opp_pos_pts = 22.0
            pace_diff = 0.0
            
        is_home_game = 1.0 if game_row and int(game_row[1]) == team_id else 0.0

        # 3. Vacated usage for upcoming game
        co_stars = self.get_co_stars(team_id, exclude_player_id=player_id)
        vacated_usage = 0.0

        co_star_map = {}
        if isinstance(co_stars, dict):
            co_star_map = co_stars
        elif isinstance(co_stars, list):
            for c in co_stars:
                if isinstance(c, dict):
                    co_star_map[str(c["player_id"])] = float(c.get("avg_usg", 0.25))
                elif isinstance(c, (list, tuple)) and len(c) >= 2:
                    co_star_map[str(c[0])] = float(c[1])
                elif isinstance(c, str):
                    co_star_map[c] = 0.25

        co_star_ids = list(co_star_map.keys())

        if co_star_ids:
            query_team_games = """
                SELECT COUNT(DISTINCT game_id)
                FROM games
                WHERE (home_team_id = :team_id OR away_team_id = :team_id)
                  AND (
                    (:target_date IS NOT NULL AND start_time < :target_date)
                    OR (:target_date IS NULL AND start_time < NOW())
                  );
            """
            query_star_games = """
                SELECT g.player_id, COUNT(DISTINCT g.game_id)
                FROM player_game_logs g
                JOIN games gm ON g.game_id = gm.game_id
                WHERE g.player_id IN :co_stars
                  AND (
                    (:target_date IS NOT NULL AND gm.start_time < :target_date)
                    OR (:target_date IS NULL AND gm.start_time < NOW())
                  )
                GROUP BY g.player_id;
            """
            try:
                with self.engine.connect() as conn:
                    team_games_count = conn.execute(sqlalchemy.text(query_team_games), {
                        "team_id": team_id,
                        "target_date": target_date
                    }).scalar() or 1
                    star_games_res = conn.execute(sqlalchemy.text(query_star_games), {
                        "co_stars": tuple(co_star_ids),
                        "target_date": target_date
                    }).fetchall()
                star_games = {str(row[0]): row[1] for row in star_games_res}

                for costar_id, costar_usg in co_star_map.items():
                    star_count = star_games.get(costar_id, 0)
                    is_sidelined = False
                    # Sidelined if played < 50% of team games
                    if star_count / team_games_count < 0.50:
                        is_sidelined = True
                    
                    # Or if costar is marked out / not in active polled players
                    if active_polled_player_ids and costar_id not in active_polled_player_ids:
                        is_sidelined = True

                    if is_sidelined:
                        vacated_usage += costar_usg
            except Exception as e:
                logger.error(f"Error evaluating active vacated usage: {e}")

        # 4. Rolling tracking features for the active/upcoming game
        roll_pot_ast = 0.0
        roll_touches = 0.0
        roll_passes_made = 0.0
        roll_time_of_possession_sec = 0.0
        roll_def_dist = 4.5

        query_tracking = """
            SELECT t.potential_assists, t.touches, t.passes_made, t.time_of_possession_sec, t.avg_defender_distance_ft, t.shorthanded_points
            FROM player_tracking_logs t
            JOIN games gm ON t.game_id = gm.game_id
            WHERE t.player_id = :player_id
              AND (:target_date IS NULL OR gm.start_time < :target_date)
            ORDER BY gm.start_time DESC
            LIMIT 3;
        """
        try:
            with self.engine.connect() as conn:
                track_rows = conn.execute(
                    sqlalchemy.text(query_tracking),
                    {"player_id": player_id, "target_date": target_date}
                ).fetchall()
            if track_rows:
                pot_ast_vals = [float(r[0]) if r[0] is not None else 0.0 for r in track_rows]
                touch_vals = [float(r[1]) if r[1] is not None else 0.0 for r in track_rows]
                pass_vals = [float(r[2]) if r[2] is not None else 0.0 for r in track_rows]
                top_vals = [float(r[3]) if r[3] is not None else 0.0 for r in track_rows]
                def_dist_vals = [float(r[4]) if r[4] is not None else 4.5 for r in track_rows]
                short_pts_vals = [float(r[5]) if r[5] is not None else 0.0 for r in track_rows]
                roll_pot_ast = float(np.mean(pot_ast_vals))
                roll_touches = float(np.mean(touch_vals))
                roll_passes_made = float(np.mean(pass_vals))
                roll_time_of_possession_sec = float(np.mean(top_vals))
                roll_def_dist = float(np.mean(def_dist_vals))
                roll_shorthanded_points = float(np.mean(short_pts_vals))
        except Exception as e:
            logger.error(f"Error querying active tracking features: {e}")

        return {
            "days_rest": days_rest,
            "is_back_to_back": is_b2b,
            "opp_def_rating": opp_def,
            "opp_pace_factor": opp_pace,
            "opp_pos_pts_allowed": opp_pos_pts,
            "vacated_usage": float(vacated_usage),
            "roll_potential_assists": roll_pot_ast,
            "roll_touches": roll_touches,
            "roll_passes_made": roll_passes_made,
            "roll_time_of_possession_sec": roll_time_of_possession_sec,
            "roll_defender_distance": roll_def_dist,
            "roll_shorthanded_points": locals().get("roll_shorthanded_points", 0.0),
            "is_home_game": is_home_game,
            "pace_differential": pace_diff
        }
