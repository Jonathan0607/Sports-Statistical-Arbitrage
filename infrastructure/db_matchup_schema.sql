-- Schema for Team Matchup Metrics
CREATE TABLE IF NOT EXISTS team_matchup_metrics (
    team_id INT REFERENCES teams(team_id),
    game_id INT,
    rolling_def_rating NUMERIC(6, 2),
    rolling_pace_factor NUMERIC(5, 2),
    points_allowed_vs_pg NUMERIC(5, 2),
    points_allowed_vs_sg NUMERIC(5, 2),
    points_allowed_vs_sf NUMERIC(5, 2),
    points_allowed_vs_pf NUMERIC(5, 2),
    points_allowed_vs_c NUMERIC(5, 2),
    PRIMARY KEY (team_id, game_id)
);
