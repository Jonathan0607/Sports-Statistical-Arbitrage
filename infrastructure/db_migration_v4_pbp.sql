CREATE TABLE IF NOT EXISTS play_by_play_events (
    game_id VARCHAR(20) NOT NULL,
    event_num INTEGER NOT NULL,
    event_msg_type INTEGER,
    event_msg_action_type INTEGER,
    period INTEGER,
    wc_timestring VARCHAR(10),
    pctimestring VARCHAR(10),
    homedescription TEXT,
    neutraldescription TEXT,
    visitordescription TEXT,
    score VARCHAR(10),
    score_margin VARCHAR(10),
    person1type INTEGER,
    player1_id VARCHAR(20),
    player1_name VARCHAR(100),
    player1_team_id VARCHAR(20),
    player2_id VARCHAR(20),
    player2_name VARCHAR(100),
    player2_team_id VARCHAR(20),
    player3_id VARCHAR(20),
    player3_name VARCHAR(100),
    player3_team_id VARCHAR(20),
    PRIMARY KEY (game_id, event_num)
);

CREATE INDEX IF NOT EXISTS idx_pbp_game_id ON play_by_play_events(game_id);
CREATE INDEX IF NOT EXISTS idx_pbp_player1 ON play_by_play_events(player1_id);
CREATE INDEX IF NOT EXISTS idx_pbp_player2 ON play_by_play_events(player2_id);
CREATE INDEX IF NOT EXISTS idx_pbp_event_type ON play_by_play_events(event_msg_type);
