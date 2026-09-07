-- ============================================================================
-- V3 Migration Script — Player Tracking Data (Second Spectrum Metrics)
-- Adds player_tracking_logs table for NBA tracking metrics
-- (touches, passes, potential assists, defender distance, etc.)
-- ============================================================================

-- ============================================================================
-- STEP 1: Create player_tracking_logs table
-- Links to player_game_logs via game_id + player_id
-- ============================================================================

CREATE TABLE IF NOT EXISTS player_tracking_logs (
    tracking_id SERIAL PRIMARY KEY,
    game_id INT REFERENCES games(game_id),
    player_id VARCHAR(50) REFERENCES players(master_player_id),
    potential_assists NUMERIC(5, 2),
    passes_made NUMERIC(6, 2),
    touches NUMERIC(6, 2),
    time_of_possession_sec NUMERIC(6, 2),
    avg_defender_distance_ft NUMERIC(4, 2),
    shorthanded_points NUMERIC(5, 2),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (game_id, player_id)
);

-- ============================================================================
-- STEP 2: Create indexes for efficient feature engine JOINs
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_tracking_player_game
    ON player_tracking_logs (player_id, game_id);

CREATE INDEX IF NOT EXISTS idx_tracking_game_id
    ON player_tracking_logs (game_id);

-- ============================================================================
-- VERIFICATION: List all tables
-- ============================================================================
SELECT table_name FROM information_schema.tables
WHERE table_schema = 'public' ORDER BY table_name;
