-- ============================================================================
-- V2 Migration Script — Adapted for existing master_player_id VARCHAR schema
-- Safe, additive upgrade to full Gemini blueprint
-- ============================================================================

-- ============================================================================
-- STEP 1: Upgrade the players table with new columns
-- ============================================================================

DO $$
BEGIN
    -- Add sport column
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'players' AND column_name = 'sport'
    ) THEN
        ALTER TABLE players ADD COLUMN sport VARCHAR(10) DEFAULT 'NBA';
    END IF;

    -- Add is_active column
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'players' AND column_name = 'is_active'
    ) THEN
        ALTER TABLE players ADD COLUMN is_active BOOLEAN DEFAULT TRUE;
    END IF;

    -- Add current_team_id FK column
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'players' AND column_name = 'current_team_id'
    ) THEN
        ALTER TABLE players ADD COLUMN current_team_id INT REFERENCES teams(team_id);
    END IF;

    -- Add created_at column
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'players' AND column_name = 'created_at'
    ) THEN
        ALTER TABLE players ADD COLUMN created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP;
    END IF;
END $$;

-- ============================================================================
-- STEP 2: Upgrade player_mappings with new columns
-- ============================================================================

DO $$
BEGIN
    -- Add remote_player_id column
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'player_mappings' AND column_name = 'remote_player_id'
    ) THEN
        ALTER TABLE player_mappings ADD COLUMN remote_player_id VARCHAR(100);
    END IF;

    -- Add updated_at column
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'player_mappings' AND column_name = 'updated_at'
    ) THEN
        ALTER TABLE player_mappings ADD COLUMN updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP;
    END IF;
END $$;

-- ============================================================================
-- STEP 3: Add the new odds_ticks hypertable
-- Uses master_player_id VARCHAR to match existing players table PK
-- ============================================================================

CREATE TABLE IF NOT EXISTS odds_ticks (
    timestamp TIMESTAMPTZ NOT NULL,
    game_id INT REFERENCES games(game_id),
    player_id VARCHAR(50) REFERENCES players(master_player_id),
    book_name VARCHAR(50) NOT NULL,
    market_type VARCHAR(50) NOT NULL,
    line_value NUMERIC(6, 2) NOT NULL,
    over_price INT NOT NULL,
    under_price INT NOT NULL,
    is_suspended BOOLEAN DEFAULT FALSE
);

SELECT create_hypertable('odds_ticks', 'timestamp', chunk_time_interval => INTERVAL '1 day', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_odds_lookup ON odds_ticks (game_id, player_id, book_name, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_odds_player_market ON odds_ticks (player_id, market_type, timestamp DESC);

-- ============================================================================
-- STEP 4: Add player_game_logs for model training data
-- Uses master_player_id VARCHAR to match existing players table PK
-- ============================================================================

CREATE TABLE IF NOT EXISTS player_game_logs (
    log_id SERIAL PRIMARY KEY,
    game_id INT REFERENCES games(game_id),
    player_id VARCHAR(50) REFERENCES players(master_player_id),
    minutes_played NUMERIC(5, 1),
    points INT,
    rebounds INT,
    assists INT,
    steals INT,
    blocks INT,
    turnovers INT,
    field_goals_made INT,
    field_goals_attempted INT,
    three_pointers_made INT,
    free_throws_made INT,
    free_throws_attempted INT,
    plus_minus INT,
    usage_rate NUMERIC(5, 2),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_game_logs_player ON player_game_logs (player_id, game_id);

-- ============================================================================
-- STEP 5: Enhance trade_execution with CLV tracking column
-- ============================================================================

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'trade_execution' AND column_name = 'closing_line_value'
    ) THEN
        ALTER TABLE trade_execution ADD COLUMN closing_line_value NUMERIC(5, 2);
    END IF;
END $$;

-- ============================================================================
-- VERIFICATION: List all tables
-- ============================================================================
SELECT table_name FROM information_schema.tables
WHERE table_schema = 'public' ORDER BY table_name;
