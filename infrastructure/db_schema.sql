-- ============================================================================
-- StatArb Engine V2 — Production Database Schema
-- TimescaleDB + PostgreSQL for NBA Player Prop Market Microstructure
-- ============================================================================

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- ============================================================================
-- 1. RELATIONAL METADATA TABLES (Standard PostgreSQL)
-- ============================================================================

-- Master Team Dictionary
CREATE TABLE IF NOT EXISTS teams (
    team_id SERIAL PRIMARY KEY,
    sport VARCHAR(10) NOT NULL,                -- e.g., 'NBA', 'MLB', 'NFL'
    abbreviation VARCHAR(10) NOT NULL,         -- e.g., 'DAL', 'LAL'
    full_name VARCHAR(100) NOT NULL,           -- e.g., 'Dallas Mavericks'
    conference VARCHAR(20),                    -- e.g., 'Western', 'Eastern'
    division VARCHAR(30),                      -- e.g., 'Southwest'
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(sport, abbreviation)
);

-- Master Player Dictionary
CREATE TABLE IF NOT EXISTS players (
    master_player_id VARCHAR(50) PRIMARY KEY,  -- Canonical ID used across all subsystems
    full_name VARCHAR(100) NOT NULL,           -- e.g., 'Luka Doncic'
    team_abbrev VARCHAR(10),                   -- e.g., 'DAL' (denormalized for fast lookups)
    position VARCHAR(10),                      -- e.g., 'PG', 'SG', 'SF', 'PF', 'C'
    sport VARCHAR(10) DEFAULT 'NBA',
    is_active BOOLEAN DEFAULT TRUE,
    current_team_id INT REFERENCES teams(team_id),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Schedule / Game Tracker
CREATE TABLE IF NOT EXISTS games (
    game_id SERIAL PRIMARY KEY,
    sport VARCHAR(10) NOT NULL,
    home_team_id INT REFERENCES teams(team_id),
    away_team_id INT REFERENCES teams(team_id),
    game_status VARCHAR(20) DEFAULT 'SCHEDULED',  -- SCHEDULED, LIVE, FINAL, POSTPONED
    start_time TIMESTAMPTZ NOT NULL,
    season VARCHAR(10),                        -- e.g., '2024-25'
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_games_start ON games (start_time DESC);
CREATE INDEX IF NOT EXISTS idx_games_status ON games (game_status);

-- Entity Resolution Mapping Matrix
-- Maps book-specific naming anomalies to our Master Player ID
CREATE TABLE IF NOT EXISTS player_mappings (
    mapping_id SERIAL PRIMARY KEY,
    master_player_id VARCHAR(50) REFERENCES players(master_player_id),
    book_name VARCHAR(50) NOT NULL,            -- e.g., 'prizepicks', 'draftkings', 'pinnacle'
    remote_player_name VARCHAR(150) NOT NULL,  -- e.g., 'P.J. Washington Jr.'
    remote_player_id VARCHAR(100),             -- The ID assigned by that specific book's API
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(book_name, remote_player_name)
);

-- ============================================================================
-- 2. TIME-SERIES MARKET DATA (TimescaleDB Hypertables)
-- ============================================================================

-- Continuous Odds Tick Stream (Primary market microstructure table)
-- Records every odds update from every book for line-movement analysis
CREATE TABLE IF NOT EXISTS odds_ticks (
    timestamp TIMESTAMPTZ NOT NULL,
    game_id INT REFERENCES games(game_id),
    player_id VARCHAR(50) REFERENCES players(master_player_id),
    book_name VARCHAR(50) NOT NULL,            -- e.g., 'draftkings', 'pinnacle', 'underdog'
    market_type VARCHAR(50) NOT NULL,          -- e.g., 'points', 'rebounds', 'strikeouts'
    line_value NUMERIC(6, 2) NOT NULL,         -- e.g., 25.50
    over_price INT NOT NULL,                   -- American format odds (e.g., -110)
    under_price INT NOT NULL,                  -- American format odds (e.g., +110)
    is_suspended BOOLEAN DEFAULT FALSE         -- Flags if the book temporarily locks trading
);

SELECT create_hypertable('odds_ticks', 'timestamp', chunk_time_interval => INTERVAL '1 day', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_odds_lookup ON odds_ticks (game_id, player_id, book_name, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_odds_player_market ON odds_ticks (player_id, market_type, timestamp DESC);

-- Historical Raw Market Ticks (Legacy V1 table — retained for backward compatibility)
CREATE TABLE IF NOT EXISTS market_ticks (
    timestamp TIMESTAMPTZ NOT NULL,
    bookmaker VARCHAR(50) NOT NULL,
    player_id VARCHAR(50) NOT NULL,
    player_name VARCHAR(100) NOT NULL,
    stat_type VARCHAR(20) NOT NULL DEFAULT 'points',
    line_value NUMERIC(5, 2) NOT NULL,
    over_odds INTEGER NOT NULL,
    under_odds INTEGER NOT NULL,
    implied_probability_over NUMERIC(5, 4) NOT NULL,
    implied_probability_under NUMERIC(5, 4) NOT NULL,
    is_suspended BOOLEAN NOT NULL DEFAULT FALSE
);

SELECT create_hypertable('market_ticks', 'timestamp', chunk_time_interval => INTERVAL '1 day', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_ticks_player_book ON market_ticks (player_id, bookmaker, timestamp DESC);

-- ============================================================================
-- 3. SIGNAL & EXECUTION TABLES
-- ============================================================================

-- Stale Lines / Arbitrage Signals (Detections & deviations)
CREATE TABLE IF NOT EXISTS arb_signals (
    timestamp TIMESTAMPTZ NOT NULL,
    player_id VARCHAR(50) NOT NULL,
    player_name VARCHAR(100) NOT NULL,
    sharp_book VARCHAR(50) NOT NULL,
    stale_book VARCHAR(50) NOT NULL,
    sharp_line NUMERIC(5, 2) NOT NULL,
    stale_line NUMERIC(5, 2) NOT NULL,
    sharp_over_odds INTEGER NOT NULL,
    stale_under_odds INTEGER NOT NULL,
    expected_value NUMERIC(6, 4) NOT NULL,     -- e.g., 0.0345 (3.45% EV)
    z_score NUMERIC(5, 2),                     -- Divergence strength
    kelly_fraction NUMERIC(5, 4) NOT NULL      -- Capital allocation recommendation
);

SELECT create_hypertable('arb_signals', 'timestamp', chunk_time_interval => INTERVAL '7 days', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_signals_player ON arb_signals (player_id, timestamp DESC);

-- Executed Trades (Audit trail and risk management performance tracking)
CREATE TABLE IF NOT EXISTS trade_execution (
    trade_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMPTZ NOT NULL,
    player_id VARCHAR(50) NOT NULL,
    player_name VARCHAR(100) NOT NULL,
    bookmaker VARCHAR(50) NOT NULL,
    bet_type VARCHAR(10) NOT NULL,             -- 'OVER' or 'UNDER'
    line_value NUMERIC(5, 2) NOT NULL,
    odds INTEGER NOT NULL,
    stake_amount NUMERIC(10, 2) NOT NULL,      -- Actual wager in USD
    expected_value NUMERIC(6, 4) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'PENDING',  -- PENDING, SUBMITTED, FILLED, REJECTED, SETTLED
    settlement_outcome VARCHAR(10),            -- 'WIN', 'LOSS', 'PUSH'
    pnl NUMERIC(10, 2),                        -- Profit & Loss realized
    closing_line_value NUMERIC(5, 2),          -- CLV: the sharp book's closing line at game start
    settled_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_trades_status ON trade_execution (status);
CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trade_execution (timestamp DESC);

-- ============================================================================
-- 4. HISTORICAL PERFORMANCE DATA
-- ============================================================================

-- Player Game Logs (Box scores for model training)
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
    usage_rate NUMERIC(5, 2),                  -- Calculated usage %
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_game_logs_player ON player_game_logs (player_id, game_id);

-- ============================================================================
-- 5. SEED DATA — NBA TEAMS (30 franchises)
-- ============================================================================

INSERT INTO teams (sport, abbreviation, full_name, conference, division) VALUES
    ('NBA', 'ATL', 'Atlanta Hawks', 'Eastern', 'Southeast'),
    ('NBA', 'BOS', 'Boston Celtics', 'Eastern', 'Atlantic'),
    ('NBA', 'BKN', 'Brooklyn Nets', 'Eastern', 'Atlantic'),
    ('NBA', 'CHA', 'Charlotte Hornets', 'Eastern', 'Southeast'),
    ('NBA', 'CHI', 'Chicago Bulls', 'Eastern', 'Central'),
    ('NBA', 'CLE', 'Cleveland Cavaliers', 'Eastern', 'Central'),
    ('NBA', 'DAL', 'Dallas Mavericks', 'Western', 'Southwest'),
    ('NBA', 'DEN', 'Denver Nuggets', 'Western', 'Northwest'),
    ('NBA', 'DET', 'Detroit Pistons', 'Eastern', 'Central'),
    ('NBA', 'GSW', 'Golden State Warriors', 'Western', 'Pacific'),
    ('NBA', 'HOU', 'Houston Rockets', 'Western', 'Southwest'),
    ('NBA', 'IND', 'Indiana Pacers', 'Eastern', 'Central'),
    ('NBA', 'LAC', 'Los Angeles Clippers', 'Western', 'Pacific'),
    ('NBA', 'LAL', 'Los Angeles Lakers', 'Western', 'Pacific'),
    ('NBA', 'MEM', 'Memphis Grizzlies', 'Western', 'Southwest'),
    ('NBA', 'MIA', 'Miami Heat', 'Eastern', 'Southeast'),
    ('NBA', 'MIL', 'Milwaukee Bucks', 'Eastern', 'Central'),
    ('NBA', 'MIN', 'Minnesota Timberwolves', 'Western', 'Northwest'),
    ('NBA', 'NOP', 'New Orleans Pelicans', 'Western', 'Southwest'),
    ('NBA', 'NYK', 'New York Knicks', 'Eastern', 'Atlantic'),
    ('NBA', 'OKC', 'Oklahoma City Thunder', 'Western', 'Northwest'),
    ('NBA', 'ORL', 'Orlando Magic', 'Eastern', 'Southeast'),
    ('NBA', 'PHI', 'Philadelphia 76ers', 'Eastern', 'Atlantic'),
    ('NBA', 'PHX', 'Phoenix Suns', 'Western', 'Pacific'),
    ('NBA', 'POR', 'Portland Trail Blazers', 'Western', 'Northwest'),
    ('NBA', 'SAC', 'Sacramento Kings', 'Western', 'Pacific'),
    ('NBA', 'SAS', 'San Antonio Spurs', 'Western', 'Southwest'),
    ('NBA', 'TOR', 'Toronto Raptors', 'Eastern', 'Atlantic'),
    ('NBA', 'UTA', 'Utah Jazz', 'Western', 'Northwest'),
    ('NBA', 'WAS', 'Washington Wizards', 'Eastern', 'Southeast')
ON CONFLICT (sport, abbreviation) DO NOTHING;
