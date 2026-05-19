-- TimescaleDB Schema for V1 NBA Player Points StatArb Engine

-- Enable TimescaleDB extension if available
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- 1. HISTORICAL TICKS TABLE (Raw price feed updates from books)
CREATE TABLE IF NOT EXISTS market_ticks (
    timestamp TIMESTAMPTZ NOT NULL,
    bookmaker VARCHAR(50) NOT NULL,
    player_id VARCHAR(50) NOT NULL,
    player_name VARCHAR(100) NOT NULL,
    stat_type VARCHAR(20) NOT NULL DEFAULT 'points', -- points, rebounds, assists, etc.
    line_value NUMERIC(5, 2) NOT NULL,              -- e.g. 24.5
    over_odds INTEGER NOT NULL,                      -- American Odds (e.g., -110, +120)
    under_odds INTEGER NOT NULL,                     -- American Odds (e.g., -110, +120)
    implied_probability_over NUMERIC(5, 4) NOT NULL, -- computed probability
    implied_probability_under NUMERIC(5, 4) NOT NULL,
    is_suspended BOOLEAN NOT NULL DEFAULT FALSE
);

-- Convert to hypertable on timestamp (1 day partitions for high frequency data)
SELECT create_hypertable('market_ticks', 'timestamp', chunk_time_interval => INTERVAL '1 day', if_not_exists => TRUE);

-- Create indexes for performance on queries filtering by player or book
CREATE INDEX IF NOT EXISTS idx_ticks_player_book ON market_ticks (player_id, bookmaker, timestamp DESC);

-- 2. STALE LINES / ARBITRAGE SIGNALS (Stale line detections & deviations)
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
    expected_value NUMERIC(6, 4) NOT NULL,          -- e.g., 0.0345 (3.45% EV)
    z_score NUMERIC(5, 2),                           -- Divergence strength
    kelly_fraction NUMERIC(5, 4) NOT NULL            -- Capital allocation recommendation
);

SELECT create_hypertable('arb_signals', 'timestamp', chunk_time_interval => INTERVAL '7 days', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_signals_player ON arb_signals (player_id, timestamp DESC);

-- 3. EXECUTED TRADES (Audit trail and risk management performance tracking)
CREATE TABLE IF NOT EXISTS trade_execution (
    trade_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMPTZ NOT NULL,
    player_id VARCHAR(50) NOT NULL,
    player_name VARCHAR(100) NOT NULL,
    bookmaker VARCHAR(50) NOT NULL,
    bet_type VARCHAR(10) NOT NULL,                    -- 'OVER' or 'UNDER'
    line_value NUMERIC(5, 2) NOT NULL,
    odds INTEGER NOT NULL,
    stake_amount NUMERIC(10, 2) NOT NULL,            -- Actual wager in USD
    expected_value NUMERIC(6, 4) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'PENDING',  -- PENDING, SUBMITTED, FILLED, REJECTED, SETTLED
    settlement_outcome VARCHAR(10),                 -- 'WIN', 'LOSS', 'PUSH'
    pnl NUMERIC(10, 2),                             -- Profit & Loss realized
    settled_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_trades_status ON trade_execution (status);
CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trade_execution (timestamp DESC);
