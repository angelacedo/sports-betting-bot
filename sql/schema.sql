-- ============================================================
-- Sports Betting Bot - PostgreSQL Schema
-- ============================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ------------------------------------------------------------
-- LEAGUES
-- ------------------------------------------------------------
CREATE TABLE leagues (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    external_id     VARCHAR(64) UNIQUE,
    name            VARCHAR(128) NOT NULL,
    sport           VARCHAR(64) NOT NULL,
    country         VARCHAR(64),
    is_active       BOOLEAN DEFAULT TRUE,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ------------------------------------------------------------
-- TEAMS
-- ------------------------------------------------------------
CREATE TABLE teams (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    external_id     VARCHAR(64) UNIQUE,
    name            VARCHAR(128) NOT NULL,
    short_name      VARCHAR(32),
    country         VARCHAR(64),
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ------------------------------------------------------------
-- BOOKMAKERS
-- ------------------------------------------------------------
CREATE TABLE bookmakers (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    external_id     VARCHAR(64) UNIQUE,
    name            VARCHAR(128) NOT NULL,
    is_active       BOOLEAN DEFAULT TRUE,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ------------------------------------------------------------
-- MARKETS
-- ------------------------------------------------------------
CREATE TABLE markets (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    key             VARCHAR(64) UNIQUE NOT NULL,
    name            VARCHAR(128) NOT NULL,
    description     TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ------------------------------------------------------------
-- MATCHES
-- ------------------------------------------------------------
CREATE TABLE matches (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    external_id     VARCHAR(64) UNIQUE,
    league_id       UUID NOT NULL REFERENCES leagues(id) ON DELETE CASCADE,
    home_team_id    UUID NOT NULL REFERENCES teams(id) ON DELETE CASCADE,
    away_team_id    UUID NOT NULL REFERENCES teams(id) ON DELETE CASCADE,
    kickoff         TIMESTAMPTZ NOT NULL,
    status          VARCHAR(32) DEFAULT 'scheduled',
    home_score      SMALLINT,
    away_score      SMALLINT,
    season          VARCHAR(16),
    round           VARCHAR(16),
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_matches_league   ON matches(league_id);
CREATE INDEX idx_matches_kickoff  ON matches(kickoff);
CREATE INDEX idx_matches_status   ON matches(status);

-- ------------------------------------------------------------
-- ODDS HISTORY
-- ------------------------------------------------------------
CREATE TABLE odds_history (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    match_id        UUID NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    bookmaker_id    UUID NOT NULL REFERENCES bookmakers(id) ON DELETE CASCADE,
    market_id       UUID NOT NULL REFERENCES markets(id) ON DELETE CASCADE,
    selection       VARCHAR(64) NOT NULL,
    odds_decimal    NUMERIC(8, 4) NOT NULL,
    odds_implied    NUMERIC(6, 4),
    stake_limit     NUMERIC(12, 2),
    fetched_at      TIMESTAMPTZ DEFAULT NOW(),
    raw_api_data    JSONB,
    UNIQUE(match_id, bookmaker_id, market_id, selection, fetched_at)
);

CREATE INDEX idx_odds_match       ON odds_history(match_id);
CREATE INDEX idx_odds_bookmaker   ON odds_history(bookmaker_id);
CREATE INDEX idx_odds_fetched     ON odds_history(fetched_at);
CREATE INDEX idx_odds_lookup      ON odds_history(match_id, market_id, selection);

-- ------------------------------------------------------------
-- BOT BETS
-- ------------------------------------------------------------
CREATE TABLE bot_bets (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    match_id        UUID NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    market_id       UUID NOT NULL REFERENCES markets(id) ON DELETE CASCADE,
    selection       VARCHAR(64) NOT NULL,
    odds_decimal    NUMERIC(8, 4) NOT NULL,
    stake           NUMERIC(10, 2) NOT NULL,
    kelly_fraction  NUMERIC(4, 2),
    expected_value  NUMERIC(8, 4),
    model_name      VARCHAR(64),
    model_version   VARCHAR(16),
    status          VARCHAR(32) DEFAULT 'pending',
    placed_at       TIMESTAMPTZ,
    settled_at      TIMESTAMPTZ,
    pnl             NUMERIC(10, 2),
    notes           TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_bets_match  ON bot_bets(match_id);
CREATE INDEX idx_bets_status ON bot_bets(status);
CREATE INDEX idx_bets_placed ON bot_bets(placed_at);
