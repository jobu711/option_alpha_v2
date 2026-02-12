-- 001_initial.sql: Initial schema for Option Alpha persistence layer

CREATE TABLE IF NOT EXISTS scan_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT    NOT NULL UNIQUE,
    timestamp       TEXT    NOT NULL,
    ticker_count    INTEGER NOT NULL DEFAULT 0,
    duration_seconds REAL,
    status          TEXT    NOT NULL DEFAULT 'pending',
    error_message   TEXT,
    scores_computed INTEGER NOT NULL DEFAULT 0,
    debates_completed INTEGER NOT NULL DEFAULT 0,
    options_analyzed INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS ticker_scores (
    id                          INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_run_id                 INTEGER NOT NULL,
    ticker                      TEXT    NOT NULL,
    composite_score             REAL    NOT NULL,
    direction                   TEXT    NOT NULL DEFAULT 'neutral',
    last_price                  REAL,
    avg_volume                  REAL,
    score_breakdown_json        TEXT,
    options_recommendation_json TEXT,
    timestamp                   TEXT    NOT NULL,
    FOREIGN KEY (scan_run_id) REFERENCES scan_runs(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS ai_theses (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_run_id     INTEGER NOT NULL,
    ticker          TEXT    NOT NULL,
    bull_thesis     TEXT,
    bear_thesis     TEXT,
    risk_synthesis  TEXT,
    conviction      INTEGER,
    recommendation  TEXT,
    direction       TEXT    NOT NULL DEFAULT 'neutral',
    FOREIGN KEY (scan_run_id) REFERENCES scan_runs(id) ON DELETE CASCADE
);

-- Indexes for historical queries by ticker
CREATE INDEX IF NOT EXISTS idx_ticker_scores_ticker_run
    ON ticker_scores (ticker, scan_run_id);

CREATE INDEX IF NOT EXISTS idx_ai_theses_ticker_run
    ON ai_theses (ticker, scan_run_id);

-- Index for looking up scans by status and timestamp
CREATE INDEX IF NOT EXISTS idx_scan_runs_timestamp
    ON scan_runs (timestamp DESC);
