-- 003_cleanup_stale_tickers.sql: Remove delisted/acquired tickers and fix dot-notation symbols

-- Delete stale S&P 500 tickers (acquired, merged, or delisted)
DELETE FROM universe_tickers WHERE symbol IN (
    'ANSS', 'ATVI', 'CTLT', 'DFS', 'DISH', 'FBHS', 'FLT', 'FRC',
    'IPG', 'JNPR', 'K', 'MRO', 'PARA', 'PEAK', 'PKI', 'PXD', 'RE', 'SIVB', 'WBA'
);

-- Delete stale popular-options tickers (dead/delisted)
DELETE FROM universe_tickers WHERE symbol IN (
    'AMPS', 'BLDE', 'CZOO', 'DWAC', 'EDR', 'EXAI', 'GNUS', 'LTHM',
    'MTTR', 'MULN', 'SQ', 'STEM', 'WISH', 'WKHS', 'ZI'
);

-- Rename BRK.B -> BRK-B (yfinance dash format)
INSERT OR IGNORE INTO universe_tickers (symbol, name, sector, source, is_active, created_at, last_scanned_at)
    SELECT 'BRK-B', name, sector, source, is_active, created_at, last_scanned_at
    FROM universe_tickers WHERE symbol = 'BRK.B';
INSERT OR IGNORE INTO ticker_tags (symbol, tag_id)
    SELECT 'BRK-B', tag_id FROM ticker_tags WHERE symbol = 'BRK.B';
DELETE FROM universe_tickers WHERE symbol = 'BRK.B';

-- Rename BF.B -> BF-B (yfinance dash format)
INSERT OR IGNORE INTO universe_tickers (symbol, name, sector, source, is_active, created_at, last_scanned_at)
    SELECT 'BF-B', name, sector, source, is_active, created_at, last_scanned_at
    FROM universe_tickers WHERE symbol = 'BF.B';
INSERT OR IGNORE INTO ticker_tags (symbol, tag_id)
    SELECT 'BF-B', tag_id FROM ticker_tags WHERE symbol = 'BF.B';
DELETE FROM universe_tickers WHERE symbol = 'BF.B';
