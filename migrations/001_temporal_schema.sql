-- Migration 001: Create temporal data schema for ML4T point-in-time system
-- This script creates the necessary tables and indexes for the temporal data management system

-- Create schema if it doesn't exist
CREATE SCHEMA IF NOT EXISTS ml4t;
SET search_path TO ml4t, public;

-- Point-in-time data table
CREATE TABLE IF NOT EXISTS pit_data (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    data_type VARCHAR(50) NOT NULL,
    as_of_date DATE NOT NULL,
    value_date DATE NOT NULL,
    value_numeric DECIMAL(20,6),
    value_text TEXT,
    value_json JSONB,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    version INTEGER DEFAULT 1,
    
    -- Constraints
    CONSTRAINT pit_data_unique UNIQUE (symbol, data_type, as_of_date, value_date, version),
    CONSTRAINT pit_data_dates_check CHECK (as_of_date >= value_date - INTERVAL '365 days'), -- Reasonable lag limit
    CONSTRAINT pit_data_version_positive CHECK (version > 0)
);

-- Indexes for optimal query performance
CREATE INDEX IF NOT EXISTS idx_pit_symbol_asof_type ON pit_data (symbol, as_of_date, data_type);
CREATE INDEX IF NOT EXISTS idx_pit_value_date ON pit_data (value_date);
CREATE INDEX IF NOT EXISTS idx_pit_data_type ON pit_data (data_type);
CREATE INDEX IF NOT EXISTS idx_pit_created_at ON pit_data (created_at);
CREATE INDEX IF NOT EXISTS idx_pit_metadata_gin ON pit_data USING GIN (metadata);
CREATE INDEX IF NOT EXISTS idx_pit_symbol_value_date ON pit_data (symbol, value_date);
CREATE INDEX IF NOT EXISTS idx_pit_asof_value_date ON pit_data (as_of_date, value_date);

-- Partial indexes for common data types (better performance)
CREATE INDEX IF NOT EXISTS idx_pit_price_data ON pit_data (symbol, as_of_date, value_date) 
    WHERE data_type = 'price';
CREATE INDEX IF NOT EXISTS idx_pit_volume_data ON pit_data (symbol, as_of_date, value_date) 
    WHERE data_type = 'volume';
CREATE INDEX IF NOT EXISTS idx_pit_fundamental_data ON pit_data (symbol, as_of_date, value_date) 
    WHERE data_type = 'fundamental';

-- Settlement calendar table
CREATE TABLE IF NOT EXISTS settlement_calendar (
    trade_date DATE PRIMARY KEY,
    settlement_date DATE NOT NULL,
    is_trading_day BOOLEAN NOT NULL DEFAULT TRUE,
    market_session VARCHAR(20) DEFAULT 'morning',
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT settlement_dates_check CHECK (settlement_date >= trade_date),
    CONSTRAINT market_session_check CHECK (market_session IN ('morning', 'closed', 'holiday', 'suspended'))
);

-- Index for settlement calendar
CREATE INDEX IF NOT EXISTS idx_settlement_date ON settlement_calendar (settlement_date);
CREATE INDEX IF NOT EXISTS idx_settlement_trading_day ON settlement_calendar (is_trading_day, trade_date);

-- Taiwan market metadata table
CREATE TABLE IF NOT EXISTS taiwan_stock_info (
    symbol VARCHAR(20) PRIMARY KEY,
    name_zh VARCHAR(200) NOT NULL,
    name_en VARCHAR(200),
    market VARCHAR(10) NOT NULL DEFAULT 'TWSE',
    industry_code VARCHAR(20),
    sector VARCHAR(100),
    listing_date DATE,
    par_value DECIMAL(10,2),
    outstanding_shares BIGINT,
    trading_status VARCHAR(20) DEFAULT 'normal',
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT market_check CHECK (market IN ('TWSE', 'TPEx')),
    CONSTRAINT trading_status_check CHECK (trading_status IN ('normal', 'suspended', 'halted', 'delisted', 'ipo', 'attention')),
    CONSTRAINT outstanding_shares_positive CHECK (outstanding_shares > 0)
);

-- Indexes for stock info
CREATE INDEX IF NOT EXISTS idx_stock_market ON taiwan_stock_info (market);
CREATE INDEX IF NOT EXISTS idx_stock_sector ON taiwan_stock_info (sector);
CREATE INDEX IF NOT EXISTS idx_stock_trading_status ON taiwan_stock_info (trading_status);
CREATE INDEX IF NOT EXISTS idx_stock_listing_date ON taiwan_stock_info (listing_date);

-- Data quality monitoring table
CREATE TABLE IF NOT EXISTS data_quality_log (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20),
    data_type VARCHAR(50),
    check_date DATE NOT NULL,
    issue_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL DEFAULT 'medium',
    description TEXT NOT NULL,
    resolved BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    resolved_at TIMESTAMP WITH TIME ZONE,
    
    -- Constraints
    CONSTRAINT severity_check CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    CONSTRAINT issue_type_check CHECK (issue_type IN ('missing_data', 'price_anomaly', 'volume_spike', 'lag_violation', 'bias_detected'))
);

-- Indexes for data quality monitoring
CREATE INDEX IF NOT EXISTS idx_quality_symbol_date ON data_quality_log (symbol, check_date);
CREATE INDEX IF NOT EXISTS idx_quality_severity ON data_quality_log (severity);
CREATE INDEX IF NOT EXISTS idx_quality_unresolved ON data_quality_log (resolved, created_at) WHERE NOT resolved;

-- Performance monitoring table
CREATE TABLE IF NOT EXISTS query_performance_log (
    id BIGSERIAL PRIMARY KEY,
    query_type VARCHAR(50) NOT NULL,
    symbol_count INTEGER NOT NULL,
    date_range_days INTEGER,
    execution_time_ms DECIMAL(10,3) NOT NULL,
    cache_hit_rate DECIMAL(5,4),
    memory_usage_mb DECIMAL(10,2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT symbol_count_positive CHECK (symbol_count > 0),
    CONSTRAINT execution_time_positive CHECK (execution_time_ms > 0),
    CONSTRAINT cache_hit_rate_valid CHECK (cache_hit_rate >= 0 AND cache_hit_rate <= 1)
);

-- Index for performance monitoring
CREATE INDEX IF NOT EXISTS idx_perf_query_type ON query_performance_log (query_type);
CREATE INDEX IF NOT EXISTS idx_perf_created_at ON query_performance_log (created_at);

-- Functions for common operations

-- Function to get Taiwan settlement date (T+2)
CREATE OR REPLACE FUNCTION get_taiwan_settlement_date(trade_date DATE)
RETURNS DATE AS $$
DECLARE
    settlement_date DATE;
    business_days INTEGER := 0;
BEGIN
    settlement_date := trade_date;
    
    -- Add 2 business days
    WHILE business_days < 2 LOOP
        settlement_date := settlement_date + INTERVAL '1 day';
        
        -- Check if it's a trading day (simplified - Monday to Friday)
        IF EXTRACT(dow FROM settlement_date) BETWEEN 1 AND 5 THEN
            -- Also check if it's not a holiday in settlement_calendar
            IF NOT EXISTS (
                SELECT 1 FROM settlement_calendar 
                WHERE trade_date = settlement_date::DATE 
                AND NOT is_trading_day
            ) THEN
                business_days := business_days + 1;
            END IF;
        END IF;
    END LOOP;
    
    RETURN settlement_date;
END;
$$ LANGUAGE plpgsql;

-- Function to validate temporal consistency
CREATE OR REPLACE FUNCTION validate_temporal_consistency()
RETURNS TABLE(symbol VARCHAR, issue TEXT) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        p.symbol,
        'Future data access detected: ' || p.data_type || ' on ' || p.value_date || ' known on ' || p.as_of_date
    FROM pit_data p
    WHERE p.as_of_date > p.value_date + INTERVAL '365 days'  -- Unreasonable lag
    OR (p.data_type = 'price' AND p.as_of_date > p.value_date + INTERVAL '1 day')  -- Price should be same day
    OR (p.data_type = 'fundamental' AND p.as_of_date < p.value_date + INTERVAL '30 days');  -- Fundamental needs lag
END;
$$ LANGUAGE plpgsql;

-- Function to clean old performance logs (keep last 90 days)
CREATE OR REPLACE FUNCTION cleanup_performance_logs()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM query_performance_log 
    WHERE created_at < NOW() - INTERVAL '90 days';
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Views for common queries

-- View for latest prices
CREATE OR REPLACE VIEW latest_prices AS
SELECT DISTINCT ON (symbol) 
    symbol,
    value_numeric as price,
    value_date,
    as_of_date,
    created_at
FROM pit_data
WHERE data_type = 'price'
ORDER BY symbol, as_of_date DESC, version DESC;

-- View for trading calendar with settlement dates
CREATE OR REPLACE VIEW trading_calendar_extended AS
SELECT 
    trade_date,
    settlement_date,
    get_taiwan_settlement_date(trade_date) as calculated_settlement_date,
    is_trading_day,
    market_session,
    EXTRACT(dow FROM trade_date) as day_of_week,
    notes
FROM settlement_calendar
ORDER BY trade_date;

-- View for data quality summary
CREATE OR REPLACE VIEW data_quality_summary AS
SELECT 
    DATE_TRUNC('day', check_date) as check_date,
    symbol,
    COUNT(*) as total_issues,
    COUNT(*) FILTER (WHERE severity = 'critical') as critical_issues,
    COUNT(*) FILTER (WHERE severity = 'high') as high_issues,
    COUNT(*) FILTER (WHERE NOT resolved) as unresolved_issues
FROM data_quality_log
GROUP BY DATE_TRUNC('day', check_date), symbol
ORDER BY check_date DESC, symbol;

-- Grants (adjust as needed for your security model)
-- GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA ml4t TO ml4t_app_user;
-- GRANT USAGE ON ALL SEQUENCES IN SCHEMA ml4t TO ml4t_app_user;
-- GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA ml4t TO ml4t_app_user;

-- Add comments for documentation
COMMENT ON TABLE pit_data IS 'Point-in-time temporal data storage with look-ahead bias prevention';
COMMENT ON COLUMN pit_data.as_of_date IS 'Date when this data was known/available';
COMMENT ON COLUMN pit_data.value_date IS 'Date that this data refers to';
COMMENT ON COLUMN pit_data.version IS 'Version number for handling data revisions';

COMMENT ON TABLE settlement_calendar IS 'Taiwan market trading calendar with T+2 settlement dates';
COMMENT ON COLUMN settlement_calendar.market_session IS 'Trading session type: morning, closed, holiday, suspended';

COMMENT ON TABLE taiwan_stock_info IS 'Static information about Taiwan stocks';
COMMENT ON COLUMN taiwan_stock_info.market IS 'TWSE for main board, TPEx for OTC';

COMMENT ON FUNCTION get_taiwan_settlement_date(DATE) IS 'Calculate T+2 settlement date for Taiwan market';
COMMENT ON FUNCTION validate_temporal_consistency() IS 'Check for potential look-ahead bias violations';

-- Finalization
ANALYZE pit_data;
ANALYZE settlement_calendar;
ANALYZE taiwan_stock_info;

-- Migration complete
SELECT 'Migration 001 completed successfully' AS status;