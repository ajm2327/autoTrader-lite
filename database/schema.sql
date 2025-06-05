CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE historical_data (
    data_id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    date DATE NOT NULL,
    open DECIMAL(12,4) NOT NULL,
    high DECIMAL(12,4) NOT NULL,
    low DECIMAL(12,4) NOT NULL,
    close DECIMAL(12,4) NOT NULL,
    adjusted_close DECIMAL(12,4),
    volume BIGINT NOT NULL,
    vwap DECIMAL(12,4),
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ticker, timestamp)
);

CREATE TABLE technical_indicators (
    indicator_id SERIAL PRIMARY KEY,
    data_id INTEGER REFERENCES historical_data(data_id) ON DELETE CASCADE,
    
    sma_20 DECIMAL(12,4),
    sma_50 DECIMAL(12,4),
    sma_200 DECIMAL(12,4),
    rsi DECIMAL(5,2),
    macd DECIMAL(12,6),
    signal_line DECIMAL(12,6),
    middle_band DECIMAL(12,4),
    upper_band DECIMAL(12,4),
    lower_band DECIMAL(12,4),
    ema DECIMAL(12,4),
    emaf DECIMAL(12,4),
    hist_volatility DECIMAL(10,6),
    bb_width DECIMAL(10,6),
    atr DECIMAL(12,6),
    obv BIGINT,
    
    calculated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_historical_data FOREIGN KEY(data_id) REFERENCES historical_data(data_id)
);

CREATE TABLE model_versions (
    model_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    version VARCHAR(50) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    parameters JSONB NOT NULL,
    metrics JSONB,
    is_active BOOLEAN DEFAULT true
);

CREATE TABLE predictions (
    prediction_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID REFERENCES model_versions(model_id),
    ticker VARCHAR(10) NOT NULL,
    prediction_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    target_date TIMESTAMP WITH TIME ZONE NOT NULL,
    predicted_value DECIMAL(12,4) NOT NULL,
    actual_value DECIMAL(12,4),
    confidence_score DECIMAL(5,2),
    CONSTRAINT fk_model FOREIGN KEY(model_id) REFERENCES model_versions(model_id)
);

CREATE INDEX idx_historical_data_ticker_timestamp ON historical_data(ticker, timestamp);
CREATE INDEX idx_historical_data_ticker_date ON historical_data(ticker, date);
CREATE INDEX idx_predictions_model_ticker ON predictions(model_id, ticker);
CREATE INDEX idx_predictions_target_date ON predictions(target_date);