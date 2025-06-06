from sqlalchemy import Column, Integer, String, DECIMAL, BigInteger, DateTime, Date, Boolean, ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
import uuid
import json

from .config import Base


class HistoricalData(Base):
    __tablename__ = 'historical_data'

    data_id = Column(Integer, primary_key=True)
    ticker = Column(String(10), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable = False)
    date = Column(Date, nullable=False)
    open = Column(DECIMAL(12,4), nullable=False)
    high = Column(DECIMAL(12, 4), nullable = False)
    low = Column(DECIMAL(12,4), nullable = False)
    close = Column(DECIMAL(12,4), nullable = False)
    adjusted_close = Column(DECIMAL(12,4))
    volume = Column(BigInteger, nullable = False)
    vwap = Column(DECIMAL(12,4))
    trade_count = Column(BigInteger, nullable=False)
    last_updated = Column(DateTime(timezone=True), default=func.current_timestamp())

    indicators = relationship("TechnicalIndicators", back_populates='historical_data', cascade='all, delete-orphan')

    def __repr__(self):
        return f"<HistoricalData(ticker='{self.ticker}', timestamp='{self.timestamp}', close={self.close})>"
    
class TechnicalIndicators(Base):
    __tablename__ = 'technical_indicators'

    indicator_id = Column(Integer, primary_key = True)
    data_id = Column(Integer, ForeignKey('historical_data.data_id', ondelete = 'CASCADE'), nullable=False)

    sma_20 = Column(DECIMAL(12,4))
    sma_50 = Column(DECIMAL(12,4))
    sma_200 = Column(DECIMAL(12,4))

    rsi = Column(DECIMAL(5,2))
    macd = Column(DECIMAL(12,6))
    signal_line = Column(DECIMAL(12,6))


    middle_band = Column(DECIMAL(12,4))
    upper_band = Column(DECIMAL(12,4))
    lower_band = Column(DECIMAL(12,4))

    ema = Column(DECIMAL(12, 4))


    calculated_at = Column(DateTime(timezone=True), default=func.current_timestamp())


    historical_data = relationship("HistoricalData", back_populates = "indicators")

    def __repr__(self):
        return f"<TechnicalIndicators(data_id = {self.data_id}, rsi={self.rsi}, macd={self.macd})>"
    
class ModelVersions(Base):
    __tablename__ = 'model_versions'

    model_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    version = Column(String(50), nullable=False)
    created_at = Column(DateTime(timezone=True), default = func.current_timestamp())
    parameters = Column(Text, nullable=False)
    metrics = Column(Text)
    is_active = Column(Boolean, default=True)

    # relationship for predictions
    predictions = relationship("Predictions", back_populates = 'model')

    def set_parameters(self, params_dict):
        self.parameters = json.dumps(params_dict)

    def get_parameters(self):
        return json.loads(self.parameters) if self.parameters else {}

    def __reper__(self):
        return f"<ModelVersions(version='{self.versions}', created_at='{self.created_at}', is_active={self.is_active})>"
    
class Predictions(Base):
    __tablename__ = 'predictions'
    prediction_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    model_id = Column(String(36), ForeignKey('model_versions.model_id'), nullable=False)
    ticker = Column(String(10), nullable=False)
    prediction_date = Column(DateTime(timezone=True), default=func.current_timestamp())
    target_date = Column(DateTime(timezone=True), nullable=False)
    predicted_value = Column(DECIMAL(12,4), nullable=False)
    actual_value = Column(DECIMAL(12,4))
    confidence_score = Column(DECIMAL(5,2))

    model = relationship("ModelVersions", back_populates='predictions')

    def __repr__(self):
        return f"<Predictions(ticker='{self.ticker}', predicted_value={self.predicted_value}, target_date='{self.target_date}')>"
    

class DatabaseQueries:
    @staticmethod
    def get_historical_data_range(session, ticker, start_date, end_date):
        return session.query(HistoricalData).filter(
            HistoricalData.ticker == ticker,
            HistoricalData.date >= start_date,
            HistoricalData.date <= end_date
        ).order_by(HistoricalData.timestamp).all()
    
    @staticmethod
    def get_latest_data_point(session, ticker):
        return session.query(HistoricalData).filter(
            HistoricalData.ticker == ticker
        ).order_by(HistoricalData.timestamp.desc()).first()
    
    @staticmethod
    def get_data_with_indicators(session, ticker, start_date, end_date):
        return session.query(HistoricalData).join(
            TechnicalIndicators
        ).filter(
            HistoricalData.ticker == ticker,
            HistoricalData.date >= start_date,
            HistoricalData.date <= end_date
        ).order_by(HistoricalData.timestamp).all()
    
    @staticmethod
    def get_active_model(session):
        return session.query(ModelVersions).filter(
            ModelVersions.is_active == True
        ).order_by(ModelVersions.created_at.desc()).first()
    
    @staticmethod
    def store_predictions(session, model_id, ticker, predictions_data):
        prediction_objects = []
        for pred_data in predictions_data:
            prediction = Predictions(
                model_id=model_id,
                ticker=ticker,
                target_date=pred_data['target_date'],
                predicted_value=pred_data['predicted_value'],
                confidence_score = pred_data.get('confidence_score')
            )
            prediction_objects.append(prediction)

        session.add_all(prediction_objects)
        return prediction_objects