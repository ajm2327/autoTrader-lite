from .config import db_config, get_db, Base
from .db_models import HistoricalData, TechnicalIndicators, ModelVersions, Predictions, DatabaseQueries

__all__ = [
    'db_config',
    'get_db',
    'Base',
    'HistoricalData',
    'TechnicalIndicators',
    'ModelVersions',
    'Predictions',
    'DatabaseQueries'
]

def init_database():
    db_config.init_db()
    print("Database initialized successfully")

def create_tables():
    Base.metadata.create_all(db_config.engine)
    print("All tables created successfully")

    