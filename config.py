
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
    ALPACA_API_SECRET = os.getenv('ALPACA_API_SECRET')
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
    YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')

# Export as module-level variables for easier importing
ALPACA_API_KEY = Config.ALPACA_API_KEY
ALPACA_API_SECRET = Config.ALPACA_API_SECRET
GOOGLE_API_KEY = Config.GOOGLE_API_KEY
ALPHA_VANTAGE_API_KEY = Config.ALPHA_VANTAGE_API_KEY
YOUTUBE_API_KEY = Config.YOUTUBE_API_KEY