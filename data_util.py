from alpaca_clients import data_client
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.trading.requests import GetAssetsRequest
from alpaca.data.timeframe import TimeFrame
from langchain_core.tools import tool
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from alpaca.trading.enums import AssetClass
import requests
from bs4 import BeautifulSoup
import pytz
from config import ALPACA_API_KEY, ALPACA_API_SECRET

from sqlalchemy import func

from database import (
    db_config, HistoricalData, TechnicalIndicators, 
    ModelVersions, Predictions, DatabaseQueries, init_database
)



# DEBUGGING GET ALPACA DATA FUNCTION:
def get_stock_data(ticker:str) -> str:
    """This function gets detailed stock information like price, indicators, the works."""
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = '2025-04-01'

    try:
        data = get_alpaca_data(ticker, start_date, end_date, timescale = "Minute")
        data = add_indicators(data=data, indicator_set = 'alternate')
        return print_dataframe_info(data)
    except Exception as e:
        return f"Error getting detailed stock data: {str(e)}"

def print_dataframe_info(df):
    """Print detailed information about a DataFrame to help debug issues."""
    print(f"DataFrame shape: {df.shape}")
    print(f"DataFrame columns: {df.columns.tolist()}")
    print(f"Index type: {type(df.index)}")
    print(f"Index name: {df.index.name}")
    print(f"First 3 index values: {df.index[:3].tolist()}")
    
    # Sample data
    print("\nSample data (last 3 rows):")
    print(df.tail(3))
    
    # Check for NaN values
    print(f"\nNaN values: {df.isna().sum().sum()}")
    
    # Check data types
    print("\nData types:")
    print(df.dtypes)


# GET DATAFRAME INFO FOR RETRIEVED HISTORICAL DATA
def dataframe_info(df):
    """Print detailed information about a DataFrame to help debug issues."""
    return f"DataFrame shape: {df.shape}\nDataFrame columns: {df.columns.tolist()}\nSample data (last 3 rows):\n{df.tail(3)}"


# SCRAPE FINVIZ TO GET FLOAT
def scrape_float(ticker):
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch Finviz page for {ticker} (status code: {response.status_code})")

    soup = BeautifulSoup(response.text, 'html.parser')
    data = {}
    
    try:
        # Find all table cells
        cells = soup.find_all("td", class_="snapshot-td2")

        # Go through cells in pairs: [label, value, label, value...]
        for i in range(0, len(cells)-1, 2):
            label = cells[i].text.strip()
            value = cells[i+1].text.strip()
            data[label] = value

        # Pull and convert float-related fields
        float_data = {
            "Float": parse_number(data.get("Shs Float")),
            "Float Short %": parse_percentage(data.get("Short Float / Ratio", "").split('/')[0]),
            "Shares Outstanding": parse_number(data.get("Shs Outstand"))
        }

        return float_data

    except Exception as e:
        raise Exception(f"Failed to parse float info for {ticker}: {e}")

# PARSE NUMBER AND PERCENTAGE HELPERS FOR SCRAPE FLOAT
def parse_number(val):
    if not val:
        return None
    val = val.replace('B', 'e9').replace('M', 'e6').replace(',', '')
    try:
        return float(eval(val))
    except:
        return None

def parse_percentage(val):
    try:
        return float(val.replace('%', '').strip())
    except:
        return None

# ADD INDICATOR FUNCTION FOR HISTORICAL DATA RETRIEVAL: 
def add_indicators(data, indicator_set='default', store_in_db=True, ticker=None):
    """
    Add technical indicators to the dataset using native pandas calculations
    where possible to avoid dependency issues.
    
    Args:
        data: DataFrame with stock price data
        indicator_set: 'default' or 'alternate' to determine which indicators to calculate
        
    Returns:
        DataFrame with added technical indicators
    """
    if indicator_set == 'default':
        # RSI calculation
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=15).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=15).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))

        # EMA calculations
        data['EMAF'] = data['Close'].ewm(span=20, adjust=False).mean()

        # Historical Volatility
        data['hist_volatility'] = data['Adj Close'].pct_change().rolling(window=20).std() * np.sqrt(252)

        # Bollinger Bands
        rolling_mean = data['Close'].rolling(window=20).mean()
        rolling_std = data['Close'].rolling(window=20).std()
        data['BBL_20_2.0'] = rolling_mean - (2 * rolling_std)
        data['BBM_20_2.0'] = rolling_mean
        data['BBU_20_2.0'] = rolling_mean + (2 * rolling_std)
        data['BB_width'] = (data['BBU_20_2.0'] - data['BBL_20_2.0']) / data['BBM_20_2.0']

        # MACD
        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = exp1 - exp2
        data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

        # ATR
        high_low = data['High'] - data['Low']
        high_close = abs(data['High'] - data['Close'].shift())
        low_close = abs(data['Low'] - data['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        data['ATR'] = true_range.rolling(14).mean()

        # OBV
        data['OBV'] = (np.sign(data['Close'].diff()) * data['Volume']).fillna(0).cumsum()

    elif indicator_set == 'alternate':
        # Simple Moving Averages
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['SMA_200'] = data['Close'].rolling(window=200).mean()
        
        # RSI (Relative Strength Index)
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD (Moving Average Convergence Divergence)
        ema_12 = data['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = ema_12 - ema_26
        data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        data['Middle_Band'] = data['Close'].rolling(window=20).mean()
        std_dev = data['Close'].rolling(window=20).std()
        data['Upper_Band'] = data['Middle_Band'] + (std_dev * 2)
        data['Lower_Band'] = data['Middle_Band'] - (std_dev * 2)
        
        # Add EMA
        data['EMA'] = data['Close'].ewm(span=50, adjust=False).mean()
    
    else:
        raise ValueError("Invalid indicator set. Use 'default' or 'alternate'.")

    if store_in_db and ticker:
        _update_indicators_in_database(ticker, data)
    
    return data

def get_alpaca_data(ticker, start_date=None, end_date=None, is_paper=True, timescale = "Minute", store_in_db = True):
    """
    Retrieve historical stock data from Alpaca API with debugging.
    """

    api_call_text = f"""\n ALPCA API CALL:
    Ticker: {ticker}
    Period: {start_date} to {end_date}
    Timescale: {timescale}
    Store in DB: {store_in_db}"""
    print(api_call_text)

    try:
        # Convert date strings to datetime
        eastern  = pytz.timezone('US/Eastern')
        if start_date is None:
            start = eastern.localize(datetime.now() - timedelta(minutes=5))
        else:
            start = datetime.strptime(start_date, '%Y-%m-%d')
            start = eastern.localize(datetime.combine(start.date(), datetime.min.time()))

        if end_date is None:
            end = eastern.localize(datetime.now())
        else:
            end = datetime.strptime(end_date, '%Y-%m-%d')
            end = eastern.localize(datetime.combine(end.date(), datetime.max.time()))
            # if end date is today, adjust
            if end.date() == datetime.now().date():
                end = eastern.localize(datetime.now())


        print(f"GETTING DATA FROM {start} to {end}")
        # Initialize Alpaca data client
        match timescale:
            case "Minute":
                unit = TimeFrame.Minute
            case "Hour":
                unit = TimeFrame.Hour
            case "Day":
                unit = TimeFrame.Day
            case _:
                raise ValueError("Invalid timescale")

        # Fetch historical bars
        request_params = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=unit,
            start=start,
            end=end,
            feed='iex'
        )

        print(f"Fetching data for {ticker}")
        bars = data_client.get_stock_bars(request_params)

        if not hasattr(bars, 'df') or bars.df.empty:
            print("Retrieved empty DataFrame")
            return None

        print(f"Type of bars: {type(bars)}")

        # Extract data for the specific ticker
        df = bars.df.loc[ticker].copy()
        df.index = pd.to_datetime(df.index)  # ensure timestamp index
        df.index.name = "Date"  # rename index for clarity

        # Rename columns
        column_mapping = {
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume',
        }
        df.rename(columns=column_mapping, inplace=True)

        # Add Adj Close if not present
        if 'Adj Close' not in df.columns and 'Close' in df.columns:
            df['Adj Close'] = df['Close']

        print(f"Data shape: {df.shape}")
        print(f"DataFrame columns: {df.columns.tolist()}")
        print(f"Date range in data: {df.index.min()} to {df.index.max()}")

        if store_in_db and df is not None and not df.empty:
            print(f"    📂 DIRECT STORAGE FROM GET_ALPACA_DATA, Storing {len(df)} records to database...")
            _store_dataframe_in_database(ticker, df)
        
        return df

    except Exception as e:
        print(f"Error retrieving alpaca data: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    

def _store_dataframe_in_database(ticker, df):
    """Store df in db"""
    with db_config.get_db_session() as session:

        stored_count = 0
        duplicate_count = 0

        for timestamp, row in df.iterrows():
            existing = session.query(HistoricalData).filter(
                HistoricalData.ticker == ticker,
                HistoricalData.timestamp == timestamp
            ).first()

            if not existing:
                historical_data = HistoricalData(
                    ticker = ticker,
                    timestamp = timestamp,
                    date = timestamp.date(),
                    open=row['Open'],
                    high=row['High'],
                    low=row['Low'],
                    close=row['Close'],
                    adjusted_close=row['Adj Close'],
                    trade_count = row['trade_count'],
                    volume=row['Volume'],
                    vwap = row['vwap']
                )
                session.add(historical_data)
                stored_count += 1
            else:
                duplicate_count += 1
        print(f"    📂 Stored {stored_count} new records, skipped {duplicate_count} duplicates")


def _update_indicators_in_database(ticker, df):
    with db_config.get_db_session() as session:
        for timestamp, row in df.iterrows():
            historical_record = session.query(HistoricalData).filter(
                HistoricalData.ticker == ticker,
                HistoricalData.timestamp == timestamp
            ).first()

            if historical_record:
                # update or create indicators
                indicators = TechnicalIndicators(
                    data_id = historical_record.data_id,
                    sma_20 = row.get('SMA_20'),
                    sma_50 = row.get('SMA_50'),
                    rsi = row.get('RSI'),
                    macd = row.get('MACD'),
                    signal_line=row.get('Signal_Line'),
                    middle_band = row.get('Middle_Band'),
                    upper_band = row.get('Upper_Band'),
                    lower_band = row.get('Lower_Band'),
                    ema = row.get('EMA')
                )
                session.add(indicators)


def check_database_status(ticker, start_date, end_date):
    """Check and report db status"""
    with db_config.get_db_session() as session:
        total_records = session.query(HistoricalData).filter(
            HistoricalData.ticker == ticker
        ).count()

    period_records = session.query(HistoricalData).filter(
        HistoricalData.ticker == ticker,
        HistoricalData.date >= start_date,
        HistoricalData.date <= end_date
    ).count()

    print(f"\n📊 DATABASE STATUS:")
    print(f"    Total {ticker} records: {total_records}")
    print(f"    Records in simulation period: {period_records}")

    if total_records > 0:
        latest = DatabaseQueries.get_latest_data_point(session, ticker)
        print(f"    Latest Data Point: {latest.timestamp}")


def get_current_quote(ticker: str) -> str:
    """Get current quote for a ticker (price and volume)."""
    try:
        req = StockLatestQuoteRequest(symbol_or_symbols=[ticker])
        quote = data_client.get_stock_latest_quote(req)
        ask_price = quote[ticker].ask_price
        bid_price = quote[ticker].bid_price
        return f"{ticker} current price is ${ask_price:.2f} (Ask), ${bid_price:.2f} (Bid)"
    except Exception as e:
        return f"Error getting quote for {ticker}: {str(e)}"
    

def get_stock_price(symbol: str) -> str:
    """Fetch the current ask and bid price of a stock from Alpaca."""
    symbol_str = symbol.upper()
    symbol = [str(symbol_str)]
    
    try:
        request_param = StockLatestQuoteRequest(symbol_or_symbols=symbol)
        quote = data_client.get_stock_latest_quote(request_param)
        return f"{symbol} ask price: {quote[symbol_str].ask_price}, bid price: {quote[symbol_str].bid_price}"
    except Exception as e:
        return f"Price fetch failed: {str(e)}"
    

def remove_duplicate_records(ticker):
    """Clean database for duplicates"""
    with db_config.get_db_session() as session:
        # find dups
        duplicates = session.query(
            HistoricalData.ticker,
            HistoricalData.timestamp,
            func.count(HistoricalData.data_id).label('count')
        ).group_by(
            HistoricalData.ticker,
            HistoricalData.timestamp
        ).having(func.count(HistoricalData.data_id) > 1).all()

        for dup in duplicates:
            records = session.query(HistoricalData).filter(
                HistoricalData.ticker == dup.ticker,
                HistoricalData.timestamp == dup.timestamp
            ).order_by(HistoricalData.data_id).all()

            for record in records[1:]:
                session.delete(record)