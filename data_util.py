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
from config import ALPACA_API_KEY, ALPACA_API_SECRET



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
def add_indicators(data, indicator_set='default'):
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

    return data

def get_alpaca_data(ticker, start_date, end_date, is_paper=True, timescale = "Day"):
    """
    Retrieve historical stock data from Alpaca API with debugging.
    """

    try:
        # Convert date strings to datetime
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d') if end_date else datetime.now()
        start = datetime.combine(start.date(), datetime.min.time())
        end = datetime.combine(end.date(), datetime.max.time())
        # if end date is today, adjust
        if end.date() == datetime.now().date():
            end = datetime.now() - timedelta(minutes=15) # get data on a 15 minute delay if its today.

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
            end=end
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

        return df

    except Exception as e:
        print(f"Error retrieving alpaca data: {str(e)}")
        import traceback
        traceback.print_exc()
        return None