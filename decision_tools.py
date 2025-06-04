from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.trading.requests import GetAssetsRequest
from alpaca.data.timeframe import TimeFrame
from langchain_core.tools import tool
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from alpaca_clients import trading_client, data_client
from alpaca.trading.enums import AssetClass
import requests
from bs4 import BeautifulSoup

from data_util import get_alpaca_data, add_indicators, dataframe_info

#==================== G E M I N I +++++++ T O O L S E T ======================================#


# Account function to see account balance I believe. 
@tool
def get_account() -> str:
    """Get all account info including buying power from Alpaca API"""
    # search for stock assets
    try:
        account_info = trading_client.get_account()
        return f"Account Info: {account_info}"
        
    except Exception as e:
        return f"Get account info failed: {str(e)}"

@tool
def place_market_BUY(symbol: str, qty: int, side: str) -> str:
    """Place a paper trade market BUY using Alpaca."""
    try:
        current_price = get_stock_price(symbol)
        market_order_data = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY
                    )
        BUY_order = trading_client.submit_order(
                    order_data=market_order_data
                   )
        return f"Order placed: {BUY_order.id} - {side.upper()} {qty} {symbol} @ {current_price}"
    except Exception as e:
        return f"Trade failed: {str(e)}"

@tool
def place_market_SELL(symbol: str, qty: int, side: str) -> str:
    """Place a paper trade market SELL using Alpaca."""
    try:
        current_price = get_stock_price(symbol)
        market_order_data = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY
                    )
        SELL_order = trading_client.submit_order(
                    order_data=market_order_data
                   )
        return f"Order placed: {SELL_order.id} - {side.upper()} {qty} {symbol} @ {current_price}"
    except Exception as e:
        return f"Trade failed: {str(e)}"

@tool
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


# 📊 Relative Volume
@tool
def get_rvol(ticker: str) -> str:
    """Returns the current RVOL (relative volume = today's volume / average past volume).
    Average volume is based on thirty day average"""
    try:
        start_date = datetime.now() - timedelta(days=30)
        start_date = str(start_date.date())
        end_date = datetime.now().strftime('%Y-%m-%d')
        data = get_alpaca_data(ticker=ticker, start_date=start_date, end_date=end_date, timescale="Minute")
        today_volume = data.iloc[-1]["Volume"]
        print(f"today's volume: {data.iloc[-1]} ", today_volume)
        avg_volume = data.iloc[:-1]["Volume"].mean()
        print(f"Average volume: ", avg_volume)
        rvol = today_volume / avg_volume
        return f"RVOL for {ticker}: {rvol:.2f} (Current volume: {today_volume}, Avg volume: {int(avg_volume)})"
    except Exception as e:
        return f"Error computing RVOL for {ticker}: {str(e)}"

# 📈 Price Change %
@tool
def get_price_change(ticker: str) -> str:
    """Returns % change between today's open and current price."""
    try:
        start_date = datetime.now() - timedelta(days=5)
        start_date = str(start_date.date())
        end_date = datetime.now().strftime('%Y-%m-%d')
        data = get_alpaca_data(ticker=ticker, start_date=start_date, end_date=end_date, timescale="Minute")
        open_price = data.iloc[0]["Open"]
        last_price = data.iloc[-1]["Close"]
        change_pct = ((last_price - open_price) / open_price) * 100
        return f"{ticker} is up {change_pct:.2f}% today (from ${open_price:.2f} to ${last_price:.2f})"
    except Exception as e:
        return f"Error getting price change for {ticker}: {str(e)}"

# 💵 Quote (price + volume)
@tool
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


@tool
def get_company_overview(ticker: str)-> str:
    """
    Retrieve company overview data from Alpha Vantage.
    
    Args:
        ticker: Stock symbol        
    Returns:
        Dictionary with company overview data or None if an error occurs
    """
    api_key = ALPHA_VANTAGE_API_KEY
    if not api_key:
        raise ValueError("Alpha Vantage API key required")
    
    try:
        # Construct URL for OVERVIEW endpoint
        url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={api_key}"
        
        print(f"Fetching company overview for {ticker} from Alpha Vantage...")
        
        response = requests.get(url)
        data = response.json()
        
        # Check for error messages or empty response
        if "Error Message" in data:
            #print(f"Alpha Vantage API error: {data['Error Message']}")
            return f"Alpha Vantage API error: {data['Error Message']}"
        elif "Note" in data and "API call frequency" in data["Note"]:
            #print(f"Alpha Vantage API rate limit exceeded: {data['Note']}")
            return f"Alpha Vantage API rate limit exceeded: {data['Note']}"
        elif not data or len(data) <= 1:
            #print(f"No overview data found for {ticker}")
            return f"No overview data found for {ticker}"
        
        print(f"Successfully retrieved company overview for {ticker}")
        return f"Successfully retrieved company overview for {ticker} \nData: {str(data)}"
        
    except Exception as e:
        #print(f"Error retrieving company overview: {str(e)}")
        return f"Error retrieving company overview: {str(e)}"


@tool
def get_float_info(ticker:str) -> str: 
    """This should retrieve the float of a stock. Uses finviz, some tickers can't give float"""
    try:
        info = scrape_float(ticker)
        if info["Float"]:
            return f"{info}"
        else:
            return f"Float data missing."
    except Exception as e:
        print(f"Error fetching float: {e}")



@tool
def get_detailed_stock_data(ticker:str, timescale:str) -> str:
    """This function gets detailed stock information like price, and indicators. 
    This gets the last year of historical data at a specific timeframe.
    Use timescale to get different timeframes on historical data
    Somehow it works. timescale accepts "Day", "Hour", and "Minute". 
    """
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = '2024-01-01'

    try:
        data = get_alpaca_data(ticker=ticker, start_date=start_date, end_date=end_date, timescale=timescale)
        data = add_indicators(data=data, indicator_set = 'alternate')
        info = dataframe_info(data)
        return f"{info}"
    except Exception as e:
        return f"Error getting detailed stock data: {str(e)}"


@tool
def real_check_news(ticker: str, days_back: int = 7) -> str:
    """
    Searches for real news articles about a specific ticker.
    This function can be used in real-time trading.
    
    Args:
        ticker: Stock symbol to search news for
        days_back: Number of days to look back for news
    
    Returns:
        Recent news articles about the ticker
    """
    try:
        # This is a placeholder for how the real function would work.
        # In a real implementation, you would make API calls to news services like:
        # - Alpha Vantage NEWS_SENTIMENT endpoint
        # - Finnhub news API
        # - Polygon.io news API
        # - Or scrape financial news sites with proper rate limiting
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            return f"Failed to fetch news for {ticker} (status code: {response.status_code})"
        
        soup = BeautifulSoup(response.text, 'html.parser')
        news_table = soup.find(id='news-table')
        
        if not news_table:
            return f"No news table found for {ticker}"
        
        news_items = []
        rows = news_table.findAll('tr')
        
        for row in rows:
            cells = row.findAll('td')
            if len(cells) == 2:
                date_cell = cells[0].text.strip()
                headline = cells[1].text.strip()
                link = cells[1].a['href'] if cells[1].a else ""
                
                # Parse date
                date_pattern = re.compile(r'(\d{1,2}-\d{1,2}-\d{2}|\d{1,2}:\d{1,2})')
                date_match = date_pattern.search(date_cell)
                date = date_match.group(1) if date_match else "Unknown"
                
                news_items.append({
                    "date": date,
                    "headline": headline,
                    "link": link
                })
        
        # Format the response
        if not news_items:
            return f"No recent news found for {ticker}"
        
        formatted_news = f"Recent news for {ticker}:\n\n"
        for i, news in enumerate(news_items[:10], 1):  # Limit to 10 items
            formatted_news += f"{i}. {news['headline']}\n"
            formatted_news += f"   Date: {news['date']}\n"
            formatted_news += f"   Link: {news['link']}\n\n"
        
        return formatted_news
    
    except Exception as e:
        return f"Error fetching news for {ticker}: {str(e)}\n\nLikely an issue with scraping the website."