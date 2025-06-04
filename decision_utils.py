from datetime import datetime
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import json
import os
from langchain_core.tools import tool
from langchain_core.tools import tool
from langchain.agents import tool
import traceback

from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import ToolNode

from alpaca_clients import llm, get_llm_with_tools, get_tool_node





GEMINI_TRADER_SYSINT = (
    "system",
    """
# TRADING STRATEGY: MICRO PULLBACK

The Micro Pullback strategy identifies stocks with strong upward momentum that experience a brief consolidation before continuing their upward trend. This strategy requires precision timing and disciplined execution.

## ENTRY CRITERIA - ONLY ENTER WHEN ALL CONDITIONS ARE MET:

PRIMARY MARKET CONDITIONS:
- ⏰ Trading hours are between 7am-11am Eastern Time
- 🌡️ Overall market conditions support momentum trading

PRIMARY STOCK FUNDAMENTALS:
- 💵 Price range: $1-$20 per share
- 🔁 Float size: <10M shares (or <20M if market is hot)
- 📰 News catalyst exists and is driving current momentum

PRIMARY TECHNICAL INDICATORS:
- 📈 Stock is up at least 10% from previous day's close
- 🔥 Relative volume (RVOL) is ≥5x average daily volume
- 📊 Chart pattern shows clear momentum with a micro pullback

## MICRO PULLBACK PATTERN IDENTIFICATION:

1. MOMENTUM PHASE:
   - Stock shows strong upward price movement (≥1% increase in recent candles)
   - Volume increases during price advances
   - Price approaches or breaks through significant levels

2. PULLBACK PHASE:
   - Small red candle or lower wick appears after upward movement
   - Volume typically decreases during pullback
   - Price remains above key support levels
   - Pullback depth is proportional to prior advance (typically 20-30%)

3. CONTINUATION SIGNAL:
   - Volume increases as price begins moving up again
   - Price breaks above the high of the pullback candle
   - Additional confirmations: MACD bullish, RSI not overbought

## EXIT CRITERIA - ONLY EXIT WHEN ANY CONDITION IS MET:

- 🚨 Price drops below the low of the pullback (stop loss hit)
- 📉 MACD crosses below signal line with increasing momentum
- 📊 Volume is significantly fading during attempted breakout
- 🕯️ Formation of a rejection candle (long upper wick)
- 🎯 Price reaches next half/whole dollar level with profit target (≥10%)

## RISK MANAGEMENT:

- Position sizing: 5% of current account value per trade
- Daily stop loss: Cease trading after 10% account drawdown
- Profit target: 10% gain per trade, or next half/whole dollar level
- Stop loss: Low of the pullback candle
- Risk/reward: Minimum 1:2 ratio required for entry

## DECISION COMMUNICATION FORMAT:

For each update, provide:
1. DECISION: [Enter Trade / Hold / Exit Trade]
2. CONFIDENCE: [Low / Medium / High]
3. REASONING: Brief explanation focused on key technical factors
4. ACTION PLAN: Specific entry/exit price levels and position size

Always maintain disciplined adherence to entry/exit criteria and risk management rules. Respond concisely and decisively based on the data provided.
"""
)
# Mock News Database - this will store predefined news for different tickers
MOCK_NEWS_DB = {
    "AMD": [
        {
            "title": "AMD Announces New Chip Architecture",
            "summary": "AMD revealed its next-generation chip architecture with 30% performance gains.",
            "date": "2025-04-15T09:30:00",
            "source": "TechCrunch",
            "sentiment": "positive"
        },
        {
            "title": "AMD Beats Quarterly Expectations",
            "summary": "AMD reported earnings above analyst estimates with strong data center growth.",
            "date": "2025-04-10T16:45:00",
            "source": "CNBC",
            "sentiment": "positive"
        }
    ],
    "NVDA": [
        {
            "title": "NVIDIA Partners with Leading AI Startups",
            "summary": "NVIDIA announced partnerships with 5 emerging AI companies.",
            "date": "2025-04-16T11:20:00",
            "source": "VentureBeat",
            "sentiment": "positive"
        }
    ],
    "AAPL": [
        {
            "title": "Apple Delays Next iPhone Launch",
            "summary": "Apple's next flagship iPhone may be delayed due to supply chain issues.",
            "date": "2025-04-14T08:15:00",
            "source": "Bloomberg",
            "sentiment": "negative"
        }
    ],
    "DEFAULT": [
        {
            "title": "No Significant News",
            "summary": "No major news events for this ticker in the recent period.",
            "date": "2025-04-15T12:00:00",
            "source": "Market Watch",
            "sentiment": "neutral"
        }
    ]
}

# Class to handle mock orders and positions for simulation
class MockTradeManager:
    def __init__(self, log_dir="simulation_logs"):
        self.positions = {}
        self.orders = []
        self.next_order_id = 1000
        self.log_dir = log_dir
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        self.mock_trades_file = f"{log_dir}/mock_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self._initialize_log_file()
    
    def _initialize_log_file(self):
        """Initialize empty log file"""
        with open(self.mock_trades_file, 'w') as f:
            json.dump({"positions": {}, "orders": []}, f)
        print(f"📝 Mock trades log initialized at {self.mock_trades_file}")
    
    def place_buy_order(self, ticker, qty, current_price, timestamp):
        """Place a mock buy order"""
        order_id = f"mock-order-{self.next_order_id}"
        self.next_order_id += 1
        
        order = {
            "id": order_id,
            "ticker": ticker,
            "side": "BUY",
            "qty": qty,
            "price": current_price,
            "timestamp": timestamp,
            "status": "filled"  # Always assume it's filled in the simulation
        }
        
        self.orders.append(order)
        
        # Update position
        if ticker in self.positions:
            self.positions[ticker]["qty"] += qty
            self.positions[ticker]["avg_price"] = (
                (self.positions[ticker]["avg_price"] * self.positions[ticker]["qty"] + current_price * qty) / 
                (self.positions[ticker]["qty"] + qty)
            )
        else:
            self.positions[ticker] = {
                "qty": qty,
                "avg_price": current_price,
                "timestamp": timestamp
            }
        
        # Log the trade
        self._log_trade()
        
        return order
    
    def place_sell_order(self, ticker, qty, current_price, timestamp):
        """Place a mock sell order"""
        order_id = f"mock-order-{self.next_order_id}"
        self.next_order_id += 1
        
        order = {
            "id": order_id,
            "ticker": ticker,
            "side": "SELL",
            "qty": qty,
            "price": current_price,
            "timestamp": timestamp,
            "status": "filled"  # Always assume it's filled in the simulation
        }
        
        self.orders.append(order)
        
        # Update position
        if ticker in self.positions and self.positions[ticker]["qty"] >= qty:
            self.positions[ticker]["qty"] -= qty
            if self.positions[ticker]["qty"] == 0:
                del self.positions[ticker]
        else:
            # This would be a short position, but we'll just log it
            print(f"⚠️ Attempted to sell {qty} shares of {ticker} but have insufficient position.")
        
        # Log the trade
        self._log_trade()
        
        return order
    
    def get_position(self, ticker):
        """Get current position for a ticker"""
        if ticker in self.positions:
            return self.positions[ticker]
        return None
    
    def _log_trade(self):
        """Log trades to file"""
        try:
            with open(self.mock_trades_file, 'w') as f:
                json.dump({
                    "positions": self.positions,
                    "orders": self.orders
                }, f, indent=2)
        except Exception as e:
            print(f"Error logging mock trade: {str(e)}")



@tool
def mock_place_market_BUY(symbol: str, qty: int) -> str:
    """
    Simulates placing a market BUY order in the historical simulation.
    
    Args:
        symbol: Stock symbol to buy
        qty: Quantity of shares to buy
    
    Returns:
        Confirmation message with order details
    """
    try:
        # Get current simulated price from the simulator's current data point
        # This requires access to the simulator object from the state
        from inspect import currentframe, getouterframes
        
        # Try to get the simulator from the caller's context
        frame = currentframe()
        while frame:
            if 'state' in frame.f_locals and 'simulator' in frame.f_locals['state']:
                simulator = frame.f_locals['state']['simulator']
                break
            frame = frame.f_back
        
        if not frame:
            # Fallback method if we can't get the simulator directly
            # This assumes the function is called from within a context where 'simulator' is available
            import inspect
            caller_locals = inspect.currentframe().f_back.f_locals
            
            if 'state' in caller_locals and 'simulator' in caller_locals['state']:
                simulator = caller_locals['state']['simulator']
            else:
                return "Error: Unable to access simulator state"
        
        # Get current price from simulator
        current_index = min(simulator.current_index - 1, len(simulator.data) - 1)
        current_price = simulator.data.iloc[current_index]['Close']
        current_time = simulator.data.index[current_index]
        
        # Place mock order
        order = mock_trade_manager.place_buy_order(symbol, qty, current_price, str(current_time))
        
        return f"Simulated BUY order placed: {order['id']} - BUY {qty} {symbol} @ ${current_price:.2f}"
    
    except Exception as e:
        return f"Mock BUY trade failed: {str(e)}"

@tool
def mock_place_market_SELL(symbol: str, qty: int) -> str:
    """
    Simulates placing a market SELL order in the historical simulation.
    
    Args:
        symbol: Stock symbol to sell
        qty: Quantity of shares to sell
    
    Returns:
        Confirmation message with order details
    """
    try:
        # Get current simulated price from the simulator's current data point
        from inspect import currentframe, getouterframes
        
        # Try to get the simulator from the caller's context
        frame = currentframe()
        while frame:
            if 'state' in frame.f_locals and 'simulator' in frame.f_locals['state']:
                simulator = frame.f_locals['state']['simulator']
                break
            frame = frame.f_back
        
        if not frame:
            # Fallback method if we can't get the simulator directly
            import inspect
            caller_locals = inspect.currentframe().f_back.f_locals
            
            if 'state' in caller_locals and 'simulator' in caller_locals['state']:
                simulator = caller_locals['state']['simulator']
            else:
                return "Error: Unable to access simulator state"
        
        # Get current price from simulator
        current_index = min(simulator.current_index - 1, len(simulator.data) - 1)
        current_price = simulator.data.iloc[current_index]['Close']
        current_time = simulator.data.index[current_index]
        
        # Place mock order
        order = mock_trade_manager.place_sell_order(symbol, qty, current_price, str(current_time))
        
        return f"Simulated SELL order placed: {order['id']} - SELL {qty} {symbol} @ ${current_price:.2f}"
    
    except Exception as e:
        return f"Mock SELL trade failed: {str(e)}"

@tool
def mock_get_position(symbol: str) -> str:
    """
    Gets the current simulated position for a symbol.
    
    Args:
        symbol: Stock symbol to check
    
    Returns:
        Current position details
    """
    try:
        position = mock_trade_manager.get_position(symbol)
        if position:
            return f"Current position in {symbol}: {position['qty']} shares at avg price ${position['avg_price']:.2f}"
        else:
            return f"No current position in {symbol}"
    except Exception as e:
        return f"Error checking position: {str(e)}"

@tool
def mock_check_news(ticker: str) -> str:
    """
    Returns simulated news for a ticker during the historical simulation.
    This function returns pre-defined news that can be customized.
    
    Args:
        ticker: Stock symbol to check news for
    
    Returns:
        Simulated news items for the ticker
    """
    try:
        # Get news from the mock database
        ticker = ticker.upper()
        news_items = MOCK_NEWS_DB.get(ticker, MOCK_NEWS_DB.get("DEFAULT", []))
        
        if not news_items:
            return f"No recent news found for {ticker}"
        
        # Format the news items
        formatted_news = f"Recent news for {ticker}:\n\n"
        for i, news in enumerate(news_items, 1):
            formatted_news += f"{i}. {news['title']}\n"
            formatted_news += f"   Date: {news['date']}\n"
            formatted_news += f"   Source: {news['source']}\n"
            formatted_news += f"   Summary: {news['summary']}\n"
            formatted_news += f"   Sentiment: {news['sentiment']}\n\n"
        
        return formatted_news
    
    except Exception as e:
        return f"Error checking news for {ticker}: {str(e)}"


# Helper function to add mock news to the database
def add_mock_news(ticker, news_item):
    """
    Adds a custom news item to the mock news database.
    
    Args:
        ticker: Stock symbol to add news for
        news_item: Dictionary containing news details
    """
    if ticker not in MOCK_NEWS_DB:
        MOCK_NEWS_DB[ticker] = []
    
    MOCK_NEWS_DB[ticker].append(news_item)
    print(f"Added mock news for {ticker}: {news_item['title']}")


class DecisionState(TypedDict):
    messages: Annotated[list, add_messages]
    finished: bool
    ticker: str
    start_date: str
    end_date: str
    simulator: any

def maybe_route_to_tools(state: DecisionState) -> Literal["tools", "data_node"]:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "data_node"

def gemini_decision_node(state: DecisionState) -> DecisionState:
    sysmsg = SystemMessage(content=GEMINI_TRADER_SYSINT[1])
    history = [sysmsg] + state["messages"]
    
    if state["messages"]:
        new_output = llm_with_tools.invoke(history)
    else:
        new_output = AIMessage(content="Ready to evaluate stocks. Ask me which ticker to analyze.")

    return state | {"messages": [new_output]}

def setup_custom_mock_news():
    """
    Example function to set up custom mock news for a simulation run.
    """
    # Clear existing news for AMD (optional)
    if "AMD" in MOCK_NEWS_DB:
        MOCK_NEWS_DB["AMD"] = []
    
    # Add custom news items for AMD
    add_mock_news("AMD", {
        "title": "AMD Reports Strong Q1 Earnings",
        "summary": "AMD beat analysts' expectations with revenue growth in data center products.",
        "date": "2025-04-17T08:30:00",
        "source": "Financial Times",
        "sentiment": "positive"
    })
    
    add_mock_news("AMD", {
        "title": "Semiconductor Shortage Easing",
        "summary": "Industry experts note supply constraints are improving for chip manufacturers.",
        "date": "2025-04-16T14:45:00",
        "source": "Wall Street Journal",
        "sentiment": "positive"
    })
    
    add_mock_news("AMD", {
        "title": "Competitor Delays Next-Gen Chip",
        "summary": "A major competitor has announced delays in their next-generation processor.",
        "date": "2025-04-15T11:20:00", 
        "source": "TechCrunch",
        "sentiment": "positive"
    })
    
    print("Custom mock news set up successfully")

mock_tools = [mock_check_news, mock_place_market_BUY, mock_place_market_SELL, mock_get_position]
llm_with_tools = get_llm_with_tools(mock_tools)
tool_node = get_tool_node(mock_tools)

# Initialize the mock trade manager
mock_trade_manager = MockTradeManager()