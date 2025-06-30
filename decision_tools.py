from langchain_core.tools import tool
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from alpaca_clients import trading_client, data_client
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
import json

from data_util import get_alpaca_data, add_indicators, dataframe_info, get_stock_price, scrape_float

#==================== G E M I N I +++++++ T O O L S E T ======================================#


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
def get_current_positions() -> str:
    """Get all current positions from alpaca trading account"""
    try:
        positions = trading_client.get_all_positions()

        if not positions:
            return "No Open positions"
        
        position_info = []
        for position in positions:
            position_info.append(
                f"{position.symbol}: {position.qty} shares @ ${float(position.avg_cost):.2f} "
                f"(Current: ${float(position.market_value)/float(position.qty):.2f}), "
                f"P&L: ${float(position.unrealized_pl):.2f})"
            )
        return f"Current positions:\n" + "\n".join(position_info)
    
    except Exception as e:
        return f"Error getting positions: {str(e)}"

@tool
def place_market_BUY(symbol: str, qty: int) -> str:
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
        
        trade_log = {
            "timestamp": str(datetime.now()),
            "action": "BUY",
            "symbol": symbol,
            "qty": qty,
            "price": current_price,
            "order_id": str(BUY_order.id)
        }
        
        # Append to live trading log
        with open("live_trades.json", "a") as f:
            f.write(json.dumps(trade_log) + "\n")

        
        result = f"Order placed: {BUY_order.id} - BUY {qty} {symbol} @ {current_price}"
        print(f"🔨 Tool Result: {result}")
        return result
    except Exception as e:
        error_msg = f"Trade failed: {str(e)}"
        print(f"🔨 Tool Error: {error_msg}")
        return error_msg

@tool
def place_market_SELL(symbol: str, qty: int) -> str:
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
        
        trade_log = {
            "timestamp": str(datetime.now()),
            "action": "SELL",
            "symbol": symbol,
            "qty": qty,
            "price": current_price,
            "order_id": str(SELL_order.id)
        }
        
        # Append to live trading log
        with open("live_trades.json", "a") as f:
            f.write(json.dumps(trade_log) + "\n")

        result= f"Order placed: {SELL_order.id} - SELL {qty} {symbol} @ {current_price}"
        print(f"🔨 Tool Result: {result}")
        return result
    except Exception as e:
        error_msg= f"Trade failed: {str(e)}"
        print(f"🔨 Tool Error: {error_msg}")
        return error_msg
