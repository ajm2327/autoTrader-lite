import pandas as pd
from datetime import datetime, timedelta, time
import time as time_module
import traceback
import json
import os
import sys
import pytz

from typing import Annotated, Literal
from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import ToolNode

from data_util import get_alpaca_data, add_indicators, check_database_status, _update_indicators_in_database, _store_dataframe_in_database
from alpaca_clients import llm, get_llm_with_tools, get_tool_node
from lstm import StockPredictor


from decision_utils import (
    DecisionState,          
    maybe_route_to_tools,  
    gemini_decision_node

)

from decision_tools import (
    get_account, place_market_BUY, place_market_SELL,
    get_rvol, get_price_change, get_company_overview,
    get_float_info, get_detailed_stock_data, real_check_news, get_current_positions
)

from database import (
    db_config, HistoricalData, TechnicalIndicators,
    ModelVersions, Predictions, DatabaseQueries, init_database
)

class HistoricalDataSimulator:
    """
    Enhanced historical data simulator with price change tracking and persistent logging
    """
    def __init__(self, ticker, start_date, end_date, interval_seconds=30, timescale="Minute", log_dir="simulation_logs"):
        """
        Initialize the historical data simulation
        
        Args:
            ticker: Stock symbol to analyze
            start_date: Start date for historical data
            end_date: End date (last day to simulate)
            interval_seconds: Time between updates
            timescale: Data granularity
            log_dir: Directory to save logs
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.is_replay = True
        self.interval_seconds = interval_seconds
        self.timescale = timescale
        self.data = None
        self.current_index = 0
        self.initialized = False
        self.in_position = False
        self.chunk_size = 10
        self.log_dir = log_dir
        self.sim_id = f"{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Trading metrics
        self.entry_price = None
        self.trades = []
        self.decisions = []
        self.total_profit = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
        # Logging files
        self.trade_log_file = f"{log_dir}/{self.sim_id}_trades.json"
        self.decision_log_file = f"{log_dir}/{self.sim_id}_decisions.json"
        self.summary_file = f"{log_dir}/{self.sim_id}_summary.json"

    def initialize(self):
        """Load historical data and prepare simulation"""
        try:
            print(f"Loading historical data for {self.ticker} from {self.start_date} to {self.end_date}...")
            
            init_database()

            check_database_status(self.ticker, self.start_date, self.end_date)
            #INCLUDE LSTM FOR TRAINING, new LSTM START DATE:
            lstm_start_date = (datetime.strptime(self.start_date, '%Y-%m-%d') - timedelta(days=7)).strftime('%Y-%m-%d')#timedelta(days=365*2)).strftime('%Y-%m-%d')
            
            target_date = datetime.strptime(self.end_date, '%Y-%m-%d')
            eastern = pytz.timezone('US/Eastern')

            if target_date.date() == datetime.now().date():
                current_time = eastern.localize(datetime.now())
                print(f"current_time type: {type(current_time)}, value: {current_time}")
                if current_time.time() < time(9,30):
                    print("❌ Market hasn't opened yet today")
                    return False
                end_time = current_time
                self.is_live_day = True
            else:
                end_time = eastern.localize(target_date.replace(hour=16, minute=0,second=0))
                self.is_live_day = False

            # Get full historical data
            #self.data = get_alpaca_data(ticker = self.ticker,start_date = lstm_start_date,end_date = self.end_date,timescale = self.timescale)
            self.data = self._load_or_fetch_data(lstm_start_date, self.end_date)

            if self.data is None or self.data.empty:
                print(f"❌ Could not retrieve historical data for {self.ticker}")
                return False

            # Add indicators using full historical data
            #self.data = add_indicators(data=self.data, indicator_set='alternate')
            print(f"✅ Loaded {len(self.data)} total data points")
            
            # INITIALIZE LSTM
            print("Initializing and training LSTM...")
            self.predictor = StockPredictor(data = self.data, ticker=self.ticker)
            
            try:
                metadata = self.predictor.load_model()
                print(f"✅ Loaded existing LSTM model: {metadata['version']}")
            except (ValueError, FileNotFoundError) as e:
                print(f" No existing model found, training new LSTM...")
                self.predictor.train()
                saved_path = self.predictor.save_model()
                print(f"✅ New LSTM model saved to {saved_path}")
            print("LSTM initialized successfully")
            
            target_start = eastern.localize(target_date.replace(hour=9,minute=30,second=0))


            # Separate training data from data period just for gemini
            if self.data.index.tz is None:
                target_start = target_start.replace(tzinfo=None)
            else:
                target_start = pd.Timestamp(target_start).tz_convert(self.data.index.tz).to_pydatetime()
            sim_data = self.data[self.data.index >= target_start]
            self.data = sim_data
            
            if self.data.empty:
                print(f"❌ No Data found for target date {self.end_date}")
                return False
            
            self.current_index = 0
            self.realtime_cache=[]

            print(f"📅 Simulation starting at market open: {self.data.index[0]}")
            print(f"    Data points available: {len(self.data)}")
            print(f"    ⌚ Live day: {self.is_live_day}")
            # Identify the last day's data
            # Convert all dates to string format to avoid timezone issues
            dates_as_strings = [str(d.date()) for d in self.data.index]
            unique_dates = sorted(set(dates_as_strings))
                                    
            
            # Initialize empty log files
            self._initialize_log_files()
            
            self.initialized = True
            return True

        except Exception as e:
            print(f"❌ Error initializing historical data: {str(e)}")
            traceback.print_exc()
            return False
    
    def _is_market_open(self):
        """Check if market open"""
        eastern = pytz.timezone('US/Eastern')
        now = datetime.now(eastern)

        if now.weekday() >= 5:
            return False
        
        market_open = now.replace(hour=9, minute=30, second=0)
        market_close = now.replace(hour=16, minute=0, second=0)

        return market_open <= now <= market_close

    def _cache_realtime_data(self, new_data_point):
        """Cache realtime data and write to DB"""
        self.realtime_cache.append(new_data_point)
        if len(self.realtime_cache) % 10 == 0:
            self._batch_write_to_db()

    
    def _load_or_fetch_data(self, start_date, end_date):
        """Either loads data from database or fetches from api, updates db"""

        print(f"\n📊 DATA LOADING for {self.ticker}...")
        print(f"    Checking database for existing data from {start_date} to {end_date}...")

        with db_config.get_db_session() as session:
            print(f"    🔎 Checking Database...")
            existing_data = DatabaseQueries.get_data_with_indicators(
                session, self.ticker, start_date, end_date
            )

            if existing_data:
                print(f"    ✅ Found {len(existing_data)} records in database")
                df_data = self._db_records_to_dataframe(existing_data)

                #check for gaps
                date_range = pd.date_range(start=start_date, end=end_date, freq='1min')
                missing_dates = self._find_missing_dates(df_data, date_range)

                if missing_dates:
                    print(f"Found {len(missing_dates)} missing data points, fetching from API...")
                    self._fetch_and_store_missing_data(session, start_date, end_date)

                    existing_data = DatabaseQueries.get_data_with_indicators(
                        session, self.ticker, start_date, end_date
                    )
                    df_data = self._db_records_to_dataframe(existing_data)
                return df_data
            else:
                # No data in db , fetch all data
                print(f"    ❌ No data in batabase for {self.ticker}, fetching from API...")
                return self._fetch_and_store_all_data(session, start_date, end_date)
    
    def _initialize_log_files(self):
        """Initialize empty log files"""
        with open(self.trade_log_file, 'w') as f:
            json.dump([], f)
        with open(self.decision_log_file, 'w') as f:
            json.dump([], f)
        print(f"📝 Log files initialized at {self.log_dir}")
    
    def sim_get_rvol(self):
        """
        Returns a simplified RVOL calculation that avoids timezone issues
        by using a lookback period approach.
        """
        if self.current_index <= 0 or self.current_index >= len(self.data):
            return 1.0  # Default value if we don't have enough data
            
        # Get the current timestamp and volume
        current_idx = min(self.current_index-1, len(self.data)-1)
        current_vol = self.data.iloc[current_idx]["Volume"]
        
        # Use lookback periods to avoid timezone complexities
        lookback = 30  # Periods to look back
        lookback_start = max(0, self.current_index - lookback - 1)
        lookback_end = max(0, self.current_index - 1)
        
        if lookback_start >= lookback_end:
            return 1.0  # Not enough data
        
        # Get average volume of previous periods
        prev_vols = self.data.iloc[lookback_start:lookback_end]["Volume"]
        
        if len(prev_vols) > 0:
            avg_vol = prev_vols.mean()
            if avg_vol > 0:
                return current_vol / avg_vol
                
        return 1.0  # Default to 1.0 if we can't calculate
    
    def get_lstm_prediction(self):
        """Get lstm prediction for current market state"""
        if not hasattr(self, 'predictor') or self.predictor.model is None:
            return None
        
        try:
            needed_rows = self.predictor.backcandles + 1
            start_idx = max(0, self.current_index - needed_rows)
            end_idx = self.current_index

            if (end_idx - start_idx) < needed_rows:
                end_idx = min(len(self.data), start_idx + needed_rows)

            recent_data = self.data.iloc[start_idx:end_idx]

            print(f"Getting LSTM prediction, current_index= {self.current_index}, recent_data length= {len(recent_data)}")
            print(f"start_idx={start_idx}, end_idx={end_idx}, needed_rows={needed_rows}")

            if len(recent_data) < needed_rows:
                print(f" LSTM: Not enough data for prediction, ({len(recent_data)}/{needed_rows})")
                return None
            
            predictions = self.predictor.predict_next_chunk_metrics(recent_data.iloc[-needed_rows:], chunk_size=self.chunk_size)

            return predictions
        
        except Exception as e:
            print(f"Error getting LSTM prediction: {str(e)}")
            return None

    
    def get_day_open_price(self):
        """Get the opening price for the current day"""
        if not self.data is None and not self.data.empty:
            # Get current time index
            current_time = self.data.index[min(self.current_index-1, len(self.data)-1)]
            current_day = current_time.date()
            
            # Find first data point of current day
            day_mask = [d.date() == current_day for d in self.data.index]
            day_data = self.data.iloc[day_mask]
            
            if not day_data.empty:
                return day_data.iloc[0]["Open"]
        
        return None
    
    def calculate_price_change(self, current_price):
        """Calculate percentage change from day's open"""
        open_price = self.get_day_open_price()
        if open_price is not None and open_price > 0:
            change_pct = ((current_price - open_price) / open_price) * 100
            return change_pct, open_price
        return None, None

    def get_initial_data(self):
        """Get the initial data for the agent to analyze"""
        if not self.initialized and not self.initialize():
            return None

        # Get data at the start of the last day
        chunk_size = min(20, len(self.data) - self.current_index)
        initial_chunk = self.data.iloc[self.current_index:self.current_index+chunk_size]
        
        # Update current index
        self.current_index += chunk_size
        
        # Calculate metrics
        rvol = self.sim_get_rvol()
        current_price = initial_chunk.iloc[-1]['Close']
        change_pct, open_price = self.calculate_price_change(current_price)

        lstm_prediction = self.get_lstm_prediction()
        if lstm_prediction:
            prediction_text = f"""- LSTM Predictions (next {self.chunk_size} min):
Raw Predictions: {lstm_prediction['raw_predictions']}
Direction: {lstm_prediction['direction']} ({lstm_prediction['price_change_pct']:.2f}%)
Range: ${lstm_prediction['min_price']:.2f} - ${lstm_prediction['max_price']:.2f}
Mean: ${lstm_prediction['mean_price']:.2f}"""
        else:
            prediction_text = '- LSTM Prediction: N/A'
        
        # Format initial message without leading spaces
        message = f"""
Retrieving historical stock information for {self.ticker}.

Initial historical data summary:
- Date range: {initial_chunk.index.min()} to {initial_chunk.index.max()}
- Current price: ${current_price:.2f}
- Day's open: ${f"{open_price:.2f}" if open_price is not None else 'N/A'}
- Price change: {change_pct:.2f}% today (from day's open)
{'='*60}
{prediction_text}
{'='*60}

- Recent activity:
{initial_chunk.tail(5)[['Open', 'High', 'Low', 'Close', 'Volume']]}

Indicators:
- RSI: {initial_chunk.iloc[-1]['RSI']:.2f}
- MACD: {initial_chunk.iloc[-1]['MACD']:.4f}
- Signal Line: {initial_chunk.iloc[-1]['Signal_Line']:.4f}
- SMA_20: ${initial_chunk.iloc[-1]['SMA_20']:.2f}
- SMA_50: ${initial_chunk.iloc[-1]['SMA_50']:.2f}
- RVOL: {rvol:.2f}

What is your trading decision?
"""
        print("\n🔵 INITIAL DATA SENT TO AGENT:")
        print("=" * 50)
        print(message)
        print("=" * 50)
        return HumanMessage(content=message)

    
    def get_next_update(self):
        """Get the next data update based on the simulation index"""
        if not self.initialized:
            return None

        # Check if we've reached the end of the data
        if self.current_index >= len(self.data):
            print("🏁 End of replay data reached - calling live data...")

            if not self._is_market_open():
                print(" ⛔ Market is closed - end live trading session")
                return None
            
            self.is_replay = False
            fresh_data = get_alpaca_data(self.ticker)
            if fresh_data is None or fresh_data.empty:
                print("❌ Couldn't get real time data")
                return None
            self.data = pd.concat([self.data, fresh_data]).drop_duplicates()

            self.data = add_indicators(self.data, indicator_set='alternate')

            chunk_size = min(self.chunk_size, len(self.data))
            next_chunk = self.data.iloc[-chunk_size:]

            self.current_index = len(self.data)
            time_module.sleep(60)
        
        else:
            self.is_replay = True
            

            # Get next chunk of data
            chunk_size = min(self.chunk_size, len(self.data) - self.current_index)
            next_chunk = self.data.iloc[self.current_index:self.current_index + chunk_size]
        
            # Update current index
            self.current_index += chunk_size

            # For visualization purposes, slow down the simulation
            if self.interval_seconds > 0:
                time_module.sleep(self.interval_seconds)
            
        # Calculate metrics
        rvol = self.sim_get_rvol()
        current_price = next_chunk.iloc[-1]['Close']
        change_pct, open_price = self.calculate_price_change(current_price)
        
        # Determine if price is up or down 
        change_direction = "up" if change_pct and change_pct > 0 else "down"

        lstm_prediction = self.get_lstm_prediction()
        if lstm_prediction:
            prediction_text = f"""- LSTM Predictions (next {self.chunk_size} min):
            Direction: {lstm_prediction['direction']} ({lstm_prediction['price_change_pct']:.2f}%)
            Range: ${lstm_prediction['min_price']:.2f} - ${lstm_prediction['max_price']:.2f}
            Mean: ${lstm_prediction['mean_price']:.2f}"""
        else:
            prediction_text = '- LSTM Prediction: N/A'

        replay_message = "REPLAY MODE - OBSERVATION ONLY" if self.is_replay else "LIVE DATA, TRADING AVAILABLE"
        # Format update message without leading spaces
        update_message = f"""
[Data Update] {self.ticker} at {next_chunk.index[-1].strftime('%Y-%m-%d %H:%M:%S')}:
{replay_message}
- Current price: ${current_price:.2f}
- {self.ticker} is {change_direction} {abs(change_pct):.2f}% today (from ${f"{open_price:.2f}" if open_price is not None else 'N/A'} to ${current_price:.2f})
{'='*60}
{prediction_text}
{'='*60}

- Recent activity:
{next_chunk[['Open', 'High', 'Low', 'Close', 'Volume', 'trade_count']]}

Indicators:
- RSI: {next_chunk.iloc[-1]['RSI']:.2f}
- MACD: {next_chunk.iloc[-1]['MACD']:.4f}
- Signal Line: {next_chunk.iloc[-1]['Signal_Line']:.4f}
- SMA_20: ${next_chunk.iloc[-1]['SMA_20']:.2f}
- SMA_50: ${next_chunk.iloc[-1]['SMA_50']:.2f}
- VWAP: ${next_chunk.iloc[-1]['vwap']:.2f}
- RVOL: {rvol:.2f}

Based on this data, what is your next decision?
"""
        print("\n🔄 DATA UPDATE SENT TO AGENT:")
        print(f'{'='*60}')
        #print(f"Time: {next_chunk.index[-1].strftime('%Y-%m-%d %H:%M:%S')}")
        #print(f"Price: ${current_price:.2f} ({change_direction} {abs(change_pct):.2f}%)")
        #print(f"RVOL: {rvol:.2f}")
        print(f'{update_message}')
        print(f'{'='*60}')

        
        return HumanMessage(content=update_message)

    def log_decision(self, decision_text, current_time, current_price, position_change):
        """Log a decision to the decision log file"""
        decision = {
            "time": str(current_time),
            "price": float(current_price),
            "decision": decision_text,
            "position_change": position_change
        }
        
        # Add to in-memory list
        self.decisions.append(decision)
        
        # Update log file
        try:
            try:
                with open(self.decision_log_file, 'r') as f:
                    decisions = json.load(f)
            except FileNotFoundError:
                decisions = []
            decisions.append(decision)
            with open(self.decision_log_file, 'w') as f:
                json.dump(decisions, f, indent=2)
        except Exception as e:
            print(f"Error logging decision: {str(e)}")

    def log_trade(self, trade):
        """Log a trade to the trade log file"""
        # Convert any non-serializable objects
        trade_log = {
            "entry_time": str(trade["entry_time"]),
            "entry_price": float(trade["entry_price"]),
            "exit_time": str(trade["exit_time"]),
            "exit_price": float(trade["exit_price"]),
            "pnl_pct": float(trade["pnl_pct"]),
            "pnl_abs": float(trade["pnl_abs"])
        }
        
        # Update log file
        try:
            try:
                with open(self.trade_log_file, 'r') as f:
                    trades = json.load(f)
            except FileNotFoundError:
                trades = []
            trades.append(trade_log)
            with open(self.trade_log_file, 'w') as f:
                json.dump(trades, f, indent=2)
        except Exception as e:
            print(f"Error logging trade: {str(e)}")

    def update_position(self, decision_text):
        """Update position status based on agent's decision."""
        current_price = self.data.iloc[min(self.current_index-1, len(self.data)-1)]['Close']
        current_time = self.data.index[min(self.current_index-1, len(self.data)-1)]
        position_change = "NO CHANGE"
        
        # Check for entry
        if ("ENTER TRADE" in decision_text.upper() or "Enter Trade" in decision_text) and not self.in_position:
            self.in_position = True
            self.entry_price = current_price
            position_change = "ENTERED"
            
            # Print with timestamp from the data (not real-time)
            print(f"\n🟢 ENTERING TRADE for {self.ticker} at {current_time} - ${current_price:.2f}")
            print(f"Decision text: {decision_text[:100]}...")
            
        # Check for exit
        elif ("EXIT TRADE" in decision_text.upper() or "Exit Trade" in decision_text) and self.in_position:
            self.in_position = False
            exit_price = current_price
            position_change = "EXITED"
            
            # Calculate P&L
            pnl_pct = ((exit_price - self.entry_price) / self.entry_price) * 100
            pnl_abs = exit_price - self.entry_price
            
            # Create trade record
            trade = {
                "entry_time": current_time - pd.Timedelta(minutes=self.chunk_size),  # Approximate entry time
                "entry_price": self.entry_price,
                "exit_time": current_time,
                "exit_price": exit_price,
                "pnl_pct": pnl_pct,
                "pnl_abs": pnl_abs
            }
            
            # Add to trades list
            self.trades.append(trade)
            
            # Log the trade
            self.log_trade(trade)
            
            # Update stats
            self.total_profit += pnl_abs
            if pnl_pct > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1
            
            # Print result with emoji based on profit/loss
            emoji = "✅" if pnl_pct > 0 else "❌"
            print(f"\n🔴 EXITING TRADE for {self.ticker} at {current_time} - ${exit_price:.2f}")
            print(f"{emoji} Result: {pnl_pct:.2f}% (${pnl_abs:.2f})")
            print(f"Decision text: {decision_text[:100]}...")
            
            self.entry_price = None
        
        # Log the decision
        self.log_decision(decision_text, current_time, current_price, position_change)
        
        return position_change
    
    def save_summary(self):
        """Save a summary of the simulation results"""
        win_rate = 0 if len(self.trades) == 0 else (self.winning_trades/len(self.trades)*100)
        
        summary = {
            "ticker": self.ticker,
            "simulation_period": f"{self.start_date} to {self.end_date}",
            "simulation_id": self.sim_id,
            "total_decisions": len(self.decisions),
            "trades": {
                "total_trades": len(self.trades),
                "winning_trades": self.winning_trades,
                "losing_trades": self.losing_trades,
                "win_rate": win_rate,
                "total_profit": self.total_profit
            },
            "final_position": "IN POSITION" if self.in_position else "NOT IN POSITION",
            "timestamp": str(datetime.now())
        }
        
        try:
            with open(self.summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"📊 Simulation summary saved to {self.summary_file}")
        except Exception as e:
            print(f"Error saving summary: {str(e)}")
        
        return summary
    
    # DATABASE HELPER METHODS
    def _db_records_to_dataframe(self, db_records):
        """Convert database records to pd df"""
        data = []
        for record in db_records:
            row = {
                'Open': float(record.open),
                'High': float(record.high),
                'Low': float(record.low),
                'Close': float(record.close),
                'trade_count': float(record.trade_count) if record.trade_count else None,
                'vwap': float(record.vwap) if record.vwap else None,
                'Volume': int(record.volume),
                'Adj Close': float(record.adjusted_close or record.close)
            }

            if record.indicators:
                indicator = record.indicators[0]
                row.update({
                    'SMA_20': float(indicator.sma_20) if indicator.sma_20 else None,
                    'SMA_50': float(indicator.sma_50) if indicator.sma_50 else None,
                    'SMA_200': float(indicator.sma_200) if indicator.sma_200 else None,
                    'RSI': float(indicator.rsi) if indicator.rsi else None,
                    'MACD': float(indicator.macd) if indicator.macd else None,
                    'Signal_Line': float(indicator.signal_line) if indicator.signal_line else None,
                    'Middle_Band': float(indicator.middle_band) if indicator.middle_band else None,
                    'Upper_Band': float(indicator.upper_band) if indicator.upper_band else None,
                    'Lower_Band': float(indicator.lower_band) if indicator.lower_band else None,
                    'EMA': float(indicator.ema) if indicator.ema else None,
                })

            data.append(row)

        df = pd.DataFrame(data, index=[r.timestamp for r in db_records])
        return df
    
    def _find_missing_dates(self, existing_df, expected_date_range):
        """Find missing dates in data"""
        existing_dates = set(existing_df.index)
        expected_dates = set(expected_date_range)
        return list(expected_dates - existing_dates)
    
    def _fetch_and_store_missing_data(self, session, start_date, end_date):
        """Fetch all data from api and store in db"""
        df = get_alpaca_data(self.ticker, start_date, end_date, timescale=self.timescale, store_in_db=True)
        df = add_indicators(df, indicator_set='alternate',
                            store_in_db = True, ticker=self.ticker)
        return df
    
    def _fetch_and_store_all_data(self, session, start_date, end_date):
        print(f"    📂 No Data in DB - fetching from Alpaca API...")
        df = get_alpaca_data(self.ticker, start_date, end_date, timescale=self.timescale, store_in_db=True)
        if df is not None and not df.empty:
            print(f"    🗃️Storing {len(df)} records to database...")
            _store_dataframe_in_database(self.ticker, df)

            df = add_indicators(df, indicator_set='alternate', store_in_db=True, ticker=self.ticker)
            _update_indicators_in_database(self.ticker, df)
        return df


class PersistentLogger:
    """
    Maintains a persistent master log of all simulation runs and their performance.
    This allows for analysis across multiple simulations and strategy adjustment.
    """
    def __init__(self, master_log_file="simulation_logs/master_performance_log.json"):
        """
        Initialize the persistent logger
        
        Args:
            master_log_file: Path to the master log file
        """
        self.master_log_file = master_log_file
        self._ensure_log_exists()
    
    def _ensure_log_exists(self):
        """Ensure the master log file exists, create it if it doesn't"""
        import os
        import json
        
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(self.master_log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # Create empty log file if it doesn't exist
        if not os.path.exists(self.master_log_file):
            with open(self.master_log_file, 'w') as f:
                json.dump({
                    "simulations": [],
                    "meta": {
                        "created_at": str(datetime.now()),
                        "total_simulations": 0,
                        "total_trades": 0,
                        "overall_win_rate": 0,
                        "overall_profit": 0.0
                    }
                }, f, indent=2)
            print(f"📝 Created new master log file at {self.master_log_file}")
    
    def log_simulation(self, simulator):
        """
        Add a completed simulation run to the master log
        
        Args:
            simulator: The HistoricalDataSimulator instance with completed simulation data
        """
        import json
        
        # Read existing log
        try:
            with open(self.master_log_file, 'r') as f:
                master_log = json.load(f)
        except Exception as e:
            print(f"❌ Error reading master log: {str(e)}")
            self._ensure_log_exists()
            with open(self.master_log_file, 'r') as f:
                master_log = json.load(f)
        
        # Create simulation entry
        win_rate = 0 if len(simulator.trades) == 0 else (simulator.winning_trades/len(simulator.trades)*100)
        
        simulation_entry = {
            "id": simulator.sim_id,
            "ticker": simulator.ticker,
            "date_range": f"{simulator.start_date} to {simulator.end_date}",
            "timestamp": str(datetime.now()),
            "performance": {
                "total_decisions": len(simulator.decisions),
                "total_trades": len(simulator.trades),
                "winning_trades": simulator.winning_trades,
                "losing_trades": simulator.losing_trades,
                "win_rate": win_rate,
                "total_profit": simulator.total_profit,
                "average_trade_pnl": simulator.total_profit / len(simulator.trades) if len(simulator.trades) > 0 else 0
            },
            "final_position": "IN POSITION" if simulator.in_position else "NOT IN POSITION",
            "log_files": {
                "trades": simulator.trade_log_file,
                "decisions": simulator.decision_log_file,
                "summary": simulator.summary_file
            },
            "key_decisions": self._extract_key_decisions(simulator)
        }
        
        # Add to master log
        master_log["simulations"].append(simulation_entry)
        
        # Update meta statistics
        total_sims = len(master_log["simulations"])
        total_trades = sum(sim["performance"]["total_trades"] for sim in master_log["simulations"])
        total_winning_trades = sum(sim["performance"]["winning_trades"] for sim in master_log["simulations"])
        total_profit = sum(sim["performance"]["total_profit"] for sim in master_log["simulations"])
        
        overall_win_rate = 0
        if total_trades > 0:
            overall_win_rate = (total_winning_trades / total_trades) * 100
        
        master_log["meta"] = {
            "created_at": master_log["meta"]["created_at"],
            "last_updated": str(datetime.now()),
            "total_simulations": total_sims,
            "total_trades": total_trades,
            "overall_win_rate": overall_win_rate,
            "overall_profit": total_profit
        }
        
        # Write updated log
        try:
            with open(self.master_log_file, 'w') as f:
                json.dump(master_log, f, indent=2)
            print(f"📊 Added simulation results to master log: {self.master_log_file}")
        except Exception as e:
            print(f"❌ Error updating master log: {str(e)}")
    
    def _extract_key_decisions(self, simulator):
        """Extract key decisions from a simulation (entries, exits, etc.)"""
        key_decisions = []
        
        for decision in simulator.decisions:
            if decision["position_change"] != "NO CHANGE":
                key_decisions.append({
                    "time": decision["time"],
                    "price": decision["price"],
                    "action": decision["position_change"],
                    "reasoning": self._extract_reason(decision["decision"])
                })
        
        return key_decisions
    
    def _extract_reason(self, decision_text):
        """Extract the reasoning from a decision text"""
        # Try to extract just the reason part of the decision
        import re
        
        # Look for reason patterns like "because...", "due to...", "as..."
        reason_patterns = [
            r"because\s+(.+)",
            r"due to\s+(.+)",
            r"reason:?\s+(.+)",
            r"as\s+(.+)",
            r"since\s+(.+)"
        ]
        
        for pattern in reason_patterns:
            match = re.search(pattern, decision_text.lower())
            if match:
                return match.group(1).strip()
        
        # If no specific reason found, return a truncated version of the decision
        return decision_text[:100] + "..." if len(decision_text) > 100 else decision_text


def update_data_node_with_persistent_logging(data_node_func):
    """
    Decorator to add persistent logging to the data node
    """
    def wrapped_data_node(state):
        # Get result from original data node
        result = data_node_func(state)
        
        # If simulation is finished, add to master log
        if result.get("finished", False) and state.get("simulator"):
            try:
                # Create logger and log simulation
                logger = PersistentLogger()
                logger.log_simulation(state["simulator"])
            except Exception as e:
                print(f"❌ Error in persistent logging: {str(e)}")
                import traceback
                traceback.print_exc()
        
        return result
    
    return wrapped_data_node


# Apply the decorator to the data_node function
@update_data_node_with_persistent_logging
def data_node(state):
    """
    This node replaces the human node that interacts with the langgraph agent.
    It simulates the last day of historical data and logs all decisions.
    """
    simulator = state.get("simulator")
    decisions = state.get("decisions", [])
    
    # Initialize simulator if not exist
    if simulator is None:
        ticker = state.get("ticker", "AMD") # Replace with whatever ticker to start with
        start_date = state.get("start_date", "2025-03-01")  # Historical data start
        end_date = state.get("end_date", "2025-04-17")      # Last day to simulate
        log_dir = state.get("log_dir", "simulation_logs")   # Directory for logs
        
        print(f"\n🚀 STARTING HISTORICAL SIMULATION FOR LAST DAY")
        print(f"Ticker: {ticker}")
        print(f"Historical data: {start_date} to {end_date}")
        print(f"Log directory: {log_dir}")
        print("=" * 50)
        
        simulator = HistoricalDataSimulator(ticker, start_date, end_date, 
                                           interval_seconds=5, 
                                           log_dir=log_dir)
        state["simulator"] = simulator

        # Get initial data
        try:
            message = simulator.get_initial_data()
            if message is None:
                print("❌ Could not initialize simulator data")
                return state | {"finished": True}
                
            print(f"📈 Simulation starting for last day...")
            return state | {"messages": [message]}
        except Exception as e:
            print(f"❌ Error getting initial data: {str(e)}")
            traceback.print_exc()
            return state | {"finished": True}

    # Check for agent response and record decisions
    last_message = state["messages"][-1] if state["messages"] else None
    if last_message and hasattr(last_message, "content"):
        # Print agent's response
        print("\n🧠 AGENT DECISION:")
        print("=" * 50)
        print(last_message.content)
        print("=" * 50)
        
        try:
            # Update position status based on agent decision
            position_change = simulator.update_position(last_message.content)
            
            # Store decision in state for reference
            state["decisions"] = simulator.decisions
        except Exception as e:
            print(f"❌ Error processing agent decision: {str(e)}")

    # Get next data update
    try:
        next_message = simulator.get_next_update()
        if next_message is None:
            # End of simulation - print and save summary
            print("\n📋 SIMULATION SUMMARY")
            print("=" * 50)
            print(f"Total decisions made: {len(simulator.decisions)}")
            
            # Show decision history
            print("\n📜 DECISION HISTORY:")
            for i, d in enumerate(simulator.decisions):
                if d["position_change"] != "NO CHANGE":
                    print(f"{i+1}. {d['time']} - ${d['price']:.2f} - {d['position_change']}")
            
            # Show final position and P&L if applicable
            if simulator.in_position:
                print("\n⚠️ WARNING: Still in position at end of simulation!")
                
            if simulator.trades:
                print("\n💰 TRADE SUMMARY:")
                print(f"Total trades: {len(simulator.trades)}")
                win_rate = 0 if len(simulator.trades) == 0 else simulator.winning_trades/len(simulator.trades)*100
                print(f"Winning trades: {simulator.winning_trades} ({win_rate:.1f}%)")
                print(f"Losing trades: {simulator.losing_trades} ({100-win_rate:.1f}%)")
                print(f"Total P&L: ${simulator.total_profit:.2f}")
            
            # Save final summary
            summary = simulator.save_summary()
            print(f"\n📝 Logs saved to:")
            print(f"  - Trades: {simulator.trade_log_file}")
            print(f"  - Decisions: {simulator.decision_log_file}")
            print(f"  - Summary: {simulator.summary_file}")
            print(f"  - Master Log: simulation_logs/master_performance_log.json")
            
            print("=" * 50)
            print("Simulation complete!")
            return state | {"finished": True}

        return state | {"messages": [next_message]}
    except Exception as e:
        print(f"❌ Error getting next update: {str(e)}")
        traceback.print_exc()
        return state | {"finished": True}




def run_historical_simulation(ticker="AMD", start_date="2025-03-01", end_date="2025-04-17", 
                             max_iterations=200, log_dir="simulation_logs"):
    """
    Run a historical data simulation for a specified ticker and date range,
    focusing on the last day's price action.
    
    Args:
        ticker: Stock symbol to analyze
        start_date: Start date for historical data in YYYY-MM-DD format
        end_date: End date (last day to simulate) in YYYY-MM-DD format
        max_iterations: Maximum number of iterations to prevent infinite loops
        log_dir: Directory to save simulation logs
    
    Returns:
        The final state of the simulation
    """
    class TeeOutput:
        def __init__(self, file_path):
            self.file = open(file_path, 'w')
            self.stdout = sys.stdout
        def write(self, data):
            self.stdout.write(data)
            self.file.write(data)
            self.file.flush()
        def flush(self):
            self.stdout.flush()
            self.file.flush()

    sys.stdout = TeeOutput('live_trading.log')

    print(f"\n{'='*60}")
    print(f"🚀 STARTING HISTORICAL TRADING SIMULATION FOR {ticker}")
    print(f"{'='*60}")
    print(f"🔍 Ticker: {ticker}")
    print(f"📅 Historical data period: {start_date} to {end_date}")
    print(f"🔎 Simulation focusing on last day: {end_date}")
    print(f"📝 Logs will be saved to: {log_dir}")
    print(f"⚙️ Max iterations: {max_iterations}")
    print(f"{'='*60}\n")
    
    try:
        # CREATE GRAPH
        graph = StateGraph(DecisionState)
        graph.add_node("data_node", data_node)
        graph.add_node("gemini", gemini_decision_node)
        graph.add_node("tools", tool_node)
        
        # IMPORTANT: Tools should route back to data_node to prevent loops
        graph.add_conditional_edges("gemini", maybe_route_to_tools)
        graph.add_edge("data_node", "gemini")
        graph.add_edge("tools", "gemini")
        
        graph.set_entry_point("data_node")
        
        compiled_graph = graph.compile()
        
        # Initial state
        initial_state = {
            "messages": [],
            "finished": False,
            "ticker": ticker,
            "start_date": start_date,
            "end_date": end_date,
            "log_dir": log_dir,
            "simulator": None
        }
        
        # Set recursion limit
        config = {"recursion_limit": max_iterations}
        
        # Run the simulation
        final_state = compiled_graph.invoke(initial_state, config)
        
        print("\n✅ Simulation completed successfully!")
        return final_state
        
    except Exception as e:
        print(f"\n❌ Simulation error: {str(e)}")
        if "recursion" in str(e).lower():
            print("The simulation hit the maximum number of iterations.")
            print("This usually means the model is in a loop or there's an issue with graph routing.")
            print(f"Try increasing max_iterations (current: {max_iterations}).")
        
        traceback.print_exc()
        return None



if __name__ == "__main__":
    tools = [get_account, place_market_BUY, place_market_SELL, get_current_positions]
     
    # Create LLM and bind tools
    # Use centralized LLM creation
    llm_with_tools = get_llm_with_tools(tools)
    tool_node = get_tool_node(tools)
    final_state = run_historical_simulation(
        ticker="SPY",
        start_date="2025-03-01",
        end_date="2025-06-06"
    )