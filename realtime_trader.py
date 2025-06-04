# Real-time Trading Extension
import pytz
from datetime import datetime, timedelta

def is_market_open():
    """Check if the US stock market is currently open"""
    # Get current time in Eastern timezone
    eastern = pytz.timezone('US/Eastern')
    now = datetime.now(eastern)
    
    # Check if it's a weekday
    if now.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        return False
    
    # Check if it's during market hours (9:30 AM - 4:00 PM ET)
    market_open = now.replace(hour=9, minute=30, second=0)
    market_close = now.replace(hour=16, minute=0, second=0)
    
    return market_open <= now <= market_close

# Extend HistoricalDataSimulator with real-time capabilities
class RealTimeTrader(HistoricalDataSimulator):
    """
    Extended version of HistoricalDataSimulator that can operate in real-time
    """
    def __init__(self, ticker, interval_seconds=30, log_dir="trading_logs"):
        # Get current date for end_date
        end_date = datetime.now().strftime('%Y-%m-%d')
        # Use fixed start_date for enough historical context
        start_date = "2024-01-01"
        
        # Call parent constructor
        super().__init__(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            interval_seconds=interval_seconds,
            timescale="Minute",
            log_dir=log_dir
        )
        
        # Real-time specific attributes
        self.realtime_mode = True
        self.last_update_time = datetime.now()
    
    def initialize(self):
        """Modified initialize method for real-time trading"""
        # Check if market is open
        if not is_market_open():
            print("❌ Market is currently closed. Real-time trading not available.")
            return False
            
        # Rest of initialization remains similar to parent class
        try:
            print(f"📂 Loading historical data for {self.ticker} from {self.start_date} to {self.end_date}...")
            # Get initial historical data for context
            self.data = get_alpaca_data(
                ticker=self.ticker,
                start_date=self.start_date,
                end_date=self.end_date,
                timescale=self.timescale
            )

            if self.data is None or self.data.empty:
                print(f"❌ Could not retrieve historical data for {self.ticker}")
                return False

            # Add indicators using full historical data
            self.data = add_indicators(data=self.data, indicator_set='alternate')
            print(f"✅ Loaded {len(self.data)} total data points for context")
            
            # Initialize log files
            self._initialize_log_files()
            
            self.initialized = True
            return True

        except Exception as e:
            print(f"❌ Error initializing real-time trader: {str(e)}")
            traceback.print_exc()
            return False
    
    def get_next_update(self):
        """Get real-time market data update"""
        if not self.initialized:
            return None
            
        # Check if market is still open
        if not is_market_open():
            print("🏁 Market has closed. Ending trading session.")
            return None
            
        # Enforce update interval
        time_since_update = (datetime.now() - self.last_update_time).total_seconds()
        if time_since_update < self.interval_seconds:
            time.sleep(1)  # Sleep for 1 second and recheck
            return self.get_next_update()
            
        # Update last update time
        self.last_update_time = datetime.now()
        
        try:
            # Fetch current data (last 5-10 minutes)
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(minutes=30)).strftime('%Y-%m-%d %H:%M:%S')
            
            current_data = get_alpaca_data(
                ticker=self.ticker,
                start_date=start_date,
                end_date=end_date,
                timescale=self.timescale
            )
            
            if current_data is None or current_data.empty:
                print(f"⚠️ Could not retrieve current data for {self.ticker}")
                time.sleep(self.interval_seconds)
                return self.get_next_update()
                
            # Add indicators
            current_data = add_indicators(current_data, indicator_set='alternate')
            
            # Update our full data
            # We append the new data to maintain our history (needed for indicators)
            self.data = pd.concat([self.data, current_data]).drop_duplicates()
            
            # Calculate metrics
            rvol = self.sim_get_rvol()
            current_price = current_data.iloc[-1]['Close']
            
            # Get today's open price
            today = datetime.now().date()
            today_data = self.data[self.data.index.date == today]
            if not today_data.empty:
                open_price = today_data.iloc[0]['Open']
                change_pct = ((current_price - open_price) / open_price) * 100
            else:
                open_price = None
                change_pct = None
                
            # Determine if price is up or down 
            change_direction = "up" if change_pct and change_pct > 0 else "down"

            # Format update message without leading spaces
            update_message = f"""
[Data Update] {self.ticker} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}:
- Current price: ${current_price:.2f}
- {self.ticker} is {change_direction} {abs(change_pct) if change_pct else 0:.2f}% today (from ${f"{open_price:.2f}" if open_price is not None else 'N/A'} to ${current_price:.2f})
- Recent activity:
{current_data.tail(5)[['Open', 'High', 'Low', 'Close', 'Volume']]}

Indicators:
- RSI: {current_data.iloc[-1]['RSI']:.2f}
- MACD: {current_data.iloc[-1]['MACD']:.4f}
- Signal Line: {current_data.iloc[-1]['Signal_Line']:.4f}
- SMA_20: ${current_data.iloc[-1]['SMA_20']:.2f}
- SMA_50: ${current_data.iloc[-1]['SMA_50']:.2f}
- RVOL: {rvol:.2f}

Based on this data, what is your next decision?
"""
            print("\n🔄 REAL-TIME DATA UPDATE:")
            print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
            print(f"Price: ${current_price:.2f} ({change_direction} {abs(change_pct) if change_pct else 0:.2f}%)")
            print(f"RVOL: {rvol:.2f}")
            
            return HumanMessage(content=update_message)
            
        except Exception as e:
            print(f"❌ Error getting real-time update: {str(e)}")
            traceback.print_exc()
            time.sleep(self.interval_seconds)
            return self.get_next_update()

# Modified Data Node to support both historical and real-time modes
@update_data_node_with_persistent_logging
def unified_data_node(state):
    """
    Enhanced data node that supports both historical simulation and real-time trading
    """
    mode = state.get("mode", "simulation")  # Default to simulation mode
    simulator = state.get("simulator")
    decisions = state.get("decisions", [])
    
    # Initialize simulator if not exist
    if simulator is None:
        ticker = state.get("ticker", "AMD")
        log_dir = state.get("log_dir", "trading_logs")
        
        if mode == "simulation":
            # Historical simulation mode
            start_date = state.get("start_date", "2025-03-01")
            end_date = state.get("end_date", "2025-04-17")
            
            print(f"\n🚀 STARTING HISTORICAL SIMULATION FOR {ticker}")
            print(f"Ticker: {ticker}")
            print(f"Historical data: {start_date} to {end_date}")
            print(f"Log directory: {log_dir}")
            print("=" * 50)
            
            simulator = HistoricalDataSimulator(
                ticker=ticker, 
                start_date=start_date, 
                end_date=end_date, 
                interval_seconds=30, 
                log_dir=log_dir
            )
        else:
            # Real-time trading mode
            print(f"\n🚀 STARTING REAL-TIME TRADING FOR {ticker}")
            print(f"Ticker: {ticker}")
            print(f"Log directory: {log_dir}")
            print("=" * 50)
            
            simulator = RealTimeTrader(
                ticker=ticker,
                interval_seconds=30,
                log_dir=log_dir
            )
            
        state["simulator"] = simulator

        # Get initial data
        try:
            message = simulator.get_initial_data()
            if message is None:
                print("❌ Could not initialize simulator data")
                return state | {"finished": True}
                
            print(f"📈 {'Simulation' if mode == 'simulation' else 'Trading'} starting...")
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
            # End of session - print and save summary
            print("\n📋 SESSION SUMMARY")
            print("=" * 50)
            print(f"Total decisions made: {len(simulator.decisions)}")
            
            # Show decision history
            print("\n📜 DECISION HISTORY:")
            for i, d in enumerate(simulator.decisions):
                if d["position_change"] != "NO CHANGE":
                    print(f"{i+1}. {d['time']} - ${d['price']:.2f} - {d['position_change']}")
            
            # Show final position and P&L if applicable
            if simulator.in_position:
                print("\n⚠️ WARNING: Still in position at end of session!")
                
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
            print(f"  - Master Log: {log_dir}/master_performance_log.json")
            
            print("=" * 50)
            print(f"{'Simulation' if mode == 'simulation' else 'Trading session'} complete!")
            return state | {"finished": True}
            
        # Check for micropullback pattern in simulation mode
        if mode == "simulation" and simulator.current_index % 5 == 0:
            simulator.evaluate_micropullback(
                ross_entry_time=simulator.ross_entry_time,
                ross_entry_price=simulator.ross_entry_price,
                ross_exit_time=simulator.ross_exit_time,
                ross_exit_price=simulator.ross_exit_price
            )
            
        return state | {"messages": [next_message]}
    except Exception as e:
        print(f"❌ Error getting next update: {str(e)}")
        traceback.print_exc()
        return state | {"finished": True}

# Function to run real-time trading
def run_realtime_trading(ticker="AMD", max_iterations=500, log_dir="trading_logs"):
    """
    Run a real-time trading session for a specified ticker
    
    Args:
        ticker: Stock symbol to trade
        max_iterations: Maximum number of iterations to prevent infinite loops
        log_dir: Directory to save trading logs
    
    Returns:
        The final state of the trading session
    """
    print(f"\n{'='*60}")
    print(f"🚀 STARTING REAL-TIME TRADING FOR {ticker}")
    print(f"{'='*60}")
    print(f"🔍 Ticker: {ticker}")
    print(f"⏰ Market Status: {'OPEN' if is_market_open() else 'CLOSED'}")
    
    if not is_market_open():
        print("❌ Market is currently closed. Real-time trading not available.")
        return None
    
    print(f"📝 Logs will be saved to: {log_dir}")
    print(f"⚙️ Max iterations: {max_iterations}")
    print(f"{'='*60}\n")
    
    try:
        # CREATE GRAPH
        graph = StateGraph(DecisionState)
        graph.add_node("data_node", unified_data_node)
        graph.add_node("gemini", gemini_decision_node)
        
        # Use actual tools instead of mock tools for real-time trading
        real_tools = [get_account, place_market_BUY, place_market_SELL, get_stock_price,
                     get_rvol, get_price_change, get_current_quote, get_company_overview,
                     get_float_info, get_detailed_stock_data, real_check_news]
        
        real_tool_node = ToolNode(real_tools)
        graph.add_node("tools", real_tool_node)
        
        # IMPORTANT: Tools should route back to data_node to prevent loops
        graph.add_conditional_edges("gemini", maybe_route_to_tools)
        graph.add_edge("data_node", "gemini")
        graph.add_edge("tools", "gemini")
        
        graph.set_entry_point("data_node")
        
        compiled_graph = graph.compile()
        
        # Initial state - specify real-time mode
        initial_state = {
            "messages": [],
            "finished": False,
            "ticker": ticker,
            "mode": "realtime",  # Specify real-time mode
            "log_dir": log_dir,
            "simulator": None
        }
        
        # Set recursion limit
        config = {"recursion_limit": max_iterations}
        
        # Run the trading session
        final_state = compiled_graph.invoke(initial_state, config)
        
        print("\n✅ Trading session completed successfully!")
        return final_state
        
    except Exception as e:
        print(f"\n❌ Trading error: {str(e)}")
        if "recursion" in str(e).lower():
            print("The trading session hit the maximum number of iterations.")
            print("This usually means the model is in a loop or there's an issue with graph routing.")
            print(f"Try increasing max_iterations (current: {max_iterations}).")
        
        traceback.print_exc()
        return None

    
# For real-time trading (only run during market hours)
final_state = run_realtime_trading(ticker="AMD")