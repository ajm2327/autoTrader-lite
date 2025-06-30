import schedule
import time as time_module
import subprocess
import pytz
from datetime import datetime, date, time
import logging
import signal
import sys
import os
import pandas_market_calendars as mcal
import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers = [
        logging.FileHandler('market_scheduler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MarketScheduler:
    def __init__(self):
        self.eastern = pytz.timezone('US/Eastern')
        self.simulator_process = None
        self.is_running = False

        self.nyse = mcal.get_calendar('NYSE')

        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signla {signum}, shutting down...")
        self.stop_trading()
        sys.exit(0)


    def is_market_day(self):
        now = datetime.now(self.eastern)
        today = now.date()

        start_date = today
        end_date = today

        try:
            trading_days = self.nyse.valid_days(
                start_date=start_date,
                end_date=end_date
            )

            today_ts = pd.Timestamp(today, tz=self.eastern)

            return len(trading_days) > 0 and today_ts.date() in [d.date() for d in trading_days]
        
        except Exception as e:
            logger.warning(f"Error checking market calendar, falling back to weekday check: {e}")
            return today.weekday() < 5
        
    def get_market_hours(self):
        """get market open/close"""
        now = datetime.now(self.eastern)
        today = now.date()

        try:
            schedule_df = self.nyse.schedule(start_date =today, end_date=today)

            if len(schedule_df) > 0:
                market_open = schedule_df.iloc[0]['market_open'].tz_convert(self.eastern)
                market_close = schedule_df.iloc[0]['market_close'].tz_convert(self.eastern)
                return market_open.time(), market_close.time()
            else:
                return None, None
            
        except Exception as e:
            logger.warning(f"Error getting market schedule, using default hours: {e}")
            return time(9,30), time(16,0)
        
    def is_market_hours(self):
        """check if currently market hours"""
        if not self.is_market_day():
            return False
        
        now = datetime.now(self.eastern)
        market_open_time, market_close_time = self.get_market_hours()

        if market_open_time is None or market_close_time is None:
            return False
        
        current_time = now.time()
        return market_open_time <= current_time <= market_close_time
    
    def start_trading(self):
        if not self.is_market_day():
            logger.info("Not a trading day, skipping market open")
            return
        
        if self.simulator_process is not None:
            logger.warning("Trading simulator already running")
            return
        
        try:
            logger.info("🟩 Market Open - Starting Trading Simulator")
            self.simulator_process = subprocess.Popen(
                ["python", "historical_data_simulator.py"],
                cwd = os.getcwd()
            )
            self.is_running = True
            logger.info(f"Trading simulator started with PID: {self.simulator_process.pid}")

        except Exception as e:
            logger.error(f"Failed to start trading simulator: {str(e)}")

    def stop_trading(self):
        """Stop trading simulator"""
        if self.simulator_process is None:
            logger.info("No trading simulator running")
            return
        
        try:
            logger.info("❌ Market Close - Stopping trading simulator")

            self.simulator_process.terminate()

            try:
                self.simulator_process.wait(timeout=10)

            except subprocess.TimeoutExpired:
                logger.warning("Graceful shutdown timed out, forcing termination")
                self.simulator_process.kill()
                self.simulator_process.wait()

            logger.info("Trading Simulator Stopped")
            self.simulator_process = None
            self.is_running = None

        except Exception as e:
            logger.error(f"Error stopping trading simulator: {str(e)}")

    def check_process_health(self):
        """Check if the simulator process is still running and handle early close"""
        if self.is_running and not self.is_market_hours():
            logger.info("Market closed (possibly early close), stopping trading")
            self.stop_trading()
            return
        
        if self.simulator_process is not None:
            poll_result = self.simulator_process.poll()
            if poll_result is not None:
                logger.warning(f"Trading simulator process died with return code: {poll_result}")
                self.simulator_process = None
                self.is_running = False


                if self.is_market_hours():
                    logger.info("Attempting to restart trading simulator")
                    self.start_trading()

        elif self.is_market_hours() and not self.is_running:
            logger.info("Market is open but trading not running, starting now")
            self.start_trading()

    def schedule_market_hours(self):
        """Set up market schedule"""

        current_system_tz = datetime.now().astimezone().tzinfo
        logger.info(f"System timezone: {current_system_tz}")
        logger.info(f"Target timezone: {self.eastern}")

        # Schedule market open (9:30 AM ET)
        schedule.every().monday.at("09:30").do(self.start_trading)
        schedule.every().tuesday.at("09:30").do(self.start_trading)
        schedule.every().wednesday.at("09:30").do(self.start_trading)
        schedule.every().thursday.at("09:30").do(self.start_trading)
        schedule.every().friday.at("09:30").do(self.start_trading)
        
        # Schedule market close (4:00 PM ET)
        schedule.every().monday.at("16:00").do(self.stop_trading)
        schedule.every().tuesday.at("16:00").do(self.stop_trading)
        schedule.every().wednesday.at("16:00").do(self.stop_trading)
        schedule.every().thursday.at("16:00").do(self.stop_trading)
        schedule.every().friday.at("16:00").do(self.stop_trading)
        
        # Health check every 2 minutes (catches early closes faster)
        schedule.every(2).minutes.do(self.check_process_health)
        
        logger.info("Market hours scheduled:")
        logger.info("- Market Open: Monday-Friday 9:30 AM SYSTEM TIME")
        logger.info("- Market Close: Monday-Friday 4:00 PM SYSTEM TIME")
        logger.info("- Health checks every 2 minutes (handles early closes)")
        logger.info("- NOTE: Ensure Docker container timezone matches Eastern Time!")

        try:
            market_open_time, market_close_time = self.get_market_hours()
            if market_open_time and market_close_time:
                logger.info(f"- Today's actual hours: {market_open_time} - {market_close_time}")
            else:
                logger.info("- Today is not a trading day")
        except Exception as e:
            logger.warning(f"Could not get today's market hours: {e}")

    def run(self):
        """Main scheduler loop"""
        logger.info("🚀 Market Scheduler starting...")
        logger.info(f"Current time: {datetime.now(self.eastern).strftime('%Y-%m-%d %H:%M:S %Z')}")

        self.schedule_market_hours()

        if self.is_market_hours():
            logger.info("Market is currently open, starting trading immediately")
            self.start_trading()

        else:
            logger.info("Market is currently closed")
            if self.is_market_day():
                now = datetime.now(self.eastern)
                market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
                if now < market_open:
                    logger.info(f"Next market open: Today at {market_open.strftime('%H:%M:%S')}")
                else:
                    logger.info("Market closed for today")
            else:
                logger.info("Not a trading day")

        try:
            while True:
                schedule.run_pending()
                time_module.sleep(60)

        except KeyboardInterrupt:
            logger.info("Scheduler interrupted by user")
        except Exception as e:
            logger.error(f"Scheduler error: {str(e)}")
        finally:
            self.stop_trading()
            logger.info("Market Scheduler stopped")



if __name__ == "__main__":
    scheduler = MarketScheduler()
    scheduler.run()
    
