#!/usr/bin/env python3
"""
Test script to check for gaps in Alpaca data retrieval
"""

import pandas as pd
import pytz
from datetime import datetime, timedelta
import sys
import os

# Add the project directory to the path so we can import our modules
sys.path.append('.')

try:
    from data_util import get_alpaca_data
except ImportError as e:
    print(f"❌ Error importing get_alpaca_data: {e}")
    print("Make sure you're running this from the project root directory")
    sys.exit(1)

def test_alpaca_data_gaps():
    """Test for gaps in Alpaca data retrieval"""
    
    eastern = pytz.timezone('US/Eastern')
    today = datetime.now(eastern).date()
    
    print(f"🔍 TESTING ALPACA DATA RETRIEVAL FOR {today}")
    print("=" * 60)
    
    # Test 1: Full day data
    print("\n📊 Test 1: Getting full day data...")
    start_date = today.strftime('%Y-%m-%d')
    
    try:
        full_day_data = get_alpaca_data('SPY', start_date=start_date, end_date=None, store_in_db=False)
        
        if full_day_data is None or full_day_data.empty:
            print("❌ No data returned for full day")
            return
            
        print(f"✅ Retrieved {len(full_day_data)} data points")
        print(f"📅 Date range: {full_day_data.index.min()} to {full_day_data.index.max()}")
        
        # Check for gaps in the data
        check_for_gaps(full_day_data, "Full Day")
        
    except Exception as e:
        print(f"❌ Error getting full day data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 2: Specific time range that caused the gap
    print(f"\n📊 Test 2: Getting data from 10:49 AM to now...")
    
    try:
        start_time = f"{today.strftime('%Y-%m-%d')} 10:49:00"
        gap_test_data = get_alpaca_data('SPY', start_date=start_time, end_date=None, store_in_db=False)
        
        if gap_test_data is None or gap_test_data.empty:
            print("❌ No data returned for specific time range")
            return
            
        print(f"✅ Retrieved {len(gap_test_data)} data points")
        print(f"📅 Date range: {gap_test_data.index.min()} to {gap_test_data.index.max()}")
        
        # Check for gaps in the data
        check_for_gaps(gap_test_data, "10:49 AM to Now")
        
    except Exception as e:
        print(f"❌ Error getting gap test data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 3: Hour-by-hour breakdown to find where gaps occur
    print(f"\n📊 Test 3: Hour-by-hour breakdown...")
    
    market_hours = [
        ("09:30", "10:30"),
        ("10:30", "11:30"), 
        ("11:30", "12:30"),
        ("12:30", "13:30"),
        ("13:30", "14:30"),
        ("14:30", "15:30"),
        ("15:30", "16:00")
    ]
    
    for start_hour, end_hour in market_hours:
        try:
            start_time = f"{today.strftime('%Y-%m-%d')} {start_hour}:00"
            end_time = f"{today.strftime('%Y-%m-%d')} {end_hour}:00"
            
            hourly_data = get_alpaca_data('SPY', start_date=start_time, end_date=end_time, store_in_db=False)
            
            if hourly_data is None or hourly_data.empty:
                print(f"❌ {start_hour}-{end_hour}: No data")
            else:
                print(f"✅ {start_hour}-{end_hour}: {len(hourly_data)} points | {hourly_data.index.min().strftime('%H:%M')} to {hourly_data.index.max().strftime('%H:%M')}")
                
        except Exception as e:
            print(f"❌ {start_hour}-{end_hour}: Error - {e}")

def check_for_gaps(data, test_name):
    """Check for gaps in minute-by-minute data"""
    print(f"\n🔍 Gap Analysis for {test_name}:")
    
    if len(data) < 2:
        print("  ⚠️ Not enough data to check for gaps")
        return
    
    # Convert to Eastern time if needed
    eastern = pytz.timezone('US/Eastern')
    if data.index.tz is None:
        data.index = pd.to_datetime(data.index).tz_localize('UTC').tz_convert(eastern)
    elif data.index.tz != eastern:
        data.index = data.index.tz_convert(eastern)
    
    # Check for gaps larger than 1 minute
    time_diffs = data.index.to_series().diff()
    large_gaps = time_diffs[time_diffs > pd.Timedelta(minutes=2)]
    
    if len(large_gaps) == 0:
        print("  ✅ No significant gaps found!")
    else:
        print(f"  ⚠️ Found {len(large_gaps)} gaps larger than 2 minutes:")
        for timestamp, gap in large_gaps.items():
            prev_timestamp = data.index[data.index.get_loc(timestamp) - 1]
            gap_minutes = gap.total_seconds() / 60
            print(f"    Gap: {prev_timestamp.strftime('%H:%M:%S')} → {timestamp.strftime('%H:%M:%S')} ({gap_minutes:.1f} minutes)")
    
    # Check for expected market hours coverage
    market_start = data.index[0].replace(hour=9, minute=30, second=0, microsecond=0)
    market_end = data.index[0].replace(hour=16, minute=0, second=0, microsecond=0)
    
    data_start = data.index.min()
    data_end = data.index.max()
    
    print(f"  📊 Coverage: {data_start.strftime('%H:%M')} to {data_end.strftime('%H:%M')}")
    
    if data_start > market_start:
        missing_start = (data_start - market_start).total_seconds() / 60
        print(f"  ⚠️ Missing {missing_start:.0f} minutes from market open")
    
    if data_end < market_end and datetime.now(eastern).time() > market_end.time():
        missing_end = (market_end - data_end).total_seconds() / 60
        print(f"  ⚠️ Missing {missing_end:.0f} minutes before market close")

def main():
    """Main test function"""
    eastern = pytz.timezone('US/Eastern')
    now = datetime.now(eastern)
    
    # Check if market is open or was open today
    if now.weekday() >= 5:
        print("⚠️ Weekend - no market data expected")
        return
    
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    
    if now < market_open:
        print("⚠️ Market hasn't opened yet today")
        return
    
    print(f"🕐 Current time: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    
    test_alpaca_data_gaps()
    
    print("\n" + "=" * 60)
    print("✅ Test completed!")

if __name__ == "__main__":
    main()
