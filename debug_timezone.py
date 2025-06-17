#!/usr/bin/env python3
"""
Test script to debug the timezone conversion issue
"""

import pandas as pd
import pytz
from datetime import datetime
import sys

# Add the project directory to the path
sys.path.append('.')

try:
    from data_util import get_alpaca_data
except ImportError as e:
    print(f"❌ Error importing: {e}")
    sys.exit(1)

def test_timezone_conversion():
    """Test the exact timezone conversion that happens in the simulator"""
    
    eastern = pytz.timezone('US/Eastern')
    
    print("🔍 TESTING TIMEZONE CONVERSION")
    print("=" * 50)
    
    # Simulate the historical data endpoint (10:49 AM Eastern)
    historical_time_str = "2025-06-17 10:49:00"
    historical_time = eastern.localize(datetime.strptime(historical_time_str, '%Y-%m-%d %H:%M:%S'))
    
    print(f"📊 Historical data last timestamp: {historical_time}")
    print(f"📊 Historical timezone: {historical_time.tzinfo}")
    print(f"📊 Historical in UTC: {historical_time.astimezone(pytz.UTC)}")
    
    print(f"\n🔄 Calling get_alpaca_data with start_date='{historical_time_str}'...")
    
    # This is the exact call that happens in your simulator
    fresh_data = get_alpaca_data('SPY', start_date=historical_time_str, end_date=None, store_in_db=False)
    
    if fresh_data is None or fresh_data.empty:
        print("❌ No fresh data returned")
        return
    
    print(f"\n📊 Fresh data returned:")
    print(f"   Shape: {fresh_data.shape}")
    print(f"   First timestamp: {fresh_data.index[0]}")
    print(f"   Last timestamp: {fresh_data.index[-1]}")
    print(f"   Timezone: {fresh_data.index.tz}")
    
    # Now simulate the conversion that happens in get_next_update
    print(f"\n🔄 Applying timezone conversion (simulating get_next_update logic):")
    
    if fresh_data.index.tz is None:
        print("   Converting from naive to UTC then Eastern...")
        fresh_data.index = pd.to_datetime(fresh_data.index, utc=True).tz_convert(eastern)
    elif fresh_data.index.tz != eastern:
        print(f"   Converting from {fresh_data.index.tz} to {eastern}...")
        fresh_data.index = fresh_data.index.tz_convert(eastern)
    else:
        print("   Already in Eastern timezone")
    
    print(f"\n📊 After conversion:")
    print(f"   First timestamp: {fresh_data.index[0]}")
    print(f"   Last timestamp: {fresh_data.index[-1]}")
    print(f"   Timezone: {fresh_data.index.tz}")
    
    # Calculate the gap
    time_gap = fresh_data.index[0] - historical_time
    gap_minutes = time_gap.total_seconds() / 60
    
    print(f"\n⏰ TIME GAP ANALYSIS:")
    print(f"   Historical end: {historical_time}")
    print(f"   Fresh data start: {fresh_data.index[0]}")
    print(f"   Gap: {time_gap}")
    print(f"   Gap in minutes: {gap_minutes:.1f}")
    
    if abs(gap_minutes) < 5:
        print("   ✅ Gap is acceptable (< 5 minutes)")
    else:
        print(f"   ❌ Gap is too large ({gap_minutes:.1f} minutes)")
        
        # Try to identify the issue
        print(f"\n🔍 DEBUGGING THE GAP:")
        
        # Check if this is a UTC/Eastern conversion issue
        historical_utc = historical_time.astimezone(pytz.UTC)
        fresh_utc = fresh_data.index[0].astimezone(pytz.UTC)
        utc_gap = fresh_utc - historical_utc
        utc_gap_minutes = utc_gap.total_seconds() / 60
        
        print(f"   Historical in UTC: {historical_utc}")
        print(f"   Fresh data in UTC: {fresh_utc}")
        print(f"   UTC gap: {utc_gap} ({utc_gap_minutes:.1f} minutes)")
        
        if abs(utc_gap_minutes) < 5:
            print("   ✅ UTC times align - this was a timezone display issue")
        else:
            print("   ❌ UTC times don't align - this is a real data gap")

def test_data_concatenation():
    """Test what happens when we concatenate historical and fresh data"""
    
    eastern = pytz.timezone('US/Eastern')
    
    print(f"\n\n🔗 TESTING DATA CONCATENATION")
    print("=" * 50)
    
    # Create mock historical data (Eastern timezone)
    historical_times = pd.date_range(
        start='2025-06-17 10:45:00', 
        end='2025-06-17 10:49:00', 
        freq='1min', 
        tz=eastern
    )
    historical_data = pd.DataFrame({
        'Close': [600.0 + i for i in range(len(historical_times))],
        'Volume': [1000] * len(historical_times)
    }, index=historical_times)
    
    print(f"📊 Mock historical data:")
    print(f"   Range: {historical_data.index[0]} to {historical_data.index[-1]}")
    print(f"   Timezone: {historical_data.index.tz}")
    
    # Get real fresh data
    fresh_data = get_alpaca_data('SPY', start_date='2025-06-17 10:49:00', end_date=None, store_in_db=False)
    
    if fresh_data is None or fresh_data.empty:
        print("❌ No fresh data for concatenation test")
        return
    
    print(f"\n📊 Fresh data before conversion:")
    print(f"   Range: {fresh_data.index[0]} to {fresh_data.index[-1]}")
    print(f"   Timezone: {fresh_data.index.tz}")
    
    # Apply timezone conversion
    if fresh_data.index.tz is None:
        fresh_data.index = pd.to_datetime(fresh_data.index, utc=True).tz_convert(eastern)
    elif fresh_data.index.tz != eastern:
        fresh_data.index = fresh_data.index.tz_convert(eastern)
    
    print(f"\n📊 Fresh data after conversion:")
    print(f"   Range: {fresh_data.index[0]} to {fresh_data.index[-1]}")
    print(f"   Timezone: {fresh_data.index.tz}")
    
    # Test concatenation
    print(f"\n🔗 Concatenating data...")
    combined_data = pd.concat([historical_data, fresh_data[['Close', 'Volume']]]).drop_duplicates()
    combined_data = combined_data.sort_index()
    
    print(f"📊 Combined data:")
    print(f"   Range: {combined_data.index[0]} to {combined_data.index[-1]}")
    print(f"   Total points: {len(combined_data)}")
    
    # Check for gaps
    time_diffs = combined_data.index.to_series().diff()
    large_gaps = time_diffs[time_diffs > pd.Timedelta(minutes=5)]
    
    if len(large_gaps) == 0:
        print("   ✅ No large gaps after concatenation")
    else:
        print(f"   ❌ Found {len(large_gaps)} large gaps:")
        for timestamp, gap in large_gaps.items():
            prev_idx = combined_data.index.get_loc(timestamp) - 1
            prev_timestamp = combined_data.index[prev_idx]
            gap_minutes = gap.total_seconds() / 60
            print(f"      {prev_timestamp} → {timestamp} ({gap_minutes:.1f} minutes)")

if __name__ == "__main__":
    test_timezone_conversion()
    test_data_concatenation()
