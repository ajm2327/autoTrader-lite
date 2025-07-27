import pytest
import os
import tempfile
import sys

# add project root to path
sys.path.append('.')
from historical_data_simulator import HistoricalDataSimulator

def test_initialize_log_files_creates_files():
    """Test that _initialize_log_files actually creates the log files"""

    # SETUP create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # create simulator instance pointing to temp directory
        simulator = HistoricalDataSimulator(
            ticker="SPY",
            start_date="2025-07-21",
            end_date="2025-07-22",
            log_dir=temp_dir
        )

        # EXECUTE call iniitialize logs to test
        simulator._initialize_log_files()

        # ASSERT check it made log files
        assert os.path.exists(simulator.trade_log_file)
        assert os.path.exists(simulator.decision_log_file)
        print("✅ Log files created successfully")