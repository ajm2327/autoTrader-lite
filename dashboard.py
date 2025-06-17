import streamlit as st
import time
from datetime import datetime
import os
import pandas as pd
import matplotlib.pyplot as plt
from plotting_utils import plot_technical_indicators, show_data_summary
import re

st.set_page_config(page_title='Lite Trading Agent', layout='wide')

st.title('Trading Agent Dashboard')
st.markdown('---')

log_col, plot_col = st.columns([1,1])

def parse_latest_messages(log_content):
    """
    Parse the log content to extract the latest data update and agent decision
    """
    # Split by the data update markers
    data_updates = log_content.split("🔄 DATA UPDATE SENT TO AGENT:")
    agent_decisions = log_content.split("🧠 AGENT DECISION:")
    
    latest_data_update = None
    latest_agent_decision = None
    
    # Get latest data update
    if len(data_updates) > 1:
        # Take the last data update section
        last_data_section = data_updates[-1]
        
        # Find the end marker - either the agent decision or the question
        agent_decision_pos = last_data_section.find("🧠 AGENT DECISION:")
        question_pos = last_data_section.find("Based on this data, what is your next decision?")
        
        # Use whichever comes first (or end of content if neither found)
        end_pos = len(last_data_section)
        if agent_decision_pos != -1:
            end_pos = min(end_pos, agent_decision_pos)
        if question_pos != -1:
            end_pos = min(end_pos, question_pos)
            
        latest_data_update = last_data_section[:end_pos].strip()
    
    # Get latest agent decision (keep original logic)
    if len(agent_decisions) > 1:
        # Take the last decision section
        last_decision_section = agent_decisions[-1]
        # Extract content between the equals signs
        equals_pattern = r'={50}(.*?)={50}'
        match = re.search(equals_pattern, last_decision_section, re.DOTALL)
        if match:
            latest_agent_decision = match.group(1).strip()
        else:
            # Fallback: take content until next data update or end
            end_marker = last_decision_section.find("🔄 DATA UPDATE SENT TO AGENT:")
            if end_marker == -1:
                latest_agent_decision = last_decision_section.strip()
            else:
                latest_agent_decision = last_decision_section[:end_marker].strip()
    
    return latest_data_update, latest_agent_decision

def is_simulation_running():
    """
    Check if simulation is currently running by looking for recent activity
    """
    if not os.path.exists('live_trading.log'):
        return False
    
    try:
        # Check if log file was modified recently (within last 30 seconds)
        log_mtime = os.path.getmtime('live_trading.log')
        current_time = time.time()
        return (current_time - log_mtime) < 30
    except:
        return False

with log_col:
    st.subheader("Latest Trading Activity")
    log_file = 'live_trading.log'
    
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            log_content = f.read()
        
        if log_content.strip():
            latest_data_update, latest_agent_decision = parse_latest_messages(log_content)
            
            if latest_data_update:
                st.markdown("### 📊 Latest Market Data")
                st.markdown(f"```\n{latest_data_update}\n```")
            
            if latest_agent_decision:
                st.markdown("### 🧠 Latest Agent Decision")
                st.markdown(latest_agent_decision)
            
            if not latest_data_update and not latest_agent_decision:
                st.info("No recent trading activity found in logs")
        else:
            st.info("Log file is empty - waiting for simulation to start...")
    else:
        st.info("Waiting for simulation to start...")

with plot_col:
    st.subheader("Current Market Data")
    
    # Check if simulation is running and data file exists
    if is_simulation_running() and os.path.exists('current_data_chunk.csv'):
        try:
            current_data = pd.read_csv('current_data_chunk.csv', index_col=0, parse_dates=True)
            fig = plot_technical_indicators(current_data, "SPY")
            st.pyplot(fig)
            plt.close(fig)

            st.text(f"Data points: {len(current_data)}")
            st.text(f"Current price: ${current_data['Close'].iloc[-1]:.2f}")
            st.text(f"Time range: {current_data.index[0]} to {current_data.index[-1]}")
        except Exception as e:
            st.error(f"Error loading agent data: {str(e)}")
    else:
        # Clean up old data file if simulation isn't running
        if os.path.exists('current_data_chunk.csv') and not is_simulation_running():
            try:
                os.remove('current_data_chunk.csv')
            except:
                pass
        
        st.info("No active simulation detected")
        st.markdown("""
        **To start the simulation:**
        1. Ensure `python market_scheduler.py` in your terminal
        2. The dashboard will automatically update when data becomes available during market hours
        
        **Status:** Waiting for trading agent to start...
        """)

# Auto-refresh every 2 seconds
time.sleep(2)
st.rerun()