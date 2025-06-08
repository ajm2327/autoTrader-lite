import streamlit as st
import time
from datetime import datetime
import os
import pandas as pd
import matplotlib.pyplot as plt
from plotting_utils import plot_technical_indicators, show_data_summary
import html

st.set_page_config(page_title='Lite Trading Agent', layout='wide')

st.title('Trading Agent Dashboard')
st.markdown('---')

log_col, plot_col = st.columns([1,1])

with log_col:
    st.subheader("Live Trading Log")
    log_file = 'live_trading.log'
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            log_content = f.read()

        st.markdown("""
        <style>
        .scroll-area {
            height: 600px;
            overflow-y: scroll;
            padding: 10px;
            border: 1px solid #ddd;
        }
        .scroll-are pre {
            font-size: 14px;
            font-family: monospace;
            margin: 0;
            white-space: pre-wrap;
        }
        </style>
        """, unsafe_allow_html=True)

        escaped_content = html.escape(log_content)
        st.markdown(f'<div class="scroll-area"><pre>{escaped_content}</pre></div>', unsafe_allow_html=True)
    else:
        st.info("Waiting for simulation to start...")

with plot_col:
    st.subheader("Current Market Data")
    try:
        # recent spy data from cache
        if os.path.exists('current_data_chunk.csv'):
            current_data = pd.read_csv('current_data_chunk.csv', index_col=0, parse_dates=True)
            fig = plot_technical_indicators(current_data, "SPY")
            st.pyplot(fig)
            plt.close(fig)

            st.text(f"Data points: {len(current_data)}")
            st.text(f"Current price: ${current_data['Close'].iloc[-1]:.2f}")
            st.text(f"Time range: {current_data.index[0]} to {current_data.index[-1]}")
        else:
            st.info("Waiting for agent data...")
    except Exception as e:
        st.error(f"Error loading agent data: {str(e)}")


st.markdown("""
<script>
setTimeout(function() {
    var scrollArea = document.querySelector('.scroll-area');
    if (scrollArea) {
        scrollArea.scrollTop = scrollArea.scrollHeight;
    }
}, 100);
</script>
""", unsafe_allow_html=True)
time.sleep(2)
st.rerun()