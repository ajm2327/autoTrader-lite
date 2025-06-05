import streamlit as st
import time
from datetime import datetime
import os

st.set_page_config(page_title='Lite Trading Agent', layout='wide')

st.title('Trading Agent Dashboard')
st.markdown('---')

col1, col2 = st.columns([1,3])
with col1:
    st.metric('Status', '🧾 Running ')
with col2:
    st.metric('Last Update', datetime.now().strftime('%H:%M:%S'))

log_file = 'live_trading.log'
if os.path.exists(log_file):
    with open(log_file, 'r') as f:
        log_content = f.read()

    st.text_area(
        "Console Output",
        value = log_content,
        height=600,
        key='console'
    )
else:
    st.info("Wating for simulation to start...")

time.sleep(2)
st.rerun()