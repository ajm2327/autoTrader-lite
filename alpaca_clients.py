# clients.py
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.trading.client import TradingClient
from config import ALPACA_API_KEY, ALPACA_API_SECRET

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import ToolNode

# Create clients once at module level
data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)

trading_client = TradingClient(
    api_key=ALPACA_API_KEY, 
    secret_key=ALPACA_API_SECRET, 
    paper=True
)

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")


def get_llm_with_tools(tools):
    """Create LLM with tools bound"""
    return llm.bind_tools(tools)

def get_tool_node(tools):
    """Create a ToolNode with the provided tools"""
    return ToolNode(tools)