# Step 1: Remove all conflicting packages thoroughly
pip uninstall -y google-ai-generativelanguage google-generativeai langchain-google-genai \
  async-timeout fsspec google-genai

# Step 2: Clear pip cache to avoid cached conflicts
pip cache purge

# Step 3: Install langchain-google-genai and let it pull compatible versions
pip install -q langchain-google-genai==2.1.2

# Step 4: Install remaining dependencies
pip install -q youtube-transcript-api google-api-python-client chromadb \
  langchain==0.3.24 langchain_community==0.3.22 sentence-transformers \
  langgraph==0.3.21 langgraph-prebuilt==0.1.7 alpaca-py \
  "async-timeout==4.0.0" "fsspec==2024.10.0"