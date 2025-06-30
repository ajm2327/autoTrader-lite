# autoTrader-lite
This is a demo of my decision agent intended as a portfolio piece. Within the next day or two I intend to refactor my decision agent and LSTM code to create a single stock focused, ai powered trading agent.
Well that took a bit longer than expected but it was also because i'm a perfectionist i guess. I suppose my weaknesses are that I am a perfectionist and I don't document things as I go. I know it should be standard practice to write documentation and reports as you go but when I'm not in a professional setting I just start writing and testing pretty much like brute force style. I write the code, run the main, modify the code, run the main/deploy container again. My testing style has always just been building the use flow that I have in mind and validation is just running through that flow to see if it can be used from like start to finish with no problems. I digress, this is the flow of my project, welcome!!! 

# Introducing the autoTrader-lite AI powered single stock focused trading agent
### Maybe the headings could use some work
Since I was young I have had an interest in the stock market because its something you always here about with the economy and how people get rich, and investing. Once I was older I started trading on Robinhood when that was the cool new thing, but each time I attempted to trade, I would basically have a lucky streak, gain a lot of profit, but when I start scaling my trades I lose it all because I don't take losses easy. That's a feeling that most people who take an interest in investing and trading are familiar with. While I have the knowledge for trading, but emotions are a large hurdle when it comes to scaling and risk management, so why not make something emotionless do it? 

# Langchain Agents
For Kaggle's 5 day gen AI course, I participated in their capstone project, demonstrating a use case with AI. In the course, I learned about agentic AI, artificial intelligence that can use tools. They are useful in digital workforce tasks much like the chatbots that are the first line of support before you speak to a real person. 
Well Google's gemini and other generative AI are becoming more advanced and fast acting, so a perfect use case for an AI that can use tools is to see if it can beat the market. The langgraph connects separate components that are typical in an agent system. 

### For a normal AI chatbot, a node graph would look like:
AI < -- > Human

* Where both the AI and human can communicate with each other.

### For an AI agent its node graph would look like:

AI < -- > Human
AI < -- > Tools

* Where both the AI and the human can talk to each other,
* But the AI can also communicate with a tools node which contains all of the tools that it can use. The tool node communicates the output/responses of those tools to the model.

### For the trading AI agent, its graph looks like:

AI < -- > Data
AI < -- > Tools

* So the human has been replaced with the data feed, there is no human input, aside from the occasional message injection. The data updates on intervals, automating the message function that would be used for a human to message the AI. 



The main AI driven component is the Langchain library, refer to the requirements.txt file to review all dependencies. It is a nodegraph set up, so the AI node is what invokes Gemini, specifically 'gemini-2.0-flash', which is available within a free tier without needing to pass an API key to invoke it, but there is a rate limit of 200 requests per day. 

# Google Gemini Selection
Gemini-2.0-flash was chosen for this use-case because this model is purposed for agentic tasks. It is also a stable build of a gen ai model 

## Rate limiting
The gemini api has a free tier to use their models with a 200 requests per day (RPD) limit and 15 requests per minute (RPM) limit. If this project were to be run as a demonstration, the `interval_seconds` variable will need to be at least 4 seconds to avoid the RPM limit. The `max_iterations` variable will need to be set to at most 200 to avoid the RPD limit. If this trading system were to be used consistently/extended periods, you will need an API key from google cloud console, and enabled billing for higher rate limits. If running without an API key, ensure that the API key being passed into the LLM invocation is commented out.

Visit the google gemini API documentation:
Visit the langchain documentation:

# Alpaca Py
Alpaca is an online trading broker like Robinhood, Webull, TradingView, and many others. The nice part about Alpaca is their API availability and their paper trading accounts, all available within a free tier, with exceptional depth of data. The marketfeed is called using IEX because SIP data has a 15 minute delay for real time data. IEX provides up to minute data, and has SPY and most large name tickers available to trade. 

## Alpaca Py API key required
In order to access Alpaca's API endpoints, an API key and secret is required. Make sure the API credentials are for a paper trading account, API credentials for a market account require personal funds.

View this tutorial for setting up an Alpaca account and obtaining paper trading API credentials:
Visit the Alpaca website to obtain API credentials:
Visit the Alpaca-py documentation:

# .env file configuration:
config.py sets up accessing the APIs used in this project, accessed through a .env file. 

You will need a .env file with these credentials, the GEMINI API KEY is not needed if you comment out its use in decision_util.py

## .env
ALPACA_API_KEY =
ALPACA_API_SECRET =
GOOGLE_API_KEY =

The app can be run as a docker-container.
Docker and docker-compose are required to be installed on your system. The project already has Dockerfile and docker-compose.yml. In your terminal within the project, run:

`docker-compose up --build -d`
This will build and run the docker container. Two applications are run, historical_data_simulator.py and dashboard.py, view the dashboard on port 8501. It is a basic dashboard with the current data feed chunk, the current agent decision, and a visualization of the incoming stock data. The dashboard is on a slight delay from the model's current activity, processing the logged output from historical_data_simulator.py. The dashboard also has a text input box so that you can inject a question into its next data update, the AI model will respond in the next turn. Currently right now, the dashboard is automatically updating, so if you send the model a message, it could refresh before you can read its response. The purpose of the model is to focus on the data feed, not human responses. The dashboard can be stopped using the stop button at the top right, so that you can view a message.

To run the project without a container, install the requirements.txt file and in your python environment run:
`python3 historical_data_simulator.py`
If running for the first time the current main function call begins the historical data in March 1st, 2025, up to today. If building a new database, this will take time to write the data to the database, and train the LSTM. 
Running historical_data_simulator.py is a minimal way to view the agent's activity because all of its decisions and data feed is printed to the terminal, as well as logged. Dashboard.py reads the same output that is shown on the terminal as well as a visualization of the data. 

If you want to view dashboard.py, in a separate terminal, in the same project directory, run:
`streamlit run dashboard.py`

Then view the dashboard in your browser on port 8501.

