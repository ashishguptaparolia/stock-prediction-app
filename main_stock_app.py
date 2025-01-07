import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests

# App title
st.title("Enhanced Stock Analysis and Prediction")

# Sidebar inputs
st.sidebar.header("Select Parameters")
stock_symbol = st.sidebar.text_input("Stock Ticker (e.g., AAPL, TSLA, NSE:TCS)", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-12-31"))

# News API Key
NEWS_API_KEY = "364b5eac98da4a31ac519a8d67581444"

# Function to dynamically handle columns
def extract_columns(data, stock_symbol):
    """
    Handles column names for single-stock or multi-stock DataFrames.
    Dynamically renames key columns (Close, Open, High, Low, Volume).
    """
    if isinstance(data.columns, pd.MultiIndex):  # Multi-level columns
        data.columns = ['_'.join(col).strip() for col in data.columns.values]
    
    # Identify columns dynamically based on stock symbol or default names
    columns_mapping = {}
    for col in data.columns:
        if 'Close' in col and stock_symbol in col:
            columns_mapping[col] = 'Close'
        elif 'Open' in col and stock_symbol in col:
            columns_mapping[col] = 'Open'
        elif 'High' in col and stock_symbol in col:
            columns_mapping[col] = 'High'
        elif 'Low' in col and stock_symbol in col:
            columns_mapping[col] = 'Low'
        elif 'Volume' in col and stock_symbol in col:
            columns_mapping[col] = 'Volume'

    # Rename columns
    data.rename(columns=columns_mapping, inplace=True)
    return data

# Function to calculate technical indicators
def calculate_indicators(data):
    """
    Adds Bollinger Bands and RSI indicators to the DataFrame.
    """
    # Bollinger Bands
    data['SMA20'] = data['Close'].rolling(window=20).mean()
    data['Upper Band'] = data['SMA20'] + 2 * data['Close'].rolling(window=20).std()
    data['Lower Band'] = data['SMA20'] - 2 * data['Close'].rolling(window=20).std()

    # RSI Calculation
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))

    return data

# Function for news sentiment analysis
def fetch_news_sentiment(symbol):
    """
    Fetches sentiment data for the given stock symbol using NewsAPI.
    """
    url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    sentiment = {"positive": 0, "neutral": 0, "negative": 0}
    
    if response.status_code == 200:
        articles = response.json().get("articles", [])
        for article in articles:
            title = article.get('title', '').lower()
            if "good" in title or "positive" in title or "gain" in title:
                sentiment["positive"] += 1
            elif "bad" in title or "negative" in title or "loss" in title:
                sentiment["negative"] += 1
            else:
                sentiment["neutral"] += 1
    
    return sentiment

# Main App Functionality
if st.button("Run Analysis"):
    try:
        # Fetch stock data
        data = yf.download(stock_symbol, start=start_date, end=end_date)

        if data.empty:  # Explicit check for empty DataFrame
            st.warning("No data found for the selected ticker and date range.")
            st.stop()

        # Clean and standardize columns
        data = extract_columns(data, stock_symbol)

        # Ensure the "Close" column is present
        if "Close" not in data.columns:
            st.error(f"Expected 'Close' column not found in data for {stock_symbol}.")
            st.stop()

        # Display raw data
        st.subheader(f"{stock_symbol} Historical Data")
        st.write(data.tail())

        # Calculate technical indicators
        data = calculate_indicators(data)

        # Plot Closing Price with Bollinger Bands
        plt.figure(figsize=(12, 6))
        plt.plot(data.index, data['Close'], label='Closing Price', color='blue')
        plt.plot(data.index, data['Upper Band'], label='Upper Bollinger Band', color='red')
        plt.plot(data.index, data['Lower Band'], label='Lower Bollinger Band', color='green')
        plt.fill_between(data.index, data['Upper Band'], data['Lower Band'], color='gray', alpha=0.2)
        plt.title(f"{stock_symbol} Closing Price with Bollinger Bands")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.legend()
        st.pyplot(plt)

        # Plot RSI
        plt.figure(figsize=(12, 4))
        plt.plot(data.index, data['RSI'], label='RSI', color='orange')
        plt.axhline(70, color='red', linestyle='--', label='Overbought (70)')
        plt.axhline(30, color='green', linestyle='--', label='Oversold (30)')
        plt.title("Relative Strength Index (RSI)")
        plt.xlabel("Date")
        plt.ylabel("RSI")
        plt.legend()
        st.pyplot(plt)

        # Fetch news sentiment
        sentiment = fetch_news_sentiment(stock_symbol)
        st.subheader(f"News Sentiment for {stock_symbol}")
        st.write(f"**Positive Articles:** {sentiment['positive']}")
        st.write(f"**Neutral Articles:** {sentiment['neutral']}")
        st.write(f"**Negative Articles:** {sentiment['negative']}")

        # Summary
        st.subheader("Key Insights")
        latest_rsi = data['RSI'].iloc[-1]
        if latest_rsi > 70:
            st.write(f"RSI indicates that {stock_symbol} is currently **overbought**.")
        elif latest_rsi < 30:
            st.write(f"RSI indicates that {stock_symbol} is currently **oversold**.")
        else:
            st.write(f"RSI indicates that {stock_symbol} is in a **neutral** zone.")
        
    except Exception as e:
        st.error(f"An error occurred: {e}")
