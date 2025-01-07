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
sector = st.sidebar.selectbox("Select Sector for Comparison", ["Technology", "Healthcare", "Finance", "Energy"])
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-12-31"))

# Function to calculate technical indicators
def calculate_technical_indicators(data):
    # RSI
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    rs = rs.fillna(0)  # Avoid NaN
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # ADX
    high_low = data['High'] - data['Low']
    high_close = abs(data['High'] - data['Close'].shift(1))
    low_close = abs(data['Low'] - data['Close'].shift(1))
    tr = high_low.combine(high_close, max).combine(low_close, max)
    data['TR'] = tr
    data['+DM'] = data['High'].diff().where(data['High'].diff() > data['Low'].diff(), 0)
    data['-DM'] = data['Low'].diff().where(data['Low'].diff() > data['High'].diff(), 0)
    tr_sma = data['TR'].rolling(window=14).mean()
    plus_dm_sma = data['+DM'].rolling(window=14).mean()
    minus_dm_sma = data['-DM'].rolling(window=14).mean()
    data['+DI'] = (plus_dm_sma / tr_sma) * 100
    data['-DI'] = (minus_dm_sma / tr_sma) * 100
    data['ADX'] = (abs(data['+DI'] - data['-DI']) / (data['+DI'] + data['-DI'])).fillna(0) * 100
    
    return data

# Function for news sentiment analysis
def fetch_news_sentiment(symbol):
    url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey=364b5eac98da4a31ac519a8d67581444"
    response = requests.get(url)
    sentiment = {"positive": 0, "neutral": 0, "negative": 0}
    
    if response.status_code == 200:
        articles = response.json().get("articles", [])
        for article in articles:
            title = article.get('title', '').lower()
            if "good" in title or "positive" in title:
                sentiment["positive"] += 1
            elif "bad" in title or "negative" in title:
                sentiment["negative"] += 1
            else:
                sentiment["neutral"] += 1
    
    return sentiment

# Function to fetch sector performance
def fetch_sector_performance(sector_name):
    # Simulated data (API or real data can replace this)
    sector_data = {
        "Technology": 1.5,
        "Healthcare": 0.9,
        "Finance": 1.2,
        "Energy": 0.8
    }
    return sector_data.get(sector_name, 1)

# Run analysis
if st.button("Run Enhanced Analysis"):
    try:
        # Fetch stock data
        data = yf.download(stock_symbol, start=start_date, end=end_date)
        
        # Check if data is empty
        if data.empty:
            st.warning("No data found for the selected ticker and date range.")
            st.stop()
        
        st.subheader(f"{stock_symbol} Historical Data")
        st.write(data.tail())
        
        # Technical indicators
        data = calculate_technical_indicators(data)
        
        # Plot RSI
        plt.figure(figsize=(12, 4))
        plt.plot(data['RSI'], label='RSI', color='orange')
        plt.axhline(70, color='red', linestyle='--', label='Overbought (70)')
        plt.axhline(30, color='green', linestyle='--', label='Oversold (30)')
        plt.title("RSI (Relative Strength Index)")
        plt.xlabel("Date")
        plt.ylabel("RSI")
        plt.legend()
        st.pyplot(plt)
        
        # Plot ADX
        plt.figure(figsize=(12, 4))
        plt.plot(data['ADX'], label='ADX', color='purple')
        plt.title("ADX (Trend Strength)")
        plt.xlabel("Date")
        plt.ylabel("ADX")
        plt.legend()
        st.pyplot(plt)
        
        # News sentiment
        sentiment = fetch_news_sentiment(stock_symbol)
        st.subheader(f"News Sentiment for {stock_symbol}")
        st.write(f"Positive Articles: {sentiment['positive']}")
        st.write(f"Neutral Articles: {sentiment['neutral']}")
        st.write(f"Negative Articles: {sentiment['negative']}")
        
        # Sector performance
        sector_performance = fetch_sector_performance(sector)
        st.subheader(f"{sector} Sector Performance")
        st.write(f"The sector performance index is: {sector_performance}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
