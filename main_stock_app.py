import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests

# Your News API Key
NEWS_API_KEY = "364b5eac98da4a31ac519a8d67581444"

# App title
st.set_page_config(page_title="Advanced Stock, Crypto, and Investment Insights", layout="wide")
st.title("\ud83d\udcc8 Advanced Stock, Crypto, and Investment Insights")

# Sidebar inputs
st.sidebar.header("Select Parameters")
asset_type = st.sidebar.selectbox("Asset Type", ["Stock", "Crypto"])
ticker_symbol = st.sidebar.text_input("Ticker Symbol", value="AAPL")
features = st.sidebar.multiselect("Select Features", ["Bollinger Bands", "RSI", "Sentiment Analysis", "Seasonality Insights"], default=["Bollinger Bands", "RSI"])

# Fetch Data
def fetch_data(symbol, asset_type="Stock"):
    if asset_type == "Stock":
        data = yf.download(symbol, period="5y", progress=False)
    elif asset_type == "Crypto":
        data = yf.download(f"{symbol}-USD", period="5y", progress=False)
    else:
        st.error("Invalid asset type selected.")
        return None
    return data

# Calculate Technical Indicators
def calculate_technical_indicators(data):
    if data.empty:
        raise ValueError("The dataset is empty. Please check the ticker symbol or data source.")
    if 'Close' not in data.columns:
        raise ValueError("'Close' column is missing in the dataset. Please verify the data source.")
    if data['Close'].isnull().any():
        raise ValueError("The 'Close' column contains missing values. Please check the dataset integrity.")

    try:
        # Calculate the 20-day SMA
        data['SMA20'] = data['Close'].rolling(window=20).mean()

        # Calculate the rolling standard deviation
        rolling_std = data['Close'].rolling(window=20).std()
        if rolling_std.isnull().any():
            raise ValueError("Rolling standard deviation contains null values.")

        # Add Bollinger Bands
        data['Upper Band'] = data['SMA20'] + (2 * rolling_std)
        data['Lower Band'] = data['SMA20'] - (2 * rolling_std)

        # Calculate RSI
        data['RSI'] = calculate_rsi(data)
        
        # Drop rows with missing values resulting from rolling calculations
        data.dropna(inplace=True)

    except Exception as e:
        raise ValueError(f"An error occurred while calculating technical indicators: {e}")

    return data

# Calculate RSI
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Fetch News Sentiment
def fetch_news_sentiment(symbol):
    url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey={NEWS_API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        articles = response.json()["articles"]
        positive = sum(1 for article in articles if any(word in article["description"].lower() for word in ["gain", "rise", "increase"]))
        neutral = sum(1 for article in articles if any(word in article["description"].lower() for word in ["hold", "steady", "neutral"]))
        negative = len(articles) - positive - neutral
        return positive, neutral, negative
    except Exception as e:
        st.error(f"Error fetching news sentiment: {e}")
        return None, None, None

# Main App Logic
def main():
    try:
        data = fetch_data(ticker_symbol, asset_type)
        if data is None or data.empty:
            st.error(f"No data found for {ticker_symbol}. Please check the ticker symbol or data source.")
            return

        if "Bollinger Bands" in features or "RSI" in features:
            data = calculate_technical_indicators(data)

        if "Bollinger Bands" in features:
            st.subheader("Bollinger Bands")
            plt.figure(figsize=(10, 6))
            plt.plot(data['Close'], label='Close', color='blue')
            plt.plot(data['Upper Band'], label='Upper Band', color='red')
            plt.plot(data['Lower Band'], label='Lower Band', color='green')
            plt.legend()
            st.pyplot()

        if "RSI" in features:
            st.subheader("Relative Strength Index (RSI)")
            plt.figure(figsize=(10, 6))
            plt.plot(data['RSI'], label='RSI', color='orange')
            plt.axhline(70, linestyle='--', color='red', label='Overbought')
            plt.axhline(30, linestyle='--', color='green', label='Oversold')
            plt.legend()
            st.pyplot()

        if "Sentiment Analysis" in features:
            st.subheader("News Sentiment Analysis")
            positive, neutral, negative = fetch_news_sentiment(ticker_symbol)
            if positive is not None:
                st.write(f"Positive Articles: {positive}")
                st.write(f"Neutral Articles: {neutral}")
                st.write(f"Negative Articles: {negative}")

        if "Seasonality Insights" in features:
            st.subheader("Seasonality Insights")
            monthly_avg = data['Close'].resample('M').mean()
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=monthly_avg, label='Monthly Average')
            plt.legend()
            st.pyplot()

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
