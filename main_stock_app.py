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
st.title("📈 Advanced Stock, Crypto, and Investment Insights")

# Sidebar inputs
st.sidebar.header("Select Parameters")
asset_type = st.sidebar.selectbox("Asset Type", ["Stock", "Crypto"])
symbol = st.sidebar.text_input("Ticker Symbol", "AAPL")
features = st.sidebar.multiselect("Select Features", ["Bollinger Bands", "RSI", "Sentiment Analysis"])

# Fetch historical data
def fetch_data(symbol):
    try:
        data = yf.download(symbol, start="2020-01-01", end=pd.Timestamp.now().strftime("%Y-%m-%d"))
        data = data.reset_index()
        return data
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return None

# Calculate technical indicators
def calculate_technical_indicators(data):
    try:
        if 'Close' not in data.columns:
            raise ValueError("'Close' column is missing in the data.")
        
        # Bollinger Bands
        data['SMA20'] = data['Close'].rolling(window=20).mean()
        rolling_std = data['Close'].rolling(window=20).std()
        data['Upper Band'] = data['SMA20'] + (2 * rolling_std)
        data['Lower Band'] = data['SMA20'] - (2 * rolling_std)
        
        # RSI
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        return data
    except Exception as e:
        st.error(f"Error calculating technical indicators: {e}")
        return None

# Fetch sentiment analysis
def fetch_news_sentiment(symbol):
    try:
        url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey={NEWS_API_KEY}"
        response = requests.get(url)
        articles = response.json().get('articles', [])
        positive, neutral, negative = 0, 0, 0
        for article in articles:
            description = article.get('description', '').lower()
            if any(word in description for word in ['gain', 'rise', 'increase']):
                positive += 1
            elif any(word in description for word in ['fall', 'drop', 'decrease']):
                negative += 1
            else:
                neutral += 1
        return positive, neutral, negative
    except Exception as e:
        st.error(f"Error fetching news sentiment: {e}")
        return None, None, None

# Main app logic
def main():
    try:
        data = fetch_data(symbol)
        if data is None or data.empty:
            st.error(f"No data found for {symbol}. Please check the ticker symbol or data source.")
            return

        if "Bollinger Bands" in features or "RSI" in features:
            data = calculate_technical_indicators(data)

        st.subheader(f"Historical Data for {symbol}")
        st.write(data.tail())

        if "Bollinger Bands" in features:
            st.subheader("Bollinger Bands")
            plt.figure(figsize=(10, 6))
            plt.plot(data['Date'], data['Close'], label='Close', color='blue')
            plt.plot(data['Date'], data['Upper Band'], label='Upper Band', color='red')
            plt.plot(data['Date'], data['Lower Band'], label='Lower Band', color='green')
            plt.fill_between(data['Date'], data['Lower Band'], data['Upper Band'], color='gray', alpha=0.2)
            plt.legend()
            st.pyplot(plt)

        if "RSI" in features:
            st.subheader("Relative Strength Index (RSI)")
            plt.figure(figsize=(10, 6))
            plt.plot(data['Date'], data['RSI'], label='RSI', color='orange')
            plt.axhline(70, linestyle='--', color='red', label='Overbought (70)')
            plt.axhline(30, linestyle='--', color='green', label='Oversold (30)')
            plt.legend()
            st.pyplot(plt)

        if "Sentiment Analysis" in features:
            positive, neutral, negative = fetch_news_sentiment(symbol)
            st.subheader("News Sentiment Analysis")
            st.write(f"Positive Articles: {positive}")
            st.write(f"Neutral Articles: {neutral}")
            st.write(f"Negative Articles: {negative}")

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
