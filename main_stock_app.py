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
asset_type = st.sidebar.selectbox("Asset Type", ["Stock", "Cryptocurrency", "Indian Stock"])
symbol = st.sidebar.text_input("Ticker Symbol", "AAPL")
analysis_options = st.sidebar.multiselect(
    "Select Features",
    ["Bollinger Bands", "RSI", "Sentiment Analysis", "Peer Comparison", 
     "Seasonality Insights", "Volatility Heatmap", "Investment Calculator"],
    default=["Bollinger Bands", "RSI", "Sentiment Analysis"]
)

# Helper Functions
def fetch_data(ticker):
    """Fetch historical data for the given ticker."""
    try:
        data = yf.download(ticker, period="6mo", interval="1d")
        if 'Close' not in data.columns:
            raise ValueError("'Close' column is missing in the fetched data.")
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None

def fetch_news_sentiment(symbol):
    """Fetch sentiment analysis from news articles."""
    url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey={NEWS_API_KEY}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            articles = response.json().get('articles', [])
            positive, neutral, negative = 0, 0, 0
            for article in articles[:10]:  # Limit to 10 articles
                description = article.get('description', '') or ''
                if any(word in description.lower() for word in ['gain', 'rise', 'increase', 'bull']):
                    positive += 1
                elif any(word in description.lower() for word in ['fall', 'drop', 'decrease', 'bear']):
                    negative += 1
                else:
                    neutral += 1
            return positive, neutral, negative
        else:
            st.warning("Failed to fetch news sentiment.")
            return None
    except Exception as e:
        st.error(f"Error fetching news sentiment: {e}")
        return None

def calculate_rsi(data, window=14):
    """Calculate Relative Strength Index (RSI)."""
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_technical_indicators(data):
    """Calculate Bollinger Bands and RSI."""
    if data.empty:
        raise ValueError("The dataset is empty. Please check the ticker symbol or data source.")
    if 'Close' not in data.columns:
        raise ValueError("'Close' column is missing in the dataset. Please verify the data source.")
    if data['Close'].isnull().any():
        raise ValueError("The 'Close' column contains missing values. Please check the dataset integrity.")

    try:
        data['SMA20'] = data['Close'].rolling(window=20).mean()
        rolling_std = data['Close'].rolling(window=20).std()
        data['Upper Band'] = data['SMA20'] + (2 * rolling_std)
        data['Lower Band'] = data['SMA20'] - (2 * rolling_std)
        data['RSI'] = calculate_rsi(data)
    except Exception as e:
        raise ValueError(f"An error occurred while calculating technical indicators: {e}")

    data.dropna(inplace=True)
    return data

def generate_volatility_heatmap(data):
    """Generate a heatmap for daily volatility."""
    if data.empty:
        raise ValueError("No data available for generating the volatility heatmap.")

    try:
        data['Daily Change'] = data['Close'].pct_change()
        data['Day'] = data.index.day
        data['Month'] = data.index.month

        heatmap_data = data.pivot_table(values='Daily Change', index='Day', columns='Month', aggfunc='mean')
        heatmap_data.fillna(0, inplace=True)
    except Exception as e:
        raise ValueError(f"An error occurred while generating the heatmap: {e}")

    return heatmap_data

# Main App Logic
def main():
    try:
        data = fetch_data(symbol)
        if data is None or data.empty:
            st.error(f"No data found for {symbol}. Please check the ticker symbol or data source.")
            return

        data = calculate_technical_indicators(data)
        st.write(data.tail())

        if "Bollinger Bands" in analysis_options:
            st.subheader("Bollinger Bands")
            try:
                plt.figure(figsize=(12, 6))
                plt.plot(data['Close'], label="Closing Price", color="blue")
                plt.plot(data['SMA20'], label="20-Day SMA", color="orange")
                plt.plot(data['Upper Band'], label="Upper Band", color="green")
                plt.plot(data['Lower Band'], label="Lower Band", color="red")
                plt.fill_between(data.index, data['Lower Band'], data['Upper Band'], color="gray", alpha=0.2)
                plt.title(f"Bollinger Bands for {symbol}")
                plt.legend()
                st.pyplot(plt)
            except Exception as e:
                st.error(f"Error plotting Bollinger Bands: {e}")

        if "RSI" in analysis_options:
            st.subheader("RSI (Relative Strength Index)")
            try:
                plt.figure(figsize=(12, 4))
                plt.plot(data['RSI'], label="RSI", color="purple")
                plt.axhline(70, linestyle="--", color="red", label="Overbought (70)")
                plt.axhline(30, linestyle="--", color="green", label="Oversold (30)")
                plt.title(f"RSI for {symbol}")
                plt.legend()
                st.pyplot(plt)
            except Exception as e:
                st.error(f"Error plotting RSI: {e}")

        if "Volatility Heatmap" in analysis_options:
            st.subheader("Volatility Heatmap")
            try:
                heatmap_data = generate_volatility_heatmap(data)
                plt.figure(figsize=(10, 6))
                sns.heatmap(heatmap_data, cmap="coolwarm", annot=False)
                plt.title(f"Volatility Heatmap for {symbol}")
                st.pyplot(plt)
            except Exception as e:
                st.error(f"Error generating heatmap: {e}")

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

# Run the app
if st.button("Run Analysis"):
    main()
