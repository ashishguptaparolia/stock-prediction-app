import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests

# Your News API Key
NEWS_API_KEY = "364b5eac98da4a31ac519a8d67581444"

# App title
st.set_page_config(page_title="Advanced Stock, Crypto, and Investment Insights", layout="wide")
st.title("📈 Advanced Stock, Crypto, and Investment Insights")

# Sidebar inputs
st.sidebar.header("Select Parameters")
asset_type = st.sidebar.selectbox("Asset Type", ["Stock", "Crypto"])
symbol = st.sidebar.text_input("Ticker Symbol", "AAPL").upper()
features = st.sidebar.multiselect("Select Features", ["Bollinger Bands", "RSI", "Sentiment Analysis"])

# Fetch historical data
def fetch_data(symbol):
    try:
        # Fetch data using yfinance
        data = yf.download(symbol, start="2020-01-01", end=pd.Timestamp.now().strftime("%Y-%m-%d"))
        
        # Validate the data
        if data is None or data.empty:
            raise ValueError(f"No data found for symbol: {symbol}")
        
        # Reset index to include 'Date' column
        data.reset_index(inplace=True)

        # Ensure necessary columns exist
        if 'Close' not in data.columns:
            raise ValueError("The fetched data does not contain the required 'Close' column.")
        
        return data
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return None

# Calculate technical indicators
def calculate_technical_indicators(data):
    try:
        # Check if the DataFrame is empty
        if data.empty:
            raise ValueError("The dataset is empty. Unable to calculate indicators.")
        
        # Check if 'Close' column exists
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
        loss.replace(0, np.nan, inplace=True)  # Avoid division by zero
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))

        # Fill NaN values
        data.fillna(0, inplace=True)
        return data
    except Exception as e:
        st.error(f"Error calculating technical indicators: {e}")
        return None

# Fetch sentiment analysis
def fetch_news_sentiment(symbol):
    try:
        url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey={NEWS_API_KEY}"
        response = requests.get(url)
        if response.status_code != 200:
            raise ValueError(f"News API returned status code {response.status_code}")
        
        response_data = response.json()
        articles = response_data.get('articles', [])
        
        if not articles:
            raise ValueError("No articles found in the News API response.")
        
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
        # Fetch data
        data = fetch_data(symbol)
        if data is None or data.empty:
            st.error(f"No valid data found for symbol {symbol}. Please verify the ticker or source.")
            return

        # Calculate indicators
        if "Bollinger Bands" in features or "RSI" in features:
            data = calculate_technical_indicators(data)
            if data is None or data.empty:
                st.error("Failed to calculate technical indicators.")
                return

        # Display historical data
        st.subheader(f"Historical Data for {symbol}")
        st.write(data.tail())

        # Bollinger Bands visualization
        if "Bollinger Bands" in features:
            st.subheader("Bollinger Bands")
            plt.figure(figsize=(10, 6))
            plt.plot(data['Date'], data['Close'], label='Close', color='blue')
            plt.plot(data['Date'], data['Upper Band'], label='Upper Band', color='red')
            plt.plot(data['Date'], data['Lower Band'], label='Lower Band', color='green')
            plt.fill_between(data['Date'], data['Lower Band'], data['Upper Band'], color='gray', alpha=0.2)
            plt.legend()
            st.pyplot(plt)
            plt.clf()

        # RSI visualization
        if "RSI" in features:
            st.subheader("Relative Strength Index (RSI)")
            plt.figure(figsize=(10, 6))
            plt.plot(data['Date'], data['RSI'], label='RSI', color='orange')
            plt.axhline(70, linestyle='--', color='red', label='Overbought (70)')
            plt.axhline(30, linestyle='--', color='green', label='Oversold (30)')
            plt.legend()
            st.pyplot(plt)
            plt.clf()

        # Sentiment Analysis
        if "Sentiment Analysis" in features:
            positive, neutral, negative = fetch_news_sentiment(symbol)
            if positive is not None:
                st.subheader("News Sentiment Analysis")
                st.write(f"Positive Articles: {positive}")
                st.write(f"Neutral Articles: {neutral}")
                st.write(f"Negative Articles: {negative}")

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
