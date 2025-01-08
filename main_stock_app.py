import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
import requests
from datetime import datetime
import time

# Your News API Key
NEWS_API_KEY = "364b5eac98da4a31ac519a8d67581444"

# App title
st.set_page_config(page_title="Advanced Stock & Crypto Insights", layout="wide")
st.title("📈 Advanced Stock, Crypto, and Investment Insights")

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
def fetch_news_sentiment(symbol):
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
            return None
    except:
        return None

def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_technical_indicators(data):
    data['SMA20'] = data['Close'].rolling(window=20).mean()
    rolling_std = data['Close'].rolling(window=20).std()
    data['Upper Band'] = data['SMA20'] + (2 * rolling_std)
    data['Lower Band'] = data['SMA20'] - (2 * rolling_std)
    data['RSI'] = calculate_rsi(data)
    data['Volatility'] = data['Close'].rolling(window=10).std()
    data.dropna(inplace=True)
    return data

def short_term_prediction(data):
    data['Lag1'] = data['Close'].shift(1)
    data.dropna(inplace=True)
    X = np.array(data[['Lag1']])
    y = np.array(data['Close'])
    model = LinearRegression()
    model.fit(X, y)
    predicted_price = model.predict(X[-1].reshape(1, -1))[0]
    return predicted_price

def generate_peer_comparison(peers):
    peer_data = {}
    for peer in peers:
        data = yf.download(peer, period="6mo", interval="1d")
        if not data.empty:
            peer_data[peer] = data['Close'].iloc[-1]
    return peer_data

def generate_seasonality(data):
    data['Month'] = data.index.month
    seasonality = data.groupby('Month')['Close'].mean()
    return seasonality

def generate_volatility_heatmap(data):
    data['Daily Change'] = data['Close'].pct_change()
    data['Day'] = data.index.day
    data['Month'] = data.index.month
    heatmap_data = data.pivot("Day", "Month", "Daily Change")
    return heatmap_data

def generate_investment_calculator(data, investment_amount):
    current_price = data['Close'].iloc[-1]
    projected_price = short_term_prediction(data)
    returns = (projected_price - current_price) / current_price * 100
    potential_value = investment_amount + (investment_amount * (returns / 100))
    return projected_price, potential_value, returns

# Main App Logic
def main():
    data = yf.download(symbol, period="1y", interval="1d")
    if data.empty:
        st.error(f"No data found for {symbol}. Please check the ticker symbol and try again.")
        return

    # Calculate technical indicators
    data = calculate_technical_indicators(data)

    # Bollinger Bands
    if "Bollinger Bands" in analysis_options:
        st.subheader("Bollinger Bands")
        plt.figure(figsize=(12, 6))
        plt.plot(data['Close'], label="Closing Price", color="blue")
        plt.plot(data['SMA20'], label="20-Day SMA", color="orange")
        plt.plot(data['Upper Band'], label="Upper Band", color="green")
        plt.plot(data['Lower Band'], label="Lower Band", color="red")
        plt.fill_between(data.index, data['Lower Band'], data['Upper Band'], color="gray", alpha=0.2)
        plt.title(f"Bollinger Bands for {symbol}")
        plt.legend()
        st.pyplot(plt)

    # RSI
    if "RSI" in analysis_options:
        st.subheader("RSI (Relative Strength Index)")
        plt.figure(figsize=(12, 4))
        plt.plot(data['RSI'], label="RSI", color="purple")
        plt.axhline(70, linestyle="--", color="red", label="Overbought (70)")
        plt.axhline(30, linestyle="--", color="green", label="Oversold (30)")
        plt.title(f"RSI for {symbol}")
        plt.legend()
        st.pyplot(plt)

    # Sentiment Analysis
    if "Sentiment Analysis" in analysis_options:
        st.subheader("News Sentiment Analysis")
        sentiment = fetch_news_sentiment(symbol)
        if sentiment:
            st.write(f"Positive Articles: {sentiment[0]}")
            st.write(f"Neutral Articles: {sentiment[1]}")
            st.write(f"Negative Articles: {sentiment[2]}")
        else:
            st.write("Unable to fetch news sentiment.")

    # Peer Comparison
    if "Peer Comparison" in analysis_options:
        st.subheader("Peer Comparison")
        peers = st.sidebar.text_input("Enter peer tickers (comma-separated)", "MSFT,GOOGL").split(",")
        peer_data = generate_peer_comparison(peers)
        st.write("Latest Prices of Peers:")
        st.write(peer_data)

    # Seasonality Insights
    if "Seasonality Insights" in analysis_options:
        st.subheader("Seasonality Insights")
        seasonality = generate_seasonality(data)
        st.bar_chart(seasonality)

    # Volatility Heatmap
    if "Volatility Heatmap" in analysis_options:
        st.subheader("Volatility Heatmap")
        heatmap_data = generate_volatility_heatmap(data)
        sns.heatmap(heatmap_data, cmap="coolwarm", annot=False)
        st.pyplot()

    # Investment Calculator
    if "Investment Calculator" in analysis_options:
        st.subheader("Investment Calculator")
        investment_amount = st.number_input("Enter Investment Amount ($):", min_value=100, value=1000)
        projected_price, potential_value, returns = generate_investment_calculator(data, investment_amount)
        st.write(f"Projected Price: ${projected_price:.2f}")
        st.write(f"Potential Portfolio Value: ${potential_value:.2f}")
        st.write(f"Expected Return: {returns:.2f}%")

# Run the app
if st.button("Run Analysis"):
    main()
