import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
from sklearn.linear_model import LinearRegression
import requests

# News API Key
NEWS_API_KEY = "YOUR_NEWS_API_KEY"

# App title
st.title("Advanced Stock, Crypto, and Indian Stock Analysis with Intelligent Insights")

# Sidebar inputs
st.sidebar.header("Select Parameters")
asset_type = st.sidebar.selectbox("Asset Type", ["Stock", "Cryptocurrency", "Indian Stock"])
if asset_type == "Cryptocurrency":
    st.sidebar.write("Example: BTC-USD, ETH-USD")
elif asset_type == "Indian Stock":
    st.sidebar.write("Example: TCS.NS, RELIANCE.NS")
else:
    st.sidebar.write("Example: AAPL, TSLA, MSFT")
    
symbol = st.sidebar.text_input("Ticker Symbol", "AAPL")
live_update = st.sidebar.checkbox("Enable Live Data Updates", value=False)
refresh_interval = st.sidebar.number_input("Refresh Interval (seconds)", min_value=10, max_value=300, value=60) if live_update else None

# Function to fetch live data
def fetch_live_data(symbol):
    ticker = yf.Ticker(symbol)
    live_data = ticker.history(period="1d", interval="1m")
    if live_data.empty:
        return None
    latest_row = live_data.iloc[-1]
    return {
        "price": latest_row["Close"],
        "volume": latest_row["Volume"],
        "change": (latest_row["Close"] - live_data.iloc[0]["Close"]) / live_data.iloc[0]["Close"] * 100
    }

# Function to calculate RSI
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Function to calculate technical indicators
def calculate_technical_indicators(data):
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = ['_'.join(col).strip() for col in data.columns.values]

    if 'Close' not in data.columns:
        close_col = [col for col in data.columns if 'Close' in col]
        if close_col:
            data['Close'] = data[close_col[0]]
        else:
            raise ValueError("No 'Close' column found in the data.")

    if len(data) < 20:
        raise ValueError("Not enough data to calculate technical indicators. At least 20 rows are required.")

    data['SMA20'] = data['Close'].rolling(window=20).mean()
    rolling_std = data['Close'].rolling(window=20).std()
    data['Upper Band'] = data['SMA20'] + (2 * rolling_std)
    data['Lower Band'] = data['SMA20'] - (2 * rolling_std)
    data['RSI'] = calculate_rsi(data)
    data['Volatility'] = data['Close'].rolling(window=10).std()
    data.dropna(inplace=True)
    return data

# Function for advanced prediction
def advanced_prediction(data):
    data['Lag1'] = data['Close'].shift(1)
    data.dropna(inplace=True)
    X = np.array(data[['Lag1']])
    y = np.array(data['Close'])
    model = LinearRegression()
    model.fit(X, y)
    prediction = model.predict(X[-1].reshape(1, -1))
    return prediction[0]

# Function for news sentiment analysis
def fetch_news_sentiment(symbol):
    url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        articles = response.json().get('articles', [])
        positive, neutral, negative = 0, 0, 0
        for article in articles[:10]:  # Limit to 10 articles
            description = article.get('description', '')
            if any(word in description.lower() for word in ['gain', 'rise', 'increase', 'bull']):
                positive += 1
            elif any(word in description.lower() for word in ['fall', 'drop', 'decrease', 'bear']):
                negative += 1
            else:
                neutral += 1
        return positive, neutral, negative
    else:
        st.error("Failed to fetch news sentiment.")
        return 0, 0, 0

# Function to generate insights
def generate_insights(data, symbol):
    positive, neutral, negative = fetch_news_sentiment(symbol)
    insights = f"News Sentiment - Positive: {positive}, Neutral: {neutral}, Negative: {negative}\n"
    latest_rsi = data['RSI'].iloc[-1]
    if latest_rsi > 70:
        insights += "RSI indicates overbought conditions. Possible Sell.\n"
    elif latest_rsi < 30:
        insights += "RSI indicates oversold conditions. Possible Buy.\n"
    else:
        insights += "RSI is neutral. Hold.\n"
    return insights

# Main app logic
def main():
    data = yf.download(symbol, period="6mo", interval="1d")
    if data.empty:
        st.error(f"No data found for {symbol}.")
        return

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = ['_'.join(col).strip() for col in data.columns.values]

    st.subheader(f"Historical Data for {symbol}")
    st.write(data.tail())

    try:
        data = calculate_technical_indicators(data)
    except ValueError as e:
        st.error(e)
        return

    st.subheader("Bollinger Bands and Trends")
    plt.figure(figsize=(12, 6))
    plt.plot(data['Close'], label='Closing Price', color='blue')
    plt.plot(data['Upper Band'], label='Upper Band', color='red')
    plt.plot(data['Lower Band'], label='Lower Band', color='green')
    plt.fill_between(data.index, data['Upper Band'], data['Lower Band'], color='gray', alpha=0.2)
    plt.title(f"Bollinger Bands for {symbol}")
    plt.legend()
    st.pyplot(plt)

    st.subheader("Short-Term Predictions")
    predicted_price = advanced_prediction(data)
    st.write(f"Predicted Price for Next Day: ${predicted_price:.2f}")

    insights = generate_insights(data, symbol)
    st.subheader("Intelligent Insights")
    st.write(insights)

    if live_update:
        st.subheader("Live Data Updates")
        live_data = fetch_live_data(symbol)
        if live_data:
            st.metric(label="Live Price", value=f"${live_data['price']:.2f}")
            st.metric(label="Volume", value=f"{live_data['volume']:,}")
            st.metric(label="Change (%)", value=f"{live_data['change']:.2f}%")
        else:
            st.error("Unable to fetch live data.")
        time.sleep(refresh_interval)

# Run the app
if st.button("Run Analysis"):
    main()
