import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime

# App title
st.title("Advanced Live Stock, Crypto, and Indian Stock Insights")

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

# Function to calculate technical indicators
def calculate_technical_indicators(data):
    data['SMA20'] = data['Close'].rolling(window=20).mean()
    data['Upper Band'] = data['SMA20'] + 2 * data['Close'].rolling(window=20).std()
    data['Lower Band'] = data['SMA20'] - 2 * data['Close'].rolling(window=20).std()
    data['RSI'] = calculate_rsi(data)
    data['Volatility'] = data['Close'].rolling(window=10).std()
    return data

# Function to calculate RSI
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Function to generate short-term insights
def generate_suggestions(data):
    latest_rsi = data['RSI'].iloc[-1]
    if latest_rsi > 70:
        suggestion = "Sell - The stock is overbought."
    elif latest_rsi < 30:
        suggestion = "Buy - The stock is oversold."
    else:
        suggestion = "Hold - RSI is neutral."
    
    # Check for breakout above/below Bollinger Bands
    if data['Close'].iloc[-1] > data['Upper Band'].iloc[-1]:
        suggestion += " Possible uptrend breakout."
    elif data['Close'].iloc[-1] < data['Lower Band'].iloc[-1]:
        suggestion += " Possible downtrend breakout."
    
    return suggestion

# Main app logic
def main():
    data = yf.download(symbol, period="6mo", interval="1d")
    if data.empty:
        st.error(f"No data found for {symbol}.")
        return
    
    # Display historical data
    st.subheader(f"Historical Data for {symbol}")
    st.write(data.tail())

    # Calculate technical indicators
    data = calculate_technical_indicators(data)

    # Plot Bollinger Bands
    st.subheader("Bollinger Bands and Trends")
    plt.figure(figsize=(12, 6))
    plt.plot(data['Close'], label='Closing Price', color='blue')
    plt.plot(data['Upper Band'], label='Upper Band', color='red')
    plt.plot(data['Lower Band'], label='Lower Band', color='green')
    plt.fill_between(data.index, data['Upper Band'], data['Lower Band'], color='gray', alpha=0.2)
    plt.title(f"Bollinger Bands for {symbol}")
    plt.legend()
    st.pyplot(plt)

    # Short-term suggestions
    suggestion = generate_suggestions(data)
    st.subheader("Short-Term Suggestion")
    st.write(suggestion)

    # Enable live updates
    if live_update:
        st.subheader("Live Data Updates")
        while True:
            live_data = fetch_live_data(symbol)
            if live_data:
                st.metric(label="Live Price", value=f"${live_data['price']:.2f}")
                st.metric(label="Volume", value=f"{live_data['volume']:,}")
                st.metric(label="Change (%)", value=f"{live_data['change']:.2f}%")
            else:
                st.error("Unable to fetch live data.")
            
            time.sleep(refresh_interval)
            st.experimental_rerun()

# Run the app
if st.button("Run Analysis"):
    main()