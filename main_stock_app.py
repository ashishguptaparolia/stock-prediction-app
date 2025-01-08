import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import requests

# App title
st.title("Enhanced Stock Analysis, Forecasting, and Portfolio Management")

# Sidebar inputs
st.sidebar.header("Select Parameters")
stock_symbol = st.sidebar.text_input("Stock Ticker (e.g., AAPL, TSLA, MSFT)", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-12-31"))

# News API Key
NEWS_API_KEY = "364b5eac98da4a31ac519a8d67581444"

# Function to extract key columns
def extract_columns(data, stock_symbol):
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = ['_'.join(col).strip() for col in data.columns.values]
    # Dynamically rename columns
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
    data.rename(columns=columns_mapping, inplace=True)
    return data

# Function to calculate technical indicators
def calculate_indicators(data):
    data['SMA20'] = data['Close'].rolling(window=20).mean()
    data['Upper Band'] = data['SMA20'] + 2 * data['Close'].rolling(window=20).std()
    data['Lower Band'] = data['SMA20'] - 2 * data['Close'].rolling(window=20).std()
    data['MACD'] = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()
    data['Signal Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['OBV'] = (np.sign(data['Close'].diff()) * data['Volume']).cumsum()
    return data

# LSTM-based price prediction
def predict_prices(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    X_train = []
    y_train = []
    for i in range(60, len(scaled_data)):
        X_train.append(scaled_data[i-60:i])
        y_train.append(scaled_data[i])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=32, epochs=5)
    
    last_60_days = scaled_data[-60:]
    forecast = []
    for _ in range(30):
        prediction = model.predict(last_60_days.reshape(1, -1, 1))
        forecast.append(prediction[0, 0])
        last_60_days = np.append(last_60_days[1:], prediction[0, 0]).reshape(-1, 1)
    return scaler.inverse_transform(np.array(forecast).reshape(-1, 1))

# Fetch news sentiment
def fetch_news_sentiment(symbol):
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

# Main app
if st.button("Run Analysis"):
    try:
        data = yf.download(stock_symbol, start=start_date, end=end_date)
        if data.empty:
            st.warning("No data found for the selected ticker and date range.")
            st.stop()
        
        data = extract_columns(data, stock_symbol)
        data = calculate_indicators(data)
        
        st.subheader(f"{stock_symbol} Historical Data")
        st.write(data.tail())
        
        st.subheader(f"{stock_symbol} Technical Indicators")
        plt.figure(figsize=(12, 6))
        plt.plot(data['Close'], label='Closing Price', color='blue')
        plt.plot(data['Upper Band'], label='Upper Bollinger Band', color='red')
        plt.plot(data['Lower Band'], label='Lower Bollinger Band', color='green')
        plt.fill_between(data.index, data['Upper Band'], data['Lower Band'], color='gray', alpha=0.2)
        plt.title(f"{stock_symbol} with Bollinger Bands")
        plt.legend()
        st.pyplot(plt)
        
        st.subheader("30-Day Price Forecast")
        forecast = predict_prices(data)
        forecast_dates = pd.date_range(data.index[-1] + pd.Timedelta(days=1), periods=30, freq='B')
        forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecasted Price': forecast.flatten()})
        st.write(forecast_df)
        
    except Exception as e:
        st.error(f"An error occurred: {e}")
