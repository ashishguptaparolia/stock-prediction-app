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

# Function to clean column headers
def clean_columns(data):
    if isinstance(data.columns, pd.MultiIndex):  # Check for multi-level columns
        data.columns = ['_'.join(col).strip() for col in data.columns.values]  # Flatten
    return data

# Function to calculate technical indicators
def calculate_indicators(data):
    # Bollinger Bands
    data['SMA20'] = data['Close'].rolling(window=20).mean()
    data['Upper Band'] = data['SMA20'] + 2 * data['Close'].rolling(window=20).std()
    data['Lower Band'] = data['SMA20'] - 2 * data['Close'].rolling(window=20).std()

    # RSI Calculation
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(wi
