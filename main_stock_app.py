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
