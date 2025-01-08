# Update the `calculate_technical_indicators` function
def calculate_technical_indicators(data):
    try:
        if data.empty:
            raise ValueError("The dataset is empty. Unable to calculate indicators.")
        
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
        
        data.fillna(0, inplace=True)  # Fill NaN values
        return data
    except Exception as e:
        st.error(f"Error calculating technical indicators: {e}")
        return None

# Update `fetch_news_sentiment` function
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

# Check for empty DataFrame in the main function
def main():
    try:
        data = fetch_data(symbol)
        if data is None or data.empty:
            st.error(f"No valid data found for symbol {symbol}. Please verify the ticker or source.")
            return

        if "Bollinger Bands" in features or "RSI" in features:
            data = calculate_technical_indicators(data)
            if data is None:
                return

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
            plt.clf()

        if "RSI" in features:
            st.subheader("Relative Strength Index (RSI)")
            plt.figure(figsize=(10, 6))
            plt.plot(data['Date'], data['RSI'], label='RSI', color='orange')
            plt.axhline(70, linestyle='--', color='red', label='Overbought (70)')
            plt.axhline(30, linestyle='--', color='green', label='Oversold (30)')
            plt.legend()
            st.pyplot(plt)
            plt.clf()

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
