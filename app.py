import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# Load your trained model and scaler
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('my_model.h5')
    return model

@st.cache_resource
def load_scaler():
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return scaler

model = load_model()
scaler = load_scaler()

st.title("💹Stock Price Prediction with LSTM")

# User inputs
tickers = ['AAPL', 'GOOG', 'MSFT', 'AMZN']
ticker = st.selectbox("Select stock ticker", tickers)

start_date = st.date_input("Start date", pd.to_datetime("2015-01-01"))
end_date = st.date_input("End date", pd.to_datetime("2023-01-01"))

if st.button("Predict"):

    # Fetch historical data
    data = yf.download(ticker, start=start_date, end=end_date)

    if data.empty:
        st.error("No data found. Try a different ticker or date range.")
    else:
        st.subheader(f"{ticker} Closing Prices")
        st.line_chart(data['Close'])

        # Prepare data for LSTM
        close_prices = data['Close'].values.reshape(-1, 1)
        scaled_data = scaler.transform(close_prices)

        # Create sequences for prediction
        def create_sequences(data, seq_length=60):
            x = []
            for i in range(seq_length, len(data)):
                x.append(data[i-seq_length:i, 0])
            return np.array(x)

        x_test = create_sequences(scaled_data)
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

        # Predict
        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)

        # Show prediction vs actual
        valid = data[60:].copy()
        valid['Predictions'] = predictions.flatten()

        st.subheader("Actual vs Predicted Closing Prices")
        fig, ax = plt.subplots()
        ax.plot(valid['Close'], label="Actual")
        ax.plot(valid['Predictions'], label="Predicted")
        ax.legend()
        st.pyplot(fig)
