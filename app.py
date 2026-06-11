import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model

st.title("Tesla Stock Price Prediction")

st.write("This app predicts Tesla stock closing price using Deep Learning models.")

uploaded_file = st.file_uploader("Upload Tesla CSV File", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(df.head())

    st.subheader("Dataset Information")
    st.write("Rows:", df.shape[0])
    st.write("Columns:", df.shape[1])

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    df = df.ffill()

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    st.subheader("Closing Price Trend")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df['Date'], df['Close'])
    ax.set_title("Tesla Closing Price Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price")
    st.pyplot(fig)

    st.subheader("Model Information")

    st.write("""
    Models used in notebook:
    - SimpleRNN
    - LSTM

    Forecasting done for:
    - 1 Day
    - 5 Days
    - 10 Days

    Target column:
    - Close Price
    """)

    model = load_model("tesla_lstm_model.keras")
    scaler = joblib.load("scaler.pkl")

    data = df[['Close']]
    scaled_data = scaler.transform(data)

    last_60_days = scaled_data[-60:]

    st.subheader("Choose Forecast Days")

    forecast_days = st.selectbox(
        "Select prediction period",
        [1, 5, 10]
    )

    future_predictions = []

    input_data = last_60_days.copy()

    for i in range(forecast_days):
        X_input = input_data[-60:].reshape(1, 60, 1)

        pred = model.predict(X_input)

        future_predictions.append(pred[0][0])

        input_data = np.append(input_data, pred)
        input_data = input_data.reshape(-1, 1)

    future_predictions = np.array(future_predictions).reshape(-1, 1)

    future_predictions = scaler.inverse_transform(future_predictions)

    st.subheader(f"Predicted Closing Price for Next {forecast_days} Day(s)")

    prediction_df = pd.DataFrame({
        "Day": [f"Day {i+1}" for i in range(forecast_days)],
        "Predicted Close Price": future_predictions.flatten()
    })

    st.write(prediction_df)

    st.line_chart(prediction_df.set_index("Day"))

    st.subheader("Model Comparison Summary")

    comparison = pd.DataFrame({
        "Feature": [
            "Model 1",
            "Model 2",
            "Forecasts",
            "Evaluation Metrics",
            "Best Model Used in App"
        ],
        "Details": [
            "SimpleRNN",
            "LSTM",
            "1-Day, 5-Day, 10-Day",
            "MSE, MAE, RMSE",
            "LSTM"
        ]
    })

    st.write(comparison)

else:
    st.info("Please upload the TSLA.csv file.")