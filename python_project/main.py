import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense
import matplotlib.pyplot as plt


# Function to load stock data
def load_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data['Change'] = data['Close'].pct_change()  # Percentage change
    data['Label'] = np.where(data['Change'] > 0, 1, 0)  # 1 if price up, else 0
    return data.dropna()


# Preprocess data
def preprocess_data(data):
    scaler = MinMaxScaler()
    features = data[['Open', 'High', 'Low', 'Close', 'Volume']].values
    labels = data['Label'].values

    scaled_features = scaler.fit_transform(features)
    X, y = [], []
    time_step = 30  # Lookback period

    for i in range(time_step, len(scaled_features)):
        X.append(scaled_features[i - time_step:i])
        y.append(labels[i])

    return np.array(X), np.array(y), scaler


# Build LSTM-GRU model
def build_lstm_gru_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        GRU(32, return_sequences=False),
        Dense(1, activation='sigmoid')  # For classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Train and evaluate
def train_and_evaluate(data):
    X, y, scaler = preprocess_data(data)
    split = int(0.8 * len(X))

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Train LSTM-GRU model
    model = build_lstm_gru_model((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    lstm_predictions = (model.predict(X_test) > 0.5).astype(int)

    # Train Naive Bayes on LSTM-GRU output
    nb = GaussianNB()
    nb.fit(lstm_predictions, y_test)

    nb_predictions = nb.predict(lstm_predictions)
    acc = accuracy_score(y_test, nb_predictions)

    return model, nb, scaler, acc, X_test, y_test, lstm_predictions


# Streamlit App
def main():
    st.title("LSTM-GRU + Naive Bayes Stock Prediction")
    st.sidebar.header("User Input Parameters")

    # Sidebar Inputs
    ticker = st.sidebar.text_input("Stock Ticker", "KSE")
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-01-01"))

    if st.sidebar.button("Run Prediction"):
        # Load and process data
        data = load_data(ticker, start_date, end_date)

        if data.empty:
            st.error("No data available for the selected ticker or date range.")
        else:
            st.write("### Stock Data")
            st.dataframe(data)

            # Train and evaluate the model
            model, nb_model, scaler, acc, X_test, y_test, lstm_preds = train_and_evaluate(data)

            # Display model accuracy
            st.write(f"### Naive Bayes Accuracy: {acc:.2f}")

            # Plot predictions
            st.write("### Predictions vs Actual")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(y_test, label="Actual Trend")
            ax.plot(lstm_preds, label="Predicted Trend", linestyle="dashed")
            ax.set_xlabel("Time")
            ax.set_ylabel("Trend (1 = Up, 0 = Down)")
            ax.legend()
            st.pyplot(fig)

            # User Input Prediction
            st.write("### Predict the Next Day Trend")
            latest_data = X_test[-1].reshape(1, X_test.shape[1], X_test.shape[2])
            next_day_pred = (model.predict(latest_data) > 0.5).astype(int)
            trend = "Up" if next_day_pred[0][0] == 1 else "Down"
            st.write(f"**Predicted Trend for {ticker}:** {trend}")


if __name__ == "__main__":
    main()
