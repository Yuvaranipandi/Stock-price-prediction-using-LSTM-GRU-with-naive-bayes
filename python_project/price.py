import yfinance as yf
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
from passlib.hash import pbkdf2_sha256

conn = sqlite3.connect('user_data.db')
c = conn.cursor()

c.execute('''CREATE TABLE IF NOT EXISTS user
             (username TEXT PRIMARY KEY, email TEXT UNIQUE, password TEXT)''')
conn.commit()

def create_user(username, email, password):
    hashed_password = pbkdf2_sha256.hash(password)
    c.execute("INSERT INTO user (username, email, password) VALUES (?, ?, ?)", (username, email, hashed_password))
    conn.commit()

# Function to check if a username exists

# Load the trained model


def show_home():
    st.header("Welcome to the Home Page")
    st.write("This is the main page of our e-commerce app.")

    # Check if the user is logged in
    if 'username' not in st.session_state:
        # Add radio buttons for login and register functionality
        choice = st.radio("Select an option:", ["Login", "Register"])

        if choice == "Login":
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.button("Login"):
                # Verify login credentials
                c.execute("SELECT * FROM user WHERE username=? AND password=?", (username, password))
                user = c.fetchone()
                if user:
                    st.session_state['username'] = username
                    st.session_state['user_id'] = user[0]
                    st.success(f"Welcome back, {username}!")
                    st.success("You have successfully logged in.")
                else:
                    st.error("Invalid username or password. Please try again.")

        elif choice == "Register":
            st.write("Please register to create an account.")
            new_username = st.text_input("New Username")
            new_password = st.text_input("New Password", type="password")
            email = st.text_input("Email Address")

            if st.button("Register"):
                # Check if the username already exists
                c.execute("SELECT * FROM user WHERE username=?", (new_username,))
                existing_user = c.fetchone()
                if existing_user:
                    st.error("Username already exists. Please choose a different one.")
                else:
                    # Insert new user into the database
                    c.execute("INSERT INTO user (username, password, email) VALUES (?, ?, ?)",
                              (new_username, new_password, email))
                    conn.commit()
                    st.success("Registration successful! You can now log in.")

    else:
        st.title('Stock Price Prediction App')

        # Sidebar for user input
        st.sidebar.header('User Input Parameters')

        # Select ticker symbol
        ticker = st.sidebar.text_input("Enter Ticker Symbol", 'GOOG')

        # Select date for prediction
        selected_date = st.sidebar.date_input("Select Date for Prediction", value=pd.to_datetime('today'))

        # Fetch historical stock data up to selected date
        data = yf.download(ticker, start="2023-02-01", end=selected_date)

        # Display fetched data
        st.subheader('Stock Data')
        st.write(data)

        # Feature Engineering: Adding moving averages
        data['MA50'] = data['Close'].rolling(window=50).mean()
        data['MA200'] = data['Close'].rolling(window=200).mean()

        # Drop rows with missing values
        data.dropna(inplace=True)

        # Split data into training and testing sets
        X = data[['Open', 'High', 'Low', 'Close', 'Volume', 'MA50', 'MA200']].values
        y = data['Close'].values

        X_train, X_test = X[:-1], X[-1:]
        y_train, y_test = y[:-1], y[-1:]

        # Train a predictive model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Make predictions
        prediction = model.predict(X_test)

        # Plot historical and predicted prices
        fig, ax = plt.subplots()
        ax.plot(data.index, data['Close'], label='Historical Prices')
        ax.scatter(selected_date, prediction, color='red', label='Predicted Price')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        st.pyplot(fig)

        st.subheader('Predicted Price for Selected Date')
        st.write(prediction[0])

def main():
    show_home()
# Run the app
if __name__ == "__main__":
    main()