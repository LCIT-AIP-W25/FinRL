import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Ensure 'models' folder exists to save the trained models
os.makedirs("models", exist_ok=True)

# Prepare the data for LSTM
def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), 0]
        X.append(a)
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

# Get the list of stock symbols from the data folder
data_folder = "data"
symbols = [file.split(".")[0] for file in os.listdir(data_folder) if file.endswith(".csv")]

time_step = 100

for symbol in symbols:
    # Load and preprocess the data
    file_path = os.path.join(data_folder, f"{symbol}.csv")
    data = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')

    # Normalize the data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data[['Close', 'SMA_10', 'SMA_50', 'RSI']])

    X, Y = create_dataset(data_scaled, time_step)

    # Reshape input to be [samples, time steps, features]
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Create the LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X, Y, epochs=100, batch_size=32)

    # Save the model
    model.save(os.path.join("models", f"{symbol}_lstm_model.h5"))
    print(f"✅ LSTM model for {symbol} trained and saved.")