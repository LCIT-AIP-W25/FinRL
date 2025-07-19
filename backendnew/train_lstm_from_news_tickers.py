import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from db_config import get_connection

# Ensure 'models' folder exists to save the trained models
models_dir = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(models_dir, exist_ok=True)

# Prepare the data for LSTM
def create_dataset(data, time_step=1, prediction_days=10):
    X, Y = [], []
    for i in range(len(data) - time_step - prediction_days):
        a = data[i:(i + time_step), :]
        X.append(a)
        Y.append(data[i + time_step:i + time_step + prediction_days, 0])
    return np.array(X), np.array(Y)

def get_tickers_from_finance_news():
    """Get list of unique tickers from the finance_news table"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT ticker FROM finance_news ORDER BY ticker")
        tickers = [row[0] for row in cursor.fetchall()]
        cursor.close()
        conn.close()
        return tickers
    except Exception as e:
        print(f"Error fetching tickers from finance_news: {e}")
        return []

def get_stock_data_from_db(ticker):
    """Fetch stock data for a specific ticker from the database"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        query = """
        SELECT date, close, sma_10, sma_50, rsi, volume
        FROM stock_data 
        WHERE ticker = %s 
        ORDER BY date
        """
        cursor.execute(query, (ticker,))
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        if not rows:
            return None
        # Convert to DataFrame
        df = pd.DataFrame(rows, columns=['Date', 'Close', 'SMA_10', 'SMA_50', 'RSI', 'Volume'])
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        return df
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

# Training parameters
time_step = 100
prediction_days = 10

# Get tickers from finance_news table
tickers = get_tickers_from_finance_news()
print(f"Found {len(tickers)} tickers in finance_news table")

for ticker in tickers:
    print(f"\n{'='*50}")
    print(f"Training LSTM model for {ticker}")
    print(f"{'='*50}")
    # Load data from database
    data = get_stock_data_from_db(ticker)
    if data is None or len(data) < time_step + prediction_days + 10:
        print(f"❌ Insufficient data for {ticker}. Skipping...")
        continue
    # Display the first few rows of the raw data
    print(f"Raw data for {ticker}:")
    print(data.head())
    print(f"Data shape: {data.shape}")
    # Check for NaN values in the raw data
    if np.any(np.isnan(data)):
        print(f"NaN values found in raw data for {ticker}")
        data.fillna(method='ffill', inplace=True)  # Forward fill NaN values
        data.fillna(method='bfill', inplace=True)  # Backward fill NaN values
    # Drop any remaining NaN values
    data.dropna(inplace=True)
    if len(data) < time_step + prediction_days + 10:
        print(f"❌ Insufficient data after cleaning for {ticker}. Skipping...")
        continue
    # Normalize the data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data[['Close', 'SMA_10', 'SMA_50', 'RSI', 'Volume']])
    # Display the first few rows of the scaled data
    print(f"Scaled data for {ticker}:")
    print(pd.DataFrame(data_scaled, columns=['Close', 'SMA_10', 'SMA_50', 'RSI', 'Volume']).head())
    # Check for NaN values in the scaled data
    if np.any(np.isnan(data_scaled)):
        print(f"❌ NaN values found in scaled data for {ticker}. Skipping...")
        continue
    X, Y = create_dataset(data_scaled, time_step, prediction_days)
    if len(X) == 0:
        print(f"❌ No valid sequences created for {ticker}. Skipping...")
        continue
    # Display the first few rows of the dataset
    print(f"Dataset for {ticker}:")
    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")
    # Check for NaN values in the dataset
    if np.any(np.isnan(X)) or np.any(np.isnan(Y)):
        print(f"❌ NaN values found in dataset for {ticker}. Skipping...")
        continue
    # Reshape input to be [samples, time steps, features]
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2])
    # Create the LSTM model
    model = Sequential()
    model.add(Masking(mask_value=0.0, input_shape=(time_step, X.shape[2])))
    model.add(LSTM(100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(50))
    model.add(Dense(prediction_days))  # Output layer with neurons for 10 days prediction
    # Compile the model with gradient clipping
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, clipvalue=1.0)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    # Train the model with KeyboardInterrupt handling
    try:
        print(f"Training model for {ticker}...")
        history = model.fit(X, Y, epochs=50, batch_size=32, validation_split=0.2, verbose=1)
        print(f"✅ Training completed for {ticker}")
    except KeyboardInterrupt:
        print("Training interrupted by user.")
        continue
    except Exception as e:
        print(f"❌ Error during training for {ticker}: {e}")
        continue
    # Save the model
    model_path = os.path.join(models_dir, f"{ticker}_lstm_model.h5")
    model.save(model_path)
    print(f"✅ LSTM model for {ticker} saved to {model_path}")
    # Save the scaler for later use
    scaler_path = os.path.join(models_dir, f"{ticker}_scaler.pkl")
    import pickle
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✅ Scaler for {ticker} saved to {scaler_path}")
    # Evaluate the model
    try:
        predictions = model.predict(X)
        mae = mean_absolute_error(Y, predictions)
        rmse = np.sqrt(mean_squared_error(Y, predictions))
        print(f"Model performance for {ticker}: MAE = {mae:.6f}, RMSE = {rmse:.6f}")
    except Exception as e:
        print(f"❌ Error during evaluation for {ticker}: {e}")

print(f"\n{'='*50}")
print("LSTM Training Complete!")
print(f"Models saved to: {models_dir}")
print(f"{'='*50}") 