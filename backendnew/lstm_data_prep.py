import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import sys
import psycopg2
from psycopg2.extras import execute_values
import pickle

# Add the parent directory to the path to find db_config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from db_config import get_connection

# Directory for pre-trained models
models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'models'))
print(f"models_dir is: {models_dir}")

print(f"Current working directory: {os.getcwd()}")

def get_tickers_from_db(conn):
    """Fetch all distinct tickers from the stock_data table."""
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT ticker FROM stock_data ORDER BY ticker")
        tickers = [row[0] for row in cur.fetchall()]
    return tickers

def get_data_for_ticker(conn, ticker):
    """Fetch historical data for a specific ticker from the database."""
    print(f"Querying for ticker: '{ticker}' (upper: '{ticker.upper()}')")
    sql = "SELECT date, close, sma_10, sma_50, rsi, volume FROM stock_Data WHERE UPPER(ticker) = %s ORDER BY date"
    df = pd.read_sql(sql, conn, params=(ticker.upper(),), index_col='date', parse_dates=['date'])
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    df.dropna(inplace=True)
    return df

def predict_future_prices(model, data, scaler, time_step, days=10):
    """Predict future prices for a given stock using a trained model."""
    print(f"data shape: {data.shape}")
    last_data = data[-time_step:].values
    print(f"last_data shape: {last_data.shape}")
    temp_input = [row.tolist() for row in last_data]
    lst_output = []
    i = 0
    n_features = data.shape[1]
    while i < days:
        # Always keep temp_input as a list of lists of length time_step
        x_input = np.array(temp_input[-time_step:])  # shape: (time_step, n_features)
        print(f"x_input shape: {x_input.shape}")
        x_input_scaled = scaler.transform(x_input)
        print(f"x_input_scaled shape: {x_input_scaled.shape}")
        x_input_scaled = x_input_scaled.reshape(1, time_step, n_features)
        print(f"x_input_scaled (reshaped) shape: {x_input_scaled.shape}")
        yhat = model.predict(x_input_scaled, verbose=0)
        print(f"yhat shape: {yhat.shape}")
        # Use only the first value of the model's output, repeated for all features
        yhat_value = yhat.flatten()[0]
        yhat_flat = [yhat_value] * n_features
        print(f"yhat_flat (patched): {yhat_flat}")
        temp_input.append(yhat_flat)
        print(f"temp_input length: {len(temp_input)}")
        lst_output.append(yhat_flat)
        i += 1
        # Keep temp_input at most time_step long
        if len(temp_input) > time_step:
            temp_input = temp_input[1:]
    # Inverse transform to get actual prices
    predictions_placeholder = np.zeros((len(lst_output), n_features))
    predictions_placeholder[:, 0] = np.array(lst_output)[:, 0]
    real_prices = scaler.inverse_transform(predictions_placeholder)[:, 0]
    return real_prices

def categorize_risk_levels(predictions_df):
    """Categorizes risk levels based on the volatility of predicted prices."""
    predictions_df['volatility'] = predictions_df['prediction'].pct_change().rolling(window=2).std()
    predictions_df['risk_level'] = 'low'
    predictions_df.loc[predictions_df['volatility'] > 0.02, 'risk_level'] = 'medium'
    predictions_df.loc[predictions_df['volatility'] > 0.05, 'risk_level'] = 'high'
    return predictions_df.drop(columns=['volatility'])

def create_predictions_table(conn):
    """Create the lstm_predictions table if it does not exist."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS lstm_predictions (
                id SERIAL PRIMARY KEY,
                date DATE NOT NULL,
                ticker TEXT NOT NULL,
                prediction DOUBLE PRECISION,
                risk_level TEXT,
                predicted_at TIMESTAMP NOT NULL,
                UNIQUE (date, ticker)
            );
        """)
    conn.commit()

def save_predictions_to_db(conn, predictions_df, ticker):
    """Save the future predictions DataFrame to the database."""
    predictions_df['ticker'] = ticker
    predictions_df['predicted_at'] = datetime.utcnow()
    
    tuples = [tuple(x) for x in predictions_df[['date', 'ticker', 'prediction', 'risk_level', 'predicted_at']].to_numpy()]
    
    with conn.cursor() as cur:
        sql = """
            INSERT INTO lstm_predictions (date, ticker, prediction, risk_level, predicted_at)
            VALUES %s
            ON CONFLICT (date, ticker) DO UPDATE 
            SET prediction = EXCLUDED.prediction,
                risk_level = EXCLUDED.risk_level,
                predicted_at = EXCLUDED.predicted_at;
        """
        execute_values(cur, sql, tuples)
    conn.commit()
    print(f"Inserted predictions for {ticker} into the database.")
    print(f"✅ Saved {len(tuples)} predictions for {ticker} to the database.")

def main():
    if not os.path.exists(models_dir):
        print(f"ERROR: Models directory does not exist: {models_dir}")
        return
    conn = get_connection()
    if not conn:
        return

    try:
        create_predictions_table(conn)
        
        # Build tickers list from model files that have both model and scaler
        model_files = [f for f in os.listdir(models_dir) if f.endswith('_lstm_model.h5')]
        tickers = []
        for model_file in model_files:
            ticker = model_file.replace('_lstm_model.h5', '')
            scaler_path = os.path.join(models_dir, f'{ticker}_scaler.pkl')
            if os.path.exists(scaler_path):
                tickers.append(ticker)
        print(f"Found {len(tickers)} tickers with both model and scaler in the models directory. Generating predictions...")

        for ticker in tickers:
            model_path = os.path.join(models_dir, f'{ticker}_lstm_model.h5')
            scaler_path = os.path.join(models_dir, f'{ticker}_scaler.pkl')
            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                print(f"Skipping {ticker}: model or scaler not found.")
                continue

            try:
                print(f"--- Processing {ticker} ---")
                data = get_data_for_ticker(conn, ticker)
                print(f"Data for {ticker}: {len(data)} rows")
                print(data.head())
                if data.empty:
                    print(f"Skipping {ticker}: no data found in database.")
                    continue
                
                lstm_model = load_model(model_path)
                # Load the scaler from the pkl file
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                data_for_scaling = data[['close', 'sma_10', 'sma_50', 'rsi', 'volume']]
                # Do NOT fit the scaler again, just use it to transform
                
                time_step = 100
                future_prices = predict_future_prices(lstm_model, data_for_scaling, scaler, time_step, days=10)
                
                last_date = data.index[-1]
                future_dates = pd.to_datetime([last_date + timedelta(days=i) for i in range(1, 11)])
                
                future_data = pd.DataFrame({'date': future_dates, 'prediction': future_prices})
                future_data = categorize_risk_levels(future_data)

                print(f"Inserting {len(future_data)} predictions for {ticker}")
                save_predictions_to_db(conn, future_data, ticker)

            except Exception as e:
                print(f"❌ Error processing {ticker}: {e}")

    finally:
        if conn:
            conn.close()
            print("Database connection closed.")

if __name__ == "__main__":
    main()
