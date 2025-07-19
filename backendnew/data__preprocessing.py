import os
import pandas as pd
import yfinance as yf
from datetime import datetime
import sys
from psycopg2.extras import execute_values

# Add the parent directory to the path to find db_config
# This allows the script to find the db_config.py file in the root SCRAPING directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from db_config import get_connection

# Ensure 'data' folder exists
os.makedirs("data", exist_ok=True)

def fetch_sp500_symbols():
    """Fetch the current list of S&P 500 symbols from Wikipedia."""
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    tables = pd.read_html(url)
    df = tables[0]
    # Replace symbols that have dots with dashes for yfinance compatibility
    df['Symbol'] = df['Symbol'].str.replace('.', '-', regex=False)
    return df['Symbol'].tolist()

def calculate_indicators(df):
    """Calculate SMA & RSI for the dataset."""
    df["SMA_10"] = df["Close"].rolling(window=10).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()

    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    return df

def insert_dataframe_to_db(conn, df, symbol):
    """Insert a DataFrame into the stock_data table using an existing connection."""
    # Add ticker column to the dataframe
    df['ticker'] = symbol
    
    with conn.cursor() as cur:
        # Filter out rows with NaN values that can occur from rolling calculations
        df.dropna(inplace=True)
        # Reorder columns to match the table schema
        df_final = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'SMA_10', 'SMA_50', 'RSI', 'ticker']]
        tuples = [tuple(x) for x in df_final.to_numpy()]
        
        sql = """
            INSERT INTO stock_data (date, open, high, low, close, volume, sma_10, sma_50, rsi, ticker)
            VALUES %s
            ON CONFLICT (date, ticker) DO NOTHING
        """
        if tuples:
            execute_values(cur, sql, tuples)
            print(f"✅ Processed {len(tuples)} rows for {symbol}.")
        else:
            print(f"⚠️ No new data to process for {symbol}.")

def get_stock_data(symbols, start="2015-01-01", end=None):
    if end is None:
        end = datetime.today().strftime('%Y-%m-%d')
    
    conn = None  # Initialize conn to None
    try:
        conn = get_connection()
        print("Database connection established.")
        
        # Create table once if it doesn't exist
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS stock_data (
                    id SERIAL PRIMARY KEY,
                    date DATE NOT NULL,
                    open DOUBLE PRECISION,
                    high DOUBLE PRECISION,
                    low DOUBLE PRECISION,
                    close DOUBLE PRECISION,
                    volume BIGINT,
                    sma_10 DOUBLE PRECISION,
                    sma_50 DOUBLE PRECISION,
                    rsi DOUBLE PRECISION,
                    ticker TEXT NOT NULL,
                    UNIQUE(date, ticker)
                );
            """)
        conn.commit()

        for symbol in symbols:
            try:
                print(f"--- Processing {symbol} ---")
                stock = yf.download(symbol, start=start, end=end, progress=False)
                if stock.empty:
                    print(f"⚠️ No data found for {symbol}, skipping.")
                    continue
                
                stock.reset_index(inplace=True)
                stock = stock[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
                stock = calculate_indicators(stock)

                insert_dataframe_to_db(conn, stock, symbol)
                conn.commit() # Commit after each successful symbol insertion
                
            except Exception as e:
                print(f"❌ Could not process data for {symbol}: {e}")
                if conn:
                    conn.rollback() # Rollback changes for the failed symbol

    finally:
        if conn:
            conn.close()
            print("Database connection closed.")

if __name__ == "__main__":
    symbols = fetch_sp500_symbols()
    print(f"Found {len(symbols)} S&P 500 symbols. Fetching data for all symbols.")
    get_stock_data(symbols)
