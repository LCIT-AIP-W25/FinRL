import os
import pandas as pd
import yfinance as yf
from datetime import datetime

# Ensure 'data' folder exists
os.makedirs("data", exist_ok=True)

def fetch_sp500_symbols():
    """Fetch the current list of S&P 500 symbols from Wikipedia."""
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    tables = pd.read_html(url)
    df = tables[0]
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

def get_stock_data(symbols, start="2015-01-01", end=None):
    if end is None:
        end = datetime.today().strftime('%Y-%m-%d')
    
    for symbol in symbols:
        stock = yf.download(symbol, start=start, end=end)
        stock.reset_index(inplace=True)
        stock = stock[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        stock = calculate_indicators(stock)

        file_path = os.path.join("data", f"{symbol}.csv")
        stock.to_csv(file_path, index=False)
        print(f"✅ Stock data for {symbol} with indicators saved at {file_path}.")
        print(stock.head(20))

if __name__ == "__main__":
    symbols = fetch_sp500_symbols()
    get_stock_data(symbols)
