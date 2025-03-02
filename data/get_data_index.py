from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import requests
from datetime import datetime, timedelta
import time

API_KEY = "CWHZICCXVV5NA5M8"

def get_sp500_tickers():
    """
    Fetches the list of S&P 500 tickers from Wikipedia.
    
    Returns:
        list: A list of S&P 500 ticker symbols.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response = requests.get(url)
    tables = pd.read_html(response.text)
    sp500_table = tables[0]  # The first table contains the tickers
    return sp500_table["Symbol"].tolist()

def fetch_stock_data(stock_name, api_key=API_KEY):
    """
    Fetches historical stock data for the given stock name using Alpha Vantage.
    
    Args:
        stock_name (str): The stock ticker symbol (e.g., "AAPL" for Apple).
        api_key (str): Your Alpha Vantage API key.
    
    Returns:
        pd.DataFrame: A DataFrame containing the stock data.
    """
    try:
        # Initialize Alpha Vantage TimeSeries
        ts = TimeSeries(key=api_key, output_format="pandas")
        
        # Fetch daily historical data
        data, meta_data = ts.get_daily(symbol=stock_name, outputsize="full")
        
        if data.empty:
            raise ValueError(f"No data found for stock: {stock_name}")
        
        # Filter data for the last 5 years
        five_years_ago = datetime.now() - timedelta(days=5 * 365)
        data = data[data.index >= five_years_ago.strftime("%Y-%m-%d")]
        
        return data
    except Exception as e:
        print(f"Error fetching data for {stock_name}: {e}")
        return None

def save_to_csv(data, stock_name):
    """
    Saves the stock data to a CSV file.
    
    Args:
        data (pd.DataFrame): The stock data to save.
        stock_name (str): The stock ticker symbol (e.g., "AAPL" for Apple).
    """
    try:
        # Generate a filename using the stock name
        filename = f"{stock_name}_historical_data.csv"
        data.to_csv(filename)
        print(f"Data saved to {filename}")
    except Exception as e:
        print(f"Error saving data to CSV: {e}")

def main():
    # Get the list of S&P 500 tickers
    sp500_tickers = get_sp500_tickers()
    print(f"Found {len(sp500_tickers)} S&P 500 tickers.")
    
    # Fetch and save data for each ticker
    for i, ticker in enumerate(sp500_tickers):
        print(f"Fetching data for {ticker} ({i + 1}/{len(sp500_tickers)})...")
        
        # Fetch stock data for the last 5 years
        stock_data = fetch_stock_data(ticker)
        
        if stock_data is not None:
            # Save the data to a CSV file
            save_to_csv(stock_data, ticker)
        
        # Add a delay to avoid hitting Alpha Vantage's rate limit (5 requests per minute)
        time.sleep(12)  # 12 seconds delay between requests

if __name__ == "__main__":
    main()