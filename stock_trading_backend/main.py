from flask import Flask, jsonify
from flask_cors import CORS  # Import CORS
import yfinance as yf
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)
CORS(app)  # Enable CORS for the entire app

# List of stock symbols
STOCK_SYMBOLS = [
    'MMM', 'AXP', 'AAPL', 'BA', 'CAT', 'CVX', 'CSCO', 'KO', 'DIS', 'GS',
    'HD', 'IBM', 'INTC', 'JNJ', 'JPM', 'MCD', 'MRK', 'MSFT', 'NKE',
    'PFE', 'PG', 'TRV', 'UNH', 'VZ', 'V', 'WBA', 'WMT', 'XOM'
]

@app.route('/stocks', methods=["GET"])
def get_stocks():
    stock_data = {}
    for symbol in STOCK_SYMBOLS:
        try:
            stock = yf.Ticker(symbol)
            stock_info = stock.history(period="1d")  # Get 1 day data
            if not stock_info.empty:
                stock_data[symbol] = {
                    "price": round(stock_info["Close"].iloc[-1], 2),
                    "volume": int(stock_info["Volume"].iloc[-1]),
                }
            else:
                stock_data[symbol] = {"error": "No data available"}
        except Exception as e:
            stock_data[symbol] = {"error": str(e)}

    return jsonify(stock_data)

@app.route('/api/trending')
def trending():
    trending_stocks = yf.Ticker("AAPL MSFT GOOG").history(period="1d")
    return jsonify(trending_stocks.to_dict())

@app.route('/app/gainers')
def gainers():
    gainers_url = "https://finance.yahoo.com/gainers"
    response = requests.get(gainers_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    gainers_data = []
    for row in soup.find_all('tr', attrs={'class': 'simpTblRow'}):
        cols = row.find_all('td')
        if len(cols) > 0:
            gainers_data.append({
                'symbol': cols[0].text.strip(),
                'name': cols[1].text.strip(),
                'price': cols[2].text.strip(),
                'change': cols[3].text.strip(),
                'percent_change': cols[4].text.strip(),
                'volume': cols[5].text.strip(),
                'avg_volume': cols[6].text.strip(),
                'market_cap': cols[7].text.strip(),
                'pe_ratio': cols[8].text.strip()
            })
    return jsonify(gainers_data)

@app.route('/api/losers')
def losers():
    losers_url = "https://finance.yahoo.com/losers"
    response = requests.get(losers_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    losers_data = []
    for row in soup.find_all('tr', attrs={'class': 'simpTblRow'}):
        cols = row.find_all('td')
        if len(cols) > 0:
            losers_data.append({
                'symbol': cols[0].text.strip(),
                'name': cols[1].text.strip(),
                'price': cols[2].text.strip(),
                'change': cols[3].text.strip(),
                'percent_change': cols[4].text.strip(),
                'volume': cols[5].text.strip(),
                'avg_volume': cols[6].text.strip(),
                'market_cap': cols[7].text.strip(),
                'pe_ratio': cols[8].text.strip()
            })
    return jsonify(losers_data)

if __name__ == "__main__":
    app.run(debug=True)