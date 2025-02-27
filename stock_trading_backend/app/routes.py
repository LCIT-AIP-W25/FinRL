from flask import Blueprint, jsonify
import yfinance as yf

stock_bp = Blueprint("stocks", __name__)  # No URL prefix!

STOCK_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

@stock_bp.route("/stocks", methods=["GET"])  # Make sure URL is correct
def get_stocks():
    stock_data = {}
    
    for symbol in STOCK_SYMBOLS:
        try:
            stock = yf.Ticker(symbol)
            stock_info = stock.history(period="1d")
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
