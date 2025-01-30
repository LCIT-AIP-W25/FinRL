from flask import Flask, request, jsonify
import yfinance as yf
from werkzeug.exceptions import HTTPException

app = Flask(__name__)

# Custom error handler for the entire application
@app.errorhandler(Exception)
def handle_exception(e):
    if isinstance(e, HTTPException):
        # Standard HTTP exceptions
        return jsonify({"error": e.name, "message": e.description}), e.code
    else:
        # Non-HTTP exceptions (unexpected errors)
        return jsonify({"error": "Internal Server Error", "message": str(e)}), 500

@app.route('/api/stock', methods=['GET'])
def get_stock_data():
    """
    API endpoint to get stock data.
    Takes a stock symbol as input and returns the latest stock information.
    Example: GET /api/stock?symbol=AAPL
    """
    # Get the stock symbol from the request arguments
    stock_symbol = request.args.get('symbol')
    
    if not stock_symbol:
        # Handle the case where the stock symbol is missing
        return jsonify({"error": "Bad Request", "message": "Missing 'symbol' parameter"}), 400
    
    try:
        # Fetch stock data using yfinance
        stock = yf.Ticker(stock_symbol)
        stock_info = stock.info  # Get all stock information
        
        if 'regularMarketPrice' not in stock_info:
            # Handle the case where the stock symbol is invalid
            return jsonify({"error": "Not Found", "message": f"Stock symbol '{stock_symbol}' not found"}), 404
        
        # Prepare the data to return
        stock_data = {
            "symbol": stock_symbol,
            "name": stock_info.get("shortName", "N/A"),
            "current_price": stock_info.get("regularMarketPrice", "N/A"),
            "previous_close": stock_info.get("regularMarketPreviousClose", "N/A"),
            "open": stock_info.get("regularMarketOpen", "N/A"),
            "day_high": stock_info.get("regularMarketDayHigh", "N/A"),
            "day_low": stock_info.get("regularMarketDayLow", "N/A"),
            "volume": stock_info.get("regularMarketVolume", "N/A"),
            "market_cap": stock_info.get("marketCap", "N/A"),
            "currency": stock_info.get("currency", "N/A"),
        }
        
        return jsonify({"success": True, "data": stock_data}), 200
    
    except Exception as e:
        # Handle unexpected errors
        return jsonify({"error": "Internal Server Error", "message": str(e)}), 500

if __name__ == '__main__':
    # Run the Flask app on localhost with debugging enabled
    app.run(debug=True)
