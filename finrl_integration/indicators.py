
import pandas as pd
import talib

def add_technical_indicators(stock_data):
    """
    Add technical indicators to the stock data.
    This includes common indicators like RSI, MACD, SMA, etc.
    """
    # Relative Strength Index (RSI)
    stock_data['RSI'] = talib.RSI(stock_data['Close'], timeperiod=14)

    # Moving Average Convergence Divergence (MACD)
    macd, macdsignal, macdhist = talib.MACD(stock_data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    stock_data['MACD'] = macd
    stock_data['MACD_signal'] = macdsignal
    stock_data['MACD_hist'] = macdhist

    # Simple Moving Average (SMA)
    stock_data['SMA_50'] = talib.SMA(stock_data['Close'], timeperiod=50)
    stock_data['SMA_200'] = talib.SMA(stock_data['Close'], timeperiod=200)
    
    # Exponential Moving Average (EMA)
    stock_data['EMA_12'] = talib.EMA(stock_data['Close'], timeperiod=12)
    stock_data['EMA_26'] = talib.EMA(stock_data['Close'], timeperiod=26)

    # Bollinger Bands
    upperband, middleband, lowerband = talib.BBANDS(stock_data['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    stock_data['Bollinger_Upper'] = upperband
    stock_data['Bollinger_Lower'] = lowerband

    return stock_data
