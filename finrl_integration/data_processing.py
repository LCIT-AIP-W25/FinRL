import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from finrl.preprocessing import data_processors as dp

def preprocess_data(stock_data):
    """
    Preprocess the stock data for the model.
    This includes handling missing data, scaling features, and other preprocessing steps.
    """
    # Drop rows with missing values
    stock_data = stock_data.dropna()

    # Use FinRL's built-in data processor for technical indicators
    stock_data = dp.add_ta_features(stock_data)  # FinRL feature extraction
    
    # Normalize the features (using MinMaxScaler as an example)
    scaler = MinMaxScaler()
    columns_to_scale = ['open', 'high', 'low', 'close', 'volume', 'macd', 'rsi', 'sma50', 'sma200']
    stock_data[columns_to_scale] = scaler.fit_transform(stock_data[columns_to_scale])

    # Return the preprocessed data
    return stock_data
