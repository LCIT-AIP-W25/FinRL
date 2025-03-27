from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the data and model
data = pd.read_csv('data/AAPL.csv', parse_dates=['Date'], index_col='Date')
data.fillna(method='ffill', inplace=True)
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data[['Close', 'SMA_10', 'SMA_50', 'RSI']])
time_step = 100
lstm_model = load_model('models/AAPL_lstm_model.h5')

def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), :]
        X.append(a)
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

def predict_future_prices(model, data, time_step, days=30):
    predictions = []
    last_data = data[-time_step:]
    for _ in range(days):
        last_data_scaled = scaler.transform(last_data)
        X_input = last_data_scaled[:, 0].reshape(1, time_step, 1)
        pred_price = model.predict(X_input)
        predictions.append(pred_price[0][0])
        new_row = np.append(last_data[1:], [[pred_price[0][0], np.nan, np.nan, np.nan]], axis=0)
        last_data = new_row
    placeholder = np.zeros((len(predictions), data.shape[1]))
    placeholder[:, 0] = predictions
    real_prices = scaler.inverse_transform(placeholder)[:, 0]
    return real_prices

@app.route('/predict', methods=['GET'])
def get_predictions():
    try:
        future_prices = predict_future_prices(lstm_model, data[['Close', 'SMA_10', 'SMA_50', 'RSI']].values, time_step, days=30)
        next_day_prediction = future_prices[0]  # Get the first predicted value
        last_10_days = data[['Close', 'SMA_10', 'SMA_50', 'RSI']].tail(20).reset_index().to_dict(orient='records')
        
        # Print the last 10 days of stock data to the console
        print("Last 10 Days Stock Data:")
        for day in last_10_days:
            print(day)
        
        response = {
            'next_day_prediction': next_day_prediction,
            'last_10_days': last_10_days
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)