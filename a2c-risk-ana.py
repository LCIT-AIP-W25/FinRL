from flask import Flask, jsonify, request
import yfinance as yf
import numpy as np
import pandas as pd
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from sklearn.preprocessing import MinMaxScaler
import quantstats as qs

app = Flask(__name__)

# Helper function to get stock data
def get_stock_data(ticker, period='5y'):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    df = df[['Close', 'Volume']]
    df['SMA'] = df['Close'].rolling(window=20).mean()
    df['RSI'] = qs.rsi(df['Close'])
    df.dropna(inplace=True)
    return df

# Custom Trading Environment (simplified)
class StockTradingEnv:
    def __init__(self, df):
        self.df = df
        self.current_step = 0
        self.n_steps = len(df) - 1
        self.state_size = 5  # Close, Volume, SMA, RSI, Position
        
    def reset(self):
        self.current_step = 0
        return self._next_state()
        
    def _next_state(self):
        return self.df.iloc[self.current_step].values
    
    def step(self, action):
        self.current_step += 1
        reward = self.calculate_reward(action)
        done = self.current_step >= self.n_steps
        return self._next_state(), reward, done, {}
    
    def calculate_reward(self, action):
        # Simplified reward calculation
        current_price = self.df.iloc[self.current_step]['Close']
        next_price = self.df.iloc[self.current_step+1]['Close']
        return (next_price - current_price) * action

# Risk Analysis
def perform_risk_analysis(df):
    returns = df['Close'].pct_change().dropna()
    risk_report = {
        'volatility': returns.std(),
        'sharpe_ratio': qs.stats.sharpe(returns),
        'max_drawdown': qs.stats.max_drawdown(returns),
        'value_at_risk': qs.stats.var(returns)
    }
    return risk_report

@app.route('/predict', methods=['GET'])
def predict():
    ticker = request.args.get('ticker', 'AAPL')
    
    try:
        # Get and preprocess data
        data = get_stock_data(ticker)
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
        
        # Prepare environment and model
        env = make_vec_env(lambda: StockTradingEnv(scaled_data), n_envs=1)
        model = A2C('MlpPolicy', env, verbose=0)
        model.learn(total_timesteps=1000)
        
        # Generate predictions
        obs = env.reset()
        predictions = []
        for _ in range(len(data)-1):
            action, _ = model.predict(obs)
            predictions.append(float(action[0]))
            obs, _, done, _ = env.step(action)
            if done:
                break
        
        # Risk analysis
        risk_report = perform_risk_analysis(data)
        
        # Prepare response
        response = {
            'ticker': ticker,
            'last_price': data['Close'].iloc[-1],
            'predictions': predictions[-10:],  # Last 10 predictions
            'risk_analysis': risk_report,
            'status': 'success'
        }
        
    except Exception as e:
        response = {
            'status': 'error',
            'message': str(e)
        }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)