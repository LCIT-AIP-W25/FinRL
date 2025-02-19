from finrl import config
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime

def fetch_stock_data(stock_list, start_date, end_date):
    data = {}
    for stock in stock_list:
        df = yf.download(stock, start=start_date, end=end_date)
        data[stock] = df
    return data

def process_data(data):
    processed_dfs = []
    for stock, df in data.items():
        print(f"Processing data for stock: {stock}")
        # Add symbol column
        df = df.copy()
        df['symbol'] = stock
        # Handle date index
        df = df.reset_index().rename(columns={'Date': 'date'})
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        # Select relevant columns
        if 'Adj Close' in df.columns:
            cols = ['symbol', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        else:
            print(f"Using Close instead of Adj Close for {stock}")
            cols = ['symbol', 'Open', 'High', 'Low', 'Close', 'Volume']
        processed_dfs.append(df[cols])
    return pd.concat(processed_dfs)

def create_environment(processed_data):
    num_stocks = len(processed_data['symbol'].unique())
    tech_indicator_list = ['macd', 'rsi_14', 'cci', 'atr']
    env = StockTradingEnv(
        df=processed_data,
        stock_dim=num_stocks,
        hmax=100,
        initial_amount=1000000,
        buy_cost_pct=0.001,
        sell_cost_pct=0.001,
        reward_scaling=1e-4,
        print_verbosity=100,
        num_stock_shares=100,
        state_space=len(tech_indicator_list),
        action_space=num_stocks * 3,
        tech_indicator_list=tech_indicator_list
    )
    return env

def train_and_evaluate_model(env):
    agent = DRLAgent(env=env)
    model = agent.get_model('a2c')
    trained_model = agent.train_model(model, tb_log_name='a2c')
    eval_result = agent.evaluate_trading_agent(trained_model)
    return trained_model, eval_result

def risk_analysis(env):
    portfolio_value = env.get_portfolio_value()
    daily_returns = portfolio_value.pct_change().dropna()
    volatility = daily_returns.std()
    return portfolio_value, volatility

def print_results(portfolio_value, volatility, eval_result):
    print("Portfolio Metrics:")
    print(portfolio_value.tail())
    print("\nVolatility:", volatility)
    print("\nEvaluation Result:", eval_result)

def main():
    stock_list = ['AAPL', 'MSFT', 'GOOG', 'AMZN']
    start_date = '2015-01-01'
    end_date = datetime.today().strftime('%Y-%m-%d')
    
    stock_data = fetch_stock_data(stock_list, start_date, end_date)
    processed_data = process_data(stock_data)
    env = create_environment(processed_data)
    trained_model, eval_result = train_and_evaluate_model(env)
    portfolio_value, volatility = risk_analysis(env)
    print_results(portfolio_value, volatility, eval_result)

if __name__ == '__main__':
    main()