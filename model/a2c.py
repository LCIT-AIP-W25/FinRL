import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from alpha_vantage.timeseries import TimeSeries
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.config import INDICATORS
from stable_baselines3 import A2C


# --------------------------
# STEP 1: FETCH STOCK DATA FROM ALPHA VANTAGE
# --------------------------

def fetch_data_from_alpha_vantage(tickers, start_date, end_date, api_key):
    """Fetch stock data from Alpha Vantage."""
    ts = TimeSeries(key=api_key, output_format='pandas')
    df_list = []

    for ticker in tickers:
        try:
            data, meta_data = ts.get_daily(symbol=ticker, outputsize='full')
            data = data.loc[start_date:end_date]
            data['tic'] = ticker
            data.reset_index(inplace=True)
            df_list.append(data)
        except Exception as e:
            print(f"❌ Error fetching data for {ticker}: {e}")
    
    # Raise an error if no valid data is retrieved
    if not df_list:
        raise ValueError("❌ No valid data fetched for the provided tickers.")
    
    # Combine data from multiple tickers
    data_df = pd.concat(df_list, axis=0)
    data_df.columns = ["date", "open", "high", "low", "close", "adj_close", "volume", "tic"]
    data_df.sort_values(by=["date", "tic"], inplace=True)
    data_df.reset_index(drop=True, inplace=True)
    return data_df


# --------------------------
# STEP 2: PREPROCESS DATA
# --------------------------

def preprocess_data(data):
    """Clean and prepare the data for the model."""
    data["date"] = pd.to_datetime(data["date"])
    data["day"] = data["date"].dt.day
    data["month"] = data["date"].dt.month
    data["year"] = data["date"].dt.year
    data = data.dropna()
    return data


# --------------------------
# STEP 3: SET ENVIRONMENT PARAMETERS
# --------------------------

def create_env(data, start_date, end_date, initial_amount, technical_indicators):
    """Create a stock trading environment using FinRL."""
    train_data = data[(data.date >= start_date) & (data.date < end_date)]
    
    env_kwargs = {
        "hmax": 100,                         # Max number of shares to trade
        "initial_amount": initial_amount,    # Initial portfolio value
        "buy_cost_pct": 0.001,               # Transaction cost for buying
        "sell_cost_pct": 0.001,              # Transaction cost for selling
        "state_space": 1 + 2 * len(train_data.tic.unique()) + len(technical_indicators),
        "stock_dim": len(train_data.tic.unique()),
        "tech_indicator_list": technical_indicators,
        "action_space": len(train_data.tic.unique()),
        "reward_scaling": 1e-4,              # Reward scaling factor
    }
    
    env = StockTradingEnv(df=train_data, **env_kwargs)
    return env


# --------------------------
# STEP 4: TRAIN A2C MODEL
# --------------------------

def train_a2c_model(env, timesteps=100000, model_path="./trained_models/a2c_model.zip"):
    """Train and save the A2C model."""
    model = A2C("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=timesteps)
    model.save(model_path)
    print(f"✅ Model trained and saved to {model_path}")


# --------------------------
# STEP 5: EVALUATE MODEL PERFORMANCE
# --------------------------

def evaluate_model(env, model_path):
    """Load and evaluate the trained model."""
    model = A2C.load(model_path)
    obs = env.reset()
    done = False
    rewards = []
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        rewards.append(reward)
    
    df_results = pd.DataFrame(env.df)
    return df_results


# --------------------------
# STEP 6: PERFORM RISK ANALYSIS
# --------------------------

def perform_risk_analysis(df_results):
    """Perform and display risk analysis results."""
    df_results["daily_return"] = df_results["account_value"].pct_change()
    
    # Plot cumulative returns
    df_results["cumulative_return"] = (1 + df_results["daily_return"]).cumprod()
    plt.figure(figsize=(12, 6))
    plt.plot(df_results["date"], df_results["cumulative_return"], label="A2C Portfolio Performance")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.title("📈 A2C Portfolio Cumulative Returns")
    plt.grid()
    plt.show()
    
    # Calculate Sharpe ratio
    sharpe_ratio = (df_results["daily_return"].mean() / df_results["daily_return"].std()) * np.sqrt(252)
    print(f"📊 Sharpe Ratio: {sharpe_ratio:.2f}")
    
    # Calculate maximum drawdown
    rolling_max = df_results["account_value"].cummax()
    drawdown = df_results["account_value"] / rolling_max - 1.0
    max_drawdown = drawdown.min()
    print(f"📉 Max Drawdown: {max_drawdown:.2%}")


# --------------------------
# MAIN FUNCTION
# --------------------------

if __name__ == "__main__":
    # User input for tickers and API Key
    tickers = input("📡 Enter stock ticker(s) (comma-separated): ").upper().replace(" ", "").split(",")
    start_date = "2022-01-01"
    end_date = "2024-01-01"
    initial_amount = 100000
    api_key = "952YHVJG4CEAZBQH"
    technical_indicators = INDICATORS
    
    # Fetch and process data
    print("📡 Fetching data...")
    try:
        data_df = fetch_data_from_alpha_vantage(tickers, start_date, end_date, api_key)
    except ValueError as e:
        print(f"❌ {e}")
        exit()

    data_df = preprocess_data(data_df)
    
    # Create training environment
    print("🛠️ Setting up environment...")
    env = create_env(data_df, "2022-01-01", "2023-01-01", initial_amount, technical_indicators)
    
    # Train and save A2C model
    print("🚀 Training A2C model...")
    model_path = "./trained_models/a2c_model.zip"
    train_a2c_model(env, model_path=model_path)
    
    # Evaluate model and perform risk analysis
    print("📈 Evaluating model performance...")
    test_env = create_env(data_df, "2023-01-01", "2024-01-01", initial_amount, technical_indicators)
    results_df = evaluate_model(test_env, model_path)
    
    # Perform risk analysis
    print("🔍 Performing risk analysis...")
    perform_risk_analysis(results_df)
