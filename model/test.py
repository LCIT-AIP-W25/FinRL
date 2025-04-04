import os
import numpy as np
import pandas as pd
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

# Paths
MODEL_PATH = "./models/a2c_risk_model.zip"
DATA_PATH = "./results/processed_data_test.csv"

# Load processed data
df = pd.read_csv(DATA_PATH)

# Environment configuration
stock_dimension = 1
state_space = 7
hmax = 100
initial_amount = 1_000_000
buy_cost_pct = [0.001] * stock_dimension
sell_cost_pct = [0.001] * stock_dimension
num_stock_shares = [0] * stock_dimension
action_space = stock_dimension
tech_indicator_list = ['macd', 'rsi', 'cci', 'dx']

# Make the environment
def create_env():
    return StockTradingEnv(
        df=df,
        stock_dim=stock_dimension,
        hmax=hmax,
        initial_amount=initial_amount,
        num_stock_shares=num_stock_shares,
        buy_cost_pct=buy_cost_pct,
        sell_cost_pct=sell_cost_pct,
        reward_scaling=1e-4,
        state_space=state_space,
        action_space=action_space,
        tech_indicator_list=tech_indicator_list,
        print_verbosity=1
    )

# Wrap in DummyVecEnv
env = DummyVecEnv([create_env])

# Load model
if os.path.exists(MODEL_PATH):
    model = A2C.load(MODEL_PATH)
    print("✅ Model loaded successfully!")
else:
    raise FileNotFoundError(f"❌ Model file not found at {MODEL_PATH}")

# Run one test episode
obs = env.reset()  # returns only obs when using DummyVecEnv
done = False
total_reward = 0

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)
    total_reward += reward[0]
    if done[0]:
        break

print(f"🏁 Total reward from test episode: {total_reward:.2f}")
