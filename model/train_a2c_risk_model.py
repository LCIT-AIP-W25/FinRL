import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3 import A2C
import os

# Create directories for saving models and results
if not os.path.exists("./models"):
    os.makedirs("./models")
if not os.path.exists("./results"):
    os.makedirs("./results")

def load_csv_data(csv_file_path, ticker):
    """Load stock data from CSV file and format it for FinRL"""
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file_path)
        
        # Standardize column names (assuming CSV has columns for date, open, high, low, close, volume)
        # Map columns to lowercase
        df.columns = [col.lower().strip() for col in df.columns]
        
        # Common variations of column names
        date_cols = ['date', 'time', 'datetime']
        open_cols = ['open', '1. open', 'open price']
        high_cols = ['high', '2. high', 'high price']
        low_cols = ['low', '3. low', 'low price']
        close_cols = ['close', '4. close', 'close price', 'adj close', 'adjusted close']
        volume_cols = ['volume', '5. volume', 'trading volume']
        
        # Find the appropriate column names
        date_col = next((col for col in df.columns if col in date_cols), None)
        open_col = next((col for col in df.columns if col in open_cols), None)
        high_col = next((col for col in df.columns if col in high_cols), None)
        low_col = next((col for col in df.columns if col in low_cols), None)
        close_col = next((col for col in df.columns if col in close_cols), None)
        volume_col = next((col for col in df.columns if col in volume_cols), None)
        
        # Check if all required columns are found
        if None in [date_col, open_col, high_col, low_col, close_col, volume_col]:
            print("Warning: Some columns could not be identified automatically.")
            print(f"Available columns: {df.columns.tolist()}")
            # Use positional assignment if column names aren't recognized
            if len(df.columns) >= 6:
                date_col, open_col, high_col, low_col, close_col, volume_col = df.columns[:6]
        
        # Create a new DataFrame with standardized column names
        result_df = pd.DataFrame({
            'date': pd.to_datetime(df[date_col]),
            'open': pd.to_numeric(df[open_col], errors='coerce'),
            'high': pd.to_numeric(df[high_col], errors='coerce'),
            'low': pd.to_numeric(df[low_col], errors='coerce'),
            'close': pd.to_numeric(df[close_col], errors='coerce'),
            'volume': pd.to_numeric(df[volume_col], errors='coerce'),
            'tic': ticker
        })
        
        # Drop any rows with NaN values
        result_df = result_df.dropna()
        
        print(f"Successfully loaded {len(result_df)} rows for {ticker}")
        return result_df
    
    except Exception as e:
        print(f"Error loading CSV file {csv_file_path}: {e}")
        return None

def train_a2c_model(data_files):
    """Train A2C model using CSV data files"""
    print("Processing CSV data files...")
    
    all_data = []
    for file_path, ticker in data_files:
        df = load_csv_data(file_path, ticker)
        if df is not None:
            all_data.append(df)
    
    if not all_data:
        print("Error: No valid data found in the provided files.")
        return
    
    # Combine all data
    df = pd.concat(all_data, ignore_index=True)
    
    print("Processing features...")
    # Process features
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=['macd', 'rsi', 'cci', 'dx'],
        use_vix=False,  # Set to False because we're using local data without VIX
        use_turbulence=False,  # Set to False because we're using local data without market-wide info
        user_defined_feature=False
    )
    
    processed_df = fe.preprocess_data(df)
    
    # Save processed data
    processed_df.to_csv('./results/processed_data.csv', index=False)
    print(f"Processed data saved to ./results/processed_data.csv")
    
    # Calculate state space and action space dimensions
    stock_dimension = len(processed_df.tic.unique())
    state_space = 1 + 2*stock_dimension + len(fe.tech_indicator_list)*stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")
    
    # Initialize num_stock_shares - this is required by your version of FinRL
    # Initialize with zeros (no stocks owned at start)
    num_stock_shares = [0] * stock_dimension
    
    # Initialize the environment
    try:
        print("Initializing StockTradingEnv...")
        e_train_gym = StockTradingEnv(
            df=processed_df,
            stock_dim=stock_dimension,
            hmax=100,  # maximum number of shares to trade
            initial_amount=1000000,  # initial cash
            num_stock_shares=num_stock_shares,  # initial stock shares
            buy_cost_pct=[0.001] * stock_dimension,
            sell_cost_pct=[0.001] * stock_dimension,

            reward_scaling=1e-4,  # reward scaling
            state_space=state_space,  # state space dimension
            action_space=stock_dimension,  # action space dimension
            tech_indicator_list=fe.tech_indicator_list,  # technical indicator list
            print_verbosity=5  # verbosity
        )
    except Exception as e:
        print(f"Error initializing environment: {e}")
        print("Please check the exact requirements of your FinRL version")
        return None
    
    # Setup agent
    print("Setting up DRL agent...")
    agent = DRLAgent(env=e_train_gym)
    
    # A2C model parameters - REMOVED 'verbose' and 'tensorboard_log' to avoid conflicts
    a2c_params = {
        "n_steps": 5,
        "ent_coef": 0.01,
        "learning_rate": 0.0005
    }
    
    print("Training A2C model...")
    # Train A2C model
    model_a2c = agent.get_model("a2c", model_kwargs=a2c_params)
    trained_a2c = agent.train_model(model=model_a2c, 
                                   tb_log_name='a2c',
                                   total_timesteps=50000)  
    
    # Save the trained model
    trained_a2c.save("./models/a2c_risk_model")
    print("Model saved to ./models/a2c_risk_model")
    
    # Run a simple backtest
    print("Running backtest...")
    df_daily_return, df_actions = DRLAgent.DRL_prediction(
        model=trained_a2c,
        environment=e_train_gym
    )
    
    # Save the backtest results
    df_daily_return.to_csv("./results/a2c_daily_returns.csv")
    df_actions.to_csv("./results/a2c_actions.csv")
    
    # Plot the performance
    plt.figure(figsize=(12, 5))
    plt.plot(df_daily_return.date, df_daily_return.daily_return.cumsum(), label='A2C Return')
    plt.title('Cumulative Return of A2C')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.savefig('./results/a2c_cumulative_return.png')
    
    print("Training and evaluation completed!")
    print("Check the results in the 'results' folder")
    
    return trained_a2c

def load_and_check_sample_csv(csv_file_path):
    """Load and display info about a CSV file to verify its format"""
    try:
        df = pd.read_csv(csv_file_path)
        print("\nSample CSV Data Information:")
        print(f"File: {csv_file_path}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Number of rows: {len(df)}")
        print("First 5 rows:")
        print(df.head())
        return True
    except Exception as e:
        print(f"Error examining CSV file: {e}")
        return False

if __name__ == "__main__":
    # Install required packages if they're not already installed
    try:
        import finrl
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError:
        print("Installing required packages...")
        import subprocess
        subprocess.check_call(["pip", "install", "finrl", "stable-baselines3", "matplotlib"])
        print("Packages installed successfully!")
    
    # Define your CSV data files
    # Format: (file_path, ticker_symbol)
    data_files = [
        ("AMZN_historical_data.csv", "AMZN"),  
        # ("AAPL_historical_data.csv", "AAPL"),
        # ("MSFT_historical_data.csv", "MSFT"),
    ]
    
    # Check the first file to verify format
    if data_files:
        load_and_check_sample_csv(data_files[0][0])
    
    # Ask user to confirm before proceeding
    confirmation = input("\nDo you want to proceed with training? (y/n): ")
    if confirmation.lower() == 'y':
        # Train and save the model
        trained_model = train_a2c_model(data_files)
        print("Process completed successfully!")
    else:
        print("Training cancelled.")