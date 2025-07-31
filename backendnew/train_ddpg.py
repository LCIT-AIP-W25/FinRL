import os
import pandas as pd
import numpy as np
import torch
from ddpg_agent import DDPGAgent
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from db_config import get_connection

# Directory
<<<<<<< Updated upstream:backendnew/train_ddpg.py
model_save_dir = 'models'
=======
model_save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
>>>>>>> Stashed changes:train_ddpg.py
os.makedirs(model_save_dir, exist_ok=True)

# Utility functions
def categorize_risk_levels(data):
    # Calculate volatility percentiles for better distribution
    volatility_percentiles = data['Volatility'].quantile([0.33, 0.67])
    
    # Ensure we have some data for each risk level
    if len(data) >= 3:
        # Use more balanced thresholds
        low_threshold = volatility_percentiles[0.33] if not pd.isna(volatility_percentiles[0.33]) else data['Volatility'].median() * 0.5
        high_threshold = volatility_percentiles[0.67] if not pd.isna(volatility_percentiles[0.67]) else data['Volatility'].median() * 1.5
        
        data['Risk_Level'] = 'low'
        data.loc[data['Volatility'] > low_threshold, 'Risk_Level'] = 'medium'
        data.loc[data['Volatility'] > high_threshold, 'Risk_Level'] = 'high'
    else:
        # For small datasets, distribute evenly
        data['Risk_Level'] = 'low'
        if len(data) >= 2:
            data.iloc[len(data)//2:, data.columns.get_loc('Risk_Level')] = 'medium'
        if len(data) >= 3:
            data.iloc[2*len(data)//3:, data.columns.get_loc('Risk_Level')] = 'high'
    
    # Print distribution for debugging
    print(f"Risk level distribution: {data['Risk_Level'].value_counts().to_dict()}")
    
    return data

def calculate_reward(state, action, next_state):
    return next_state[0] - state[0]

def get_trading_suggestions(agent, state, risk_level):
    state_with_risk = np.append(state, [risk_level_mapping[risk_level]])
    action = agent.select_action(state_with_risk)
    if action > 0.5:
        return 'Buy'
    elif action < -0.5:
        return 'Sell'
    else:
        return 'Hold'

# Risk level mapping
risk_level_mapping = {'low': 0, 'medium': 1, 'high': 2}

def get_tickers_from_db():
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT ticker FROM lstm_predictions ORDER BY ticker")
        tickers = [row[0] for row in cur.fetchall()]
    conn.close()
    return tickers

def get_predictions_for_ticker(conn, ticker):
    sql = """
        SELECT date, prediction, risk_level
        FROM lstm_predictions
        WHERE ticker = %s
        ORDER BY date
    """
    df = pd.read_sql(sql, conn, params=(ticker,), index_col='date', parse_dates=['date'])
    return df

def main():
    print("DDPG training script started")
    print("Attempting to get tickers from database...")
    tickers = get_tickers_from_db()
    print(f"Tickers found: {tickers}")
    print(f"Found {len(tickers)} tickers in the database. Generating DDPG training...")
    for ticker in tickers:
        print(f"\nProcessing {ticker}...")
        conn = get_connection()
        try:
            data = get_predictions_for_ticker(conn, ticker)
            if data.empty or len(data) < 10:
                print(f"Skipping {ticker}: not enough prediction data.")
                continue
<<<<<<< Updated upstream:backendnew/train_ddpg.py
            # Compute volatility and categorize risk
            data['Volatility'] = data['prediction'].pct_change().rolling(window=30).std()
            data = categorize_risk_levels(data)
            data['Risk_Level'] = data['Risk_Level'].map(risk_level_mapping)
=======
            
            # Use existing risk levels from database instead of recategorizing
            # Map string risk levels to numeric values
            risk_level_mapping_str = {'low': 0, 'medium': 1, 'high': 2}
            data['Risk_Level'] = data['risk_level'].map(risk_level_mapping_str)
            
            # Print distribution for debugging
            print(f"Risk level distribution: {data['risk_level'].value_counts().to_dict()}")
            
>>>>>>> Stashed changes:train_ddpg.py
            # Initialize DDPG agents per risk level
            agents = {
                'low': DDPGAgent(state_dim=2, action_dim=1),
                'medium': DDPGAgent(state_dim=2, action_dim=1),
                'high': DDPGAgent(state_dim=2, action_dim=1)
            }
            # Train agents
            for risk_level in ['low', 'medium', 'high']:
                model_path = os.path.join(model_save_dir, f'{ticker}_ddpg_actor_{risk_level}.pth')
                if os.path.exists(model_path):
                    print(f"Model for {ticker} - {risk_level} already exists. Skipping this risk level.")
                    continue
                risk_data = data[data['Risk_Level'] == risk_level_mapping[risk_level]]
                print(f"  {risk_level} risk data points: {len(risk_data)}")
                if risk_data.empty:
                    print(f"  Skipping {risk_level} risk: no data available")
                    continue
                agent = agents[risk_level]
                for episode in range(1000):  # you can reduce this for quicker testing
                    state = np.append(risk_data.iloc[0][['prediction']].values.astype(np.float32), [risk_level_mapping[risk_level]])
                    for t in range(1, len(risk_data)):
                        action = agent.select_action(state)
                        next_state = np.append(risk_data.iloc[t][['prediction']].values.astype(np.float32), [risk_level_mapping[risk_level]])
                        reward = calculate_reward(state, action, next_state)
                        agent.store_transition(state, action, reward, next_state, False)
                        agent.update()
                        state = next_state
                # Save trained actor model
                torch.save(agent.actor.state_dict(), model_path)
                print(f"Saved model for {ticker} - {risk_level} to {model_path}")
            # Get trading suggestions for each risk level
            current_state = data.iloc[-1][['prediction']].values.astype(np.float32)
            print("Trading Suggestions:")
            for risk_level, agent in agents.items():
                suggestion = get_trading_suggestions(agent, current_state, risk_level)
                print(f"{ticker} - {risk_level.capitalize()} Risk: {suggestion}")
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
        finally:
            conn.close()
            print("Database connection closed for ticker.")

if __name__ == "__main__":
    main()
