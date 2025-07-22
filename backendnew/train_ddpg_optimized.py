import os
import pandas as pd
import numpy as np
import torch
from ddpg_agent import DDPGAgent
import sys
from tqdm import tqdm
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from db_config import get_connection

# Directory
model_save_dir = 'models'
os.makedirs(model_save_dir, exist_ok=True)

# Configuration
MAX_EPISODES = 100  # Reduced from 1000
BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 10
MIN_DATA_POINTS = 5  # Lowered to work with current data

# Utility functions
def categorize_risk_levels(data):
    data['Risk_Level'] = 'low'
    data.loc[data['Volatility'] > 0.02, 'Risk_Level'] = 'medium'
    data.loc[data['Volatility'] > 0.05, 'Risk_Level'] = 'high'
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
    """Get all tickers with connection pooling"""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT ticker FROM lstm_predictions ORDER BY ticker")
            tickers = [row[0] for row in cur.fetchall()]
        return tickers
    finally:
        conn.close()

def get_predictions_for_ticker(conn, ticker):
    """Get predictions with better error handling"""
    sql = """
        SELECT date, prediction, risk_level
        FROM lstm_predictions
        WHERE ticker = %s
        ORDER BY date
    """
    try:
        df = pd.read_sql(sql, conn, params=(ticker,), index_col='date', parse_dates=['date'])
        return df
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

def train_agent_optimized(agent, risk_data, risk_level, ticker):
    """Optimized training with early stopping and batch processing"""
    if len(risk_data) < MIN_DATA_POINTS:
        return False
    
    best_loss = float('inf')
    patience_counter = 0
    
    # Prepare data
    states = []
    next_states = []
    rewards = []
    
    for i in range(len(risk_data) - 1):
        state = np.append(risk_data.iloc[i][['prediction']].values.astype(np.float32), [risk_level_mapping[risk_level]])
        next_state = np.append(risk_data.iloc[i+1][['prediction']].values.astype(np.float32), [risk_level_mapping[risk_level]])
        reward = calculate_reward(state, next_state, next_state)
        
        states.append(state)
        next_states.append(next_state)
        rewards.append(reward)
    
    states = np.array(states)
    next_states = np.array(next_states)
    rewards = np.array(rewards)
    
    # Training loop with early stopping
    for episode in range(MAX_EPISODES):
        episode_loss = 0
        num_batches = len(states) // BATCH_SIZE
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = start_idx + BATCH_SIZE
            
            batch_states = states[start_idx:end_idx]
            batch_next_states = next_states[start_idx:end_idx]
            batch_rewards = rewards[start_idx:end_idx]
            
            # Generate random actions for exploration
            batch_actions = np.random.uniform(-1, 1, size=(len(batch_states), 1))
            
            # Store transitions
            for i in range(len(batch_states)):
                agent.store_transition(batch_states[i], batch_actions[i], batch_rewards[i], batch_next_states[i], False)
            
            # Update agent
            if len(agent.memory) >= BATCH_SIZE:
                agent.update()
                episode_loss += agent.critic_losses[-1] if agent.critic_losses else 0
        
        # Early stopping check
        avg_loss = episode_loss / max(num_batches, 1)
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping for {ticker} - {risk_level} at episode {episode}")
            break
    
    return True

def main():
    print("🚀 Optimized DDPG training script started")
    start_time = time.time()
    
    print("📊 Getting tickers from database...")
    tickers = get_tickers_from_db()
    print(f"✅ Found {len(tickers)} tickers")
    
    # Filter tickers that need training
    tickers_to_train = []
    for ticker in tickers:
        models_exist = all(
            os.path.exists(os.path.join(model_save_dir, f'{ticker}_ddpg_actor_{risk}.pth'))
            for risk in ['low', 'medium', 'high']
        )
        if not models_exist:
            tickers_to_train.append(ticker)
    
    print(f"🎯 {len(tickers_to_train)} tickers need training")
    
    # Single database connection for all operations
    conn = get_connection()
    
    try:
        for ticker in tqdm(tickers_to_train, desc="Training DDPG models"):
            print(f"\n🔄 Processing {ticker}...")
            
            try:
                data = get_predictions_for_ticker(conn, ticker)
                if data.empty or len(data) < MIN_DATA_POINTS:
                    print(f"⏭️ Skipping {ticker}: insufficient data ({len(data)} points)")
                    continue
                
                # Compute volatility and categorize risk
                data['Volatility'] = data['prediction'].pct_change().rolling(window=30).std()
                data = categorize_risk_levels(data)
                data['Risk_Level'] = data['Risk_Level'].map(risk_level_mapping)
                
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
                        print(f"✅ Model for {ticker} - {risk_level} already exists")
                        continue
                    
                    risk_data = data[data['Risk_Level'] == risk_level_mapping[risk_level]]
                    if risk_data.empty:
                        print(f"⚠️ No {risk_level} risk data for {ticker}")
                        continue
                    
                    print(f"🎯 Training {ticker} - {risk_level} risk...")
                    agent = agents[risk_level]
                    
                    if train_agent_optimized(agent, risk_data, risk_level, ticker):
                        # Save trained actor model
                        torch.save(agent.actor.state_dict(), model_path)
                        print(f"💾 Saved model for {ticker} - {risk_level}")
                    else:
                        print(f"❌ Failed to train {ticker} - {risk_level}")
                
                # Show trading suggestions
                current_state = data.iloc[-1][['prediction']].values.astype(np.float32)
                print(f"📈 Trading Suggestions for {ticker}:")
                for risk_level, agent in agents.items():
                    suggestion = get_trading_suggestions(agent, current_state, risk_level)
                    print(f"  {risk_level.capitalize()} Risk: {suggestion}")
                    
            except Exception as e:
                print(f"❌ Error processing {ticker}: {e}")
                continue
                
    finally:
        conn.close()
        print("🔌 Database connection closed")
    
    total_time = time.time() - start_time
    print(f"\n🎉 Training completed in {total_time:.2f} seconds")
    print(f"⏱️ Average time per ticker: {total_time/len(tickers_to_train):.2f} seconds")

if __name__ == "__main__":
    main() 