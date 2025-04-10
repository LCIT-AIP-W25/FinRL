import os
import pandas as pd
import numpy as np
import torch
from ddpg_agent import DDPGAgent

# Directory
prediction_dir = 'predictions'
model_save_dir = 'models'
os.makedirs(model_save_dir, exist_ok=True)

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

# Loop through all predicted CSVs
for file_name in os.listdir(prediction_dir):
    if file_name.endswith('_future_predictions.csv'):
        symbol = file_name.split('_')[0]
        print(f"\nProcessing {symbol}...")

        try:
            # Load predicted data
            data_path = os.path.join(prediction_dir, file_name)
            data = pd.read_csv(data_path, parse_dates=['Date'], index_col='Date')

            # Compute volatility and categorize risk
            data['Volatility'] = data['LSTM_Prediction'].pct_change().rolling(window=30).std()
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
                risk_data = data[data['Risk_Level'] == risk_level_mapping[risk_level]]
                if risk_data.empty:
                    continue
                agent = agents[risk_level]

                for episode in range(1000):  # you can reduce this for quicker testing
                    state = np.append(risk_data.iloc[0][['LSTM_Prediction']].values.astype(np.float32), [risk_level_mapping[risk_level]])
                    for t in range(1, len(risk_data)):
                        action = agent.select_action(state)
                        next_state = np.append(risk_data.iloc[t][['LSTM_Prediction']].values.astype(np.float32), [risk_level_mapping[risk_level]])
                        reward = calculate_reward(state, action, next_state)
                        agent.store_transition(state, action, reward, next_state)
                        agent.update()
                        state = next_state

            # Save trained actor models
            for risk_level in ['low', 'medium', 'high']:
                model_path = os.path.join(model_save_dir, f'{symbol}_ddpg_actor_{risk_level}.pth')
                torch.save(agents[risk_level].actor.state_dict(), model_path)

            # Get trading suggestions for each risk level
            current_state = data.iloc[-1][['LSTM_Prediction']].values.astype(np.float32)
            print("Trading Suggestions:")
            for risk_level, agent in agents.items():
                suggestion = get_trading_suggestions(agent, current_state, risk_level)
                print(f"{symbol} - {risk_level.capitalize()} Risk: {suggestion}")

        except Exception as e:
            print(f"Error processing {symbol}: {e}")
