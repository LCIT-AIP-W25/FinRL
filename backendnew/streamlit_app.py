import streamlit as st
import pandas as pd
import torch
import os
import sys
import re
from difflib import get_close_matches
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ddpg_agent import DDPGAgent
from db_config import get_connection

# --- Load ticker-company mapping from Postgres ---
conn = get_connection()
if conn:
    sql = "SELECT ticker, company FROM tickers"
    ticker_df = pd.read_sql(sql, conn)
    conn.close()
    ticker_to_company = dict(zip(ticker_df['ticker'], ticker_df['company']))
    company_to_ticker = {v.upper(): k for k, v in ticker_to_company.items()}
    all_tickers = set(ticker_to_company.keys())
    all_companies = set(company_to_ticker.keys())
else:
    st.error("Could not connect to the database to load tickers.")
    ticker_to_company = {}
    company_to_ticker = {}
    all_tickers = set()
    all_companies = set()

# --- Fuzzy matching function ---
def find_ticker(user_input):
    user_input = user_input.upper()
    # Direct ticker match
    for ticker in all_tickers:
        if ticker in user_input:
            return ticker
    # Company name match (exact)
    for company in all_companies:
        if company in user_input:
            return company_to_ticker[company]
    # Fuzzy match ticker
    words = re.findall(r'\b\w+\b', user_input)
    close = get_close_matches(user_input, all_tickers, n=1, cutoff=0.8)
    if close:
        return close[0]
    # Fuzzy match company
    close = get_close_matches(user_input, all_companies, n=1, cutoff=0.8)
    if close:
        return company_to_ticker[close[0]]
    return None

# --- Expanded intent parser ---
def parse_user_input(user_input):
    user_input_lower = user_input.lower()
    ticker = find_ticker(user_input)
    if ticker:
        if any(word in user_input_lower for word in ['buy', 'invest', 'purchase']):
            return 'buy', ticker
        if any(word in user_input_lower for word in ['sell', 'exit', 'offload']):
            return 'sell', ticker
        if 'hold' in user_input_lower:
            return 'hold', ticker
        if 'recommend' in user_input_lower or 'suggest' in user_input_lower:
            return 'recommend', ticker
        if 'last' in user_input_lower and 'recommend' in user_input_lower:
            return 'last_recommendations', ticker
        if 'volatil' in user_input_lower:
            return 'volatility', ticker
        return 'ticker_query', ticker
    # Top/bottom
    if any(word in user_input_lower for word in ['top', 'best', 'buy']):
        return 'top', None
    if any(word in user_input_lower for word in ['worst', 'sell', 'underperform', 'bad']):
        return 'bottom', None
    if 'volatile' in user_input_lower:
        return 'most_volatile', None
    return 'unknown', None

# --- Sidebar help/examples ---
st.sidebar.markdown("""
**Example questions:**
- Should I buy AAPL?
- Should I invest in Apple?
- What are the top 3 stocks to buy?
- Show me the worst performing tickers.
- How much should I invest in TSLA?
- Show me the last 5 recommendations for MSFT.
- Which stocks are most volatile today?
""")

st.title("DDPG Trading Suggestions Dashboard")

# Set model directory
model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))

# Get available tickers from models
tickers = sorted({f.split('_')[0] for f in os.listdir(model_dir) if f.endswith('_ddpg_actor_low.pth')})
risk_levels = ['low', 'medium', 'high']

# User selection
ticker = st.selectbox("Select Ticker", tickers)
risk_level = st.selectbox("Select Risk Level", risk_levels)

# Load the trained DDPG model
model_path = os.path.join(model_dir, f"{ticker}_ddpg_actor_{risk_level}.pth")
agent = DDPGAgent(state_dim=2, action_dim=1)
agent.actor.load_state_dict(torch.load(model_path, map_location='cpu'))

# Load predictions from database
conn = get_connection()
data = pd.DataFrame()
if conn:
    sql = """
        SELECT date, prediction, risk_level
        FROM lstm_predictions
        WHERE ticker = %s
        ORDER BY date
    """
    data = pd.read_sql(sql, conn, params=(ticker,), index_col='date', parse_dates=['date'])
    conn.close()

if not data.empty:
    # Map risk_level to int if needed
    risk_level_mapping = {'low': 0, 'medium': 1, 'high': 2}
    data['risk_level'] = data['risk_level'].map(lambda x: risk_level_mapping.get(x, x))
    # Filter by selected risk level
    data = data[data['risk_level'] == risk_level_mapping[risk_level]]
    # Get trading suggestions
    suggestions = []
    for i in range(len(data)):
        state = [data.iloc[i]['prediction'], data.iloc[i]['risk_level']]
        action = agent.select_action(state, add_noise=False)
        if action > 0.5:
            suggestions.append('Buy')
        elif action < -0.5:
            suggestions.append('Sell')
        else:
            suggestions.append('Hold')
    data['suggestion'] = suggestions
    st.write(data[['prediction', 'suggestion']])
    st.line_chart(data['prediction'])
    st.write("Buy/Hold/Sell counts:", data['suggestion'].value_counts())
else:
    st.warning("No prediction data found for this ticker and risk level.")

# Add a number input for available capital
capital = st.number_input("Enter your available capital ($)", min_value=0.0, value=10000.0, step=100.0)

# Get the list of available tickers
available_tickers = set(tickers)

st.header("Ask the Trading Bot")

# Replace user_input handling with new parser
user_input = st.chat_input("Ask if you should buy, sell, or hold a stock (e.g., 'Should I buy AAPL?')")

if user_input:
    intent, ticker = parse_user_input(user_input)
    if intent in ['buy', 'sell', 'hold', 'recommend', 'ticker_query'] and ticker:
        user_ticker = ticker
        model_path = os.path.join(model_dir, f"{user_ticker}_ddpg_actor_{risk_level}.pth")
        if os.path.exists(model_path):
            agent = DDPGAgent(state_dim=2, action_dim=1)
            agent.actor.load_state_dict(torch.load(model_path, map_location='cpu'))
            # Try to get the latest prediction from the database
            conn = get_connection()
            latest_state = None
            if conn:
                sql = """
                    SELECT prediction, risk_level
                    FROM lstm_predictions
                    WHERE ticker = %s AND risk_level = %s
                    ORDER BY date DESC LIMIT 1
                """
                params = (user_ticker, risk_level)
                df = pd.read_sql(sql, conn, params=params)
                conn.close()
                if not df.empty:
                    risk_level_mapping = {'low': 0, 'medium': 1, 'high': 2}
                    risk_level_value = risk_level_mapping.get(df.iloc[0]['risk_level'], 0)
                    latest_state = [df.iloc[0]['prediction'], risk_level_value]
            if latest_state is None:
                latest_state = [0.0, 0]
            action = float(agent.select_action(latest_state, add_noise=False))
            if intent == 'buy':
                amount_to_invest = float(max(0, action) * capital)
                suggestion = f"Buy (suggested amount: ${amount_to_invest:,.2f})"
            elif intent == 'sell':
                suggestion = "Sell"
            elif intent == 'hold':
                suggestion = "Hold"
            else:
                if action > 0.5:
                    amount_to_invest = float(max(0, action) * capital)
                    suggestion = f"Buy (suggested amount: ${amount_to_invest:,.2f})"
                elif action < -0.5:
                    suggestion = "Sell"
                else:
                    suggestion = "Hold"
            st.chat_message("bot").write(f"For {user_ticker} ({risk_level} risk, {ticker_to_company.get(user_ticker, user_ticker)}), the model suggests: **{suggestion}**")
        else:
            st.chat_message("bot").write(f"No model found for {user_ticker} with {risk_level} risk level.")
    elif intent == 'last_recommendations' and ticker:
        # Show last 5 recommendations for ticker
        conn = get_connection()
        if conn:
            sql = """
                SELECT date, prediction, risk_level
                FROM lstm_predictions
                WHERE ticker = %s
                ORDER BY date DESC LIMIT 5
            """
            df = pd.read_sql(sql, conn, params=(ticker,), parse_dates=['date'])
            conn.close()
            if not df.empty:
                st.chat_message("bot").write(f"Last 5 recommendations for {ticker} ({ticker_to_company.get(ticker, ticker)}):")
                st.write(df)
            else:
                st.chat_message("bot").write(f"No recent recommendations found for {ticker}.")
    elif intent == 'volatility' and ticker:
        # Show volatility for ticker (example: std of predictions)
        conn = get_connection()
        if conn:
            sql = """
                SELECT prediction
                FROM lstm_predictions
                WHERE ticker = %s
                ORDER BY date DESC LIMIT 30
            """
            df = pd.read_sql(sql, conn, params=(ticker,))
            conn.close()
            if not df.empty:
                volatility = df['prediction'].std()
                st.chat_message("bot").write(f"Volatility (std dev) for {ticker} ({ticker_to_company.get(ticker, ticker)}): {volatility:.4f}")
            else:
                st.chat_message("bot").write(f"No data to compute volatility for {ticker}.")
    elif intent == 'most_volatile':
        # Show most volatile tickers (top 3 by std dev)
        conn = get_connection()
        ticker_vols = []
        if conn:
            for ticker in tickers:
                sql = """
                    SELECT prediction
                    FROM lstm_predictions
                    WHERE ticker = %s
                    ORDER BY date DESC LIMIT 30
                """
                df = pd.read_sql(sql, conn, params=(ticker,))
                if not df.empty:
                    vol = df['prediction'].std()
                    ticker_vols.append((ticker, vol))
            conn.close()
        if ticker_vols:
            top_vols = sorted(ticker_vols, key=lambda x: x[1], reverse=True)[:3]
            response = "Top 3 most volatile tickers (by std dev of prediction):\n"
            for t, v in top_vols:
                response += f"- **{t}** ({ticker_to_company.get(t, t)}): {v:.4f}\n"
            st.chat_message("bot").write(response)
        else:
            st.chat_message("bot").write("No volatility data available.")
    elif intent == 'top':
        # Show top 3 buy signals
        ticker_actions = []
        for ticker in tickers:
            model_path = os.path.join(model_dir, f"{ticker}_ddpg_actor_{risk_level}.pth")
            if os.path.exists(model_path):
                agent = DDPGAgent(state_dim=2, action_dim=1)
                agent.actor.load_state_dict(torch.load(model_path, map_location='cpu'))
                conn = get_connection()
                latest_state = None
                if conn:
                    sql = """
                        SELECT prediction, risk_level
                        FROM lstm_predictions
                        WHERE ticker = %s AND risk_level = %s
                        ORDER BY date DESC LIMIT 1
                    """
                    params = (ticker, risk_level)
                    df = pd.read_sql(sql, conn, params=params)
                    conn.close()
                    if not df.empty:
                        risk_level_mapping = {'low': 0, 'medium': 1, 'high': 2}
                        risk_level_value = risk_level_mapping.get(df.iloc[0]['risk_level'], 0)
                        latest_state = [df.iloc[0]['prediction'], risk_level_value]
                if latest_state is None:
                    latest_state = [0.0, 0]
                action = float(agent.select_action(latest_state, add_noise=False))
                amount = float(max(0, action) * capital)
                ticker_actions.append((ticker, action, amount))
        top_tickers = sorted(ticker_actions, key=lambda x: x[1], reverse=True)[:3]
        if top_tickers:
            response = "Top 3 tickers with the highest buy signals:\n"
            for t, a, amt in top_tickers:
                response += f"- **{t}** ({risk_level} risk, {ticker_to_company.get(t, t)}): Buy signal = {a:.2f}, Suggested amount: **${amt:,.2f}**\n"
            st.chat_message("bot").write(response)
        else:
            st.chat_message("bot").write("No buy recommendations found for any ticker at the selected risk level.")
    elif intent == 'bottom':
        # Show bottom 3 sell signals
        ticker_actions = []
        for ticker in tickers:
            model_path = os.path.join(model_dir, f"{ticker}_ddpg_actor_{risk_level}.pth")
            if os.path.exists(model_path):
                agent = DDPGAgent(state_dim=2, action_dim=1)
                agent.actor.load_state_dict(torch.load(model_path, map_location='cpu'))
                conn = get_connection()
                latest_state = None
                if conn:
                    sql = """
                        SELECT prediction, risk_level
                        FROM lstm_predictions
                        WHERE ticker = %s AND risk_level = %s
                        ORDER BY date DESC LIMIT 1
                    """
                    params = (ticker, risk_level)
                    df = pd.read_sql(sql, conn, params=params)
                    conn.close()
                    if not df.empty:
                        risk_level_mapping = {'low': 0, 'medium': 1, 'high': 2}
                        risk_level_value = risk_level_mapping.get(df.iloc[0]['risk_level'], 0)
                        latest_state = [df.iloc[0]['prediction'], risk_level_value]
                if latest_state is None:
                    latest_state = [0.0, 0]
                action = float(agent.select_action(latest_state, add_noise=False))
                ticker_actions.append((ticker, action))
        bottom_tickers = sorted(ticker_actions, key=lambda x: x[1])[:3]
        if bottom_tickers:
            response = "Bottom 3 tickers with the strongest sell signals:\n"
            for t, a in bottom_tickers:
                response += f"- **{t}** ({risk_level} risk, {ticker_to_company.get(t, t)}): Sell signal = {a:.2f}, Suggested action: **Sell**\n"
            st.chat_message("bot").write(response)
        else:
            st.chat_message("bot").write("No strong sell recommendations found for any ticker at the selected risk level.")
    else:
        st.chat_message("bot").write("Sorry, I didn't understand your question. Try asking about a specific ticker, company, or use one of the example questions in the sidebar.") 