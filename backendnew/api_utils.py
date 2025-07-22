import os
import torch
import pandas as pd
from ddpg_agent import DDPGAgent
from db_config import get_connection
from difflib import get_close_matches

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
RISK_LEVELS = ['low', 'medium', 'high']
RISK_LEVEL_MAPPING = {'low': 0, 'medium': 1, 'high': 2}

# Model caching to avoid reloading models
_model_cache = {}

def get_cached_agent(ticker, risk_level):
    """Get or load DDPG agent from cache"""
    cache_key = f"{ticker}_{risk_level}"
    if cache_key not in _model_cache:
        model_path = os.path.join(MODEL_DIR, f"{ticker}_ddpg_actor_{risk_level}.pth")
        if os.path.exists(model_path):
            agent = DDPGAgent(state_dim=2, action_dim=1)
            agent.actor.load_state_dict(torch.load(model_path, map_location='cpu'))
            _model_cache[cache_key] = agent
        else:
            _model_cache[cache_key] = None
    return _model_cache[cache_key]


def get_available_tickers_and_risk_levels():
    tickers = sorted({f.split('_')[0] for f in os.listdir(MODEL_DIR) if f.endswith('_ddpg_actor_low.pth')})
    return {"tickers": tickers, "risk_levels": RISK_LEVELS}


def get_trading_suggestions(ticker, risk_level):
    agent = get_cached_agent(ticker, risk_level)
    if agent is None:
        return []
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
    if data.empty:
        return []
    data['risk_level'] = data['risk_level'].map(lambda x: RISK_LEVEL_MAPPING.get(x, x))
    data = data[data['risk_level'] == RISK_LEVEL_MAPPING[risk_level]]
    suggestions = []
    for i in range(len(data)):
        state = [data.iloc[i]['prediction'], data.iloc[i]['risk_level']]
        action = agent.select_action(state, add_noise=False)
        if action > 0.5:
            suggestion = 'Buy'
        elif action < -0.5:
            suggestion = 'Sell'
        else:
            suggestion = 'Hold'
        suggestions.append({
            "date": data.index[i],
            "prediction": data.iloc[i]['prediction'],
            "suggestion": suggestion
        })
    return suggestions


def get_action_for_ticker(ticker, risk_level, capital):
    agent = get_cached_agent(ticker, risk_level)
    if agent is None:
        return {"error": f"No model found for {ticker} with {risk_level} risk level."}
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
            risk_level_value = RISK_LEVEL_MAPPING.get(df.iloc[0]['risk_level'], 0)
            latest_state = [df.iloc[0]['prediction'], risk_level_value]
    if latest_state is None:
        latest_state = [0.0, 0]
    action = float(agent.select_action(latest_state, add_noise=False))
    if action > 0.5:
        amount_to_invest = float(max(0, action) * capital)
        suggestion = f"Buy (suggested amount: ${amount_to_invest:,.2f})"
    elif action < -0.5:
        suggestion = "Sell"
    else:
        suggestion = "Hold"
    return {
        "ticker": ticker,
        "risk_level": risk_level,
        "action": action,
        "suggestion": suggestion
    }


def get_top_bottom_tickers(risk_level, capital, top=True):
    tickers = sorted({f.split('_')[0] for f in os.listdir(MODEL_DIR) if f.endswith('_ddpg_actor_low.pth')})
    ticker_actions = []
    for ticker in tickers:
        agent = get_cached_agent(ticker, risk_level)
        if agent is not None:
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
                    risk_level_value = RISK_LEVEL_MAPPING.get(df.iloc[0]['risk_level'], 0)
                    latest_state = [df.iloc[0]['prediction'], risk_level_value]
            if latest_state is None:
                latest_state = [0.0, 0]
            action = float(agent.select_action(latest_state, add_noise=False))
            amount = float(max(0, action) * capital)
            ticker_actions.append((ticker, action, amount))
    sorted_tickers = sorted(ticker_actions, key=lambda x: x[1], reverse=top)[:3]
    result = []
    for t, a, amt in sorted_tickers:
        result.append({
            "ticker": t,
            "risk_level": risk_level,
            "signal": a,
            "suggested_amount": amt
        })
    return result


def chatbot_response(message, risk_level, capital):
    import re
    tickers = sorted({f.split('_')[0] for f in os.listdir(MODEL_DIR) if f.endswith('_ddpg_actor_low.pth')})
    available_tickers = set(tickers)
    words = re.findall(r'\b\w+\b', message.upper())
    ignore_words = {'SHOULD', 'I', 'DO', 'YOU', 'WE', 'THE', 'A', 'AN', 'TO', 'WITH', 'ON', 'IN', 'OF', 'IS', 'IT', 'FOR', 'AND', 'OR', 'PLEASE'}
    possible_tickers = [w for w in words if w not in ignore_words and w in available_tickers]
    if possible_tickers:
        user_ticker = possible_tickers[0]
        return get_action_for_ticker(user_ticker, risk_level, capital)
    else:
        user_input_lower = message.lower()
        if any(word in user_input_lower for word in ["sell", "underperform", "not performing", "worst", "bad"]):
            return {"bottom_tickers": get_top_bottom_tickers(risk_level, capital, top=False)}
        else:
            return {"top_tickers": get_top_bottom_tickers(risk_level, capital, top=True)} 

# --- Utility: Load ticker/company mapping from DB ---
def get_ticker_company_mapping():
    conn = get_connection()
    if conn:
        sql = "SELECT ticker, company FROM tickers"
        df = pd.read_sql(sql, conn)
        conn.close()
        ticker_to_company = dict(zip(df['ticker'], df['company']))
        company_to_ticker = {v.upper(): k for k, v in ticker_to_company.items()}
        return ticker_to_company, company_to_ticker
    else:
        return {}, {}

# --- Utility: Fuzzy find ticker from user input (ticker or company name) ---
def fuzzy_find_ticker(user_input):
    ticker_to_company, company_to_ticker = get_ticker_company_mapping()
    all_tickers = set(ticker_to_company.keys())
    all_companies = set(company_to_ticker.keys())
    user_input_upper = user_input.upper()
    user_input_lower = user_input.lower()
    # Direct ticker match
    if user_input_upper in all_tickers:
        return user_input_upper
    # Company name match (exact, case-insensitive, substring)
    for company in all_companies:
        if user_input_lower in company.lower():
            return company_to_ticker[company]
    # Fuzzy match ticker (lower cutoff)
    close = get_close_matches(user_input_upper, all_tickers, n=1, cutoff=0.6)
    if close:
        return close[0]
    # Fuzzy match company (lower cutoff)
    close = get_close_matches(user_input_lower, [c.lower() for c in all_companies], n=1, cutoff=0.6)
    if close:
        # Find the original company name (case-sensitive) that matches
        for company in all_companies:
            if company.lower() == close[0]:
                return company_to_ticker[company]
    return None

# --- Utility: Get all tickers ---
def get_all_tickers():
    ticker_to_company, _ = get_ticker_company_mapping()
    return list(ticker_to_company.keys())

# --- Utility: Get all company names ---
def get_all_companies():
    _, company_to_ticker = get_ticker_company_mapping()
    return list(company_to_ticker.keys()) 

def parse_user_input(user_input):
    user_input_lower = user_input.lower()
    ticker = fuzzy_find_ticker(user_input)
    # Existing intents
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
        if 'current price' in user_input_lower or 'price of' in user_input_lower:
            return 'current_price', ticker
        if 'highest price' in user_input_lower or 'high in the last' in user_input_lower:
            return 'highest_price', ticker
        if 'lowest price' in user_input_lower or 'low in the last' in user_input_lower:
            return 'lowest_price', ticker
        if 'average volume' in user_input_lower:
            return 'average_volume', ticker
        if 'rsi' in user_input_lower:
            return 'rsi', ticker
        if 'sma' in user_input_lower or 'moving average' in user_input_lower:
            return 'sma', ticker
        if 'trend' in user_input_lower:
            return 'trend', ticker
        if 'risk level' in user_input_lower:
            return 'risk_level', ticker
        if 'prediction' in user_input_lower and 'tomorrow' in user_input_lower:
            return 'prediction_tomorrow', ticker
        if 'sharpe' in user_input_lower:
            return 'sharpe_ratio', ticker
        if 'dividend' in user_input_lower:
            return 'dividend_yield', ticker
        if 'market cap' in user_input_lower:
            return 'market_cap', ticker
        if 'sector' in user_input_lower or 'industry' in user_input_lower:
            return 'sector', ticker
    # Non-ticker intents
    if 'top gainers' in user_input_lower:
        return 'top_gainers', None
    if 'top losers' in user_input_lower:
        return 'top_losers', None
    if 'top' in user_input_lower or 'best' in user_input_lower or 'buy' in user_input_lower:
        return 'top', None
    if 'worst' in user_input_lower or 'sell' in user_input_lower or 'underperform' in user_input_lower or 'bad' in user_input_lower:
        return 'bottom', None
    if 'volatile' in user_input_lower:
        return 'most_volatile', None
    if 'all available tickers' in user_input_lower or 'list tickers' in user_input_lower:
        return 'all_tickers', None
    if 'all available companies' in user_input_lower or 'list companies' in user_input_lower:
        return 'all_companies', None
    # Correlation
    if 'correlation' in user_input_lower:
        # Try to extract two tickers
        words = user_input_lower.split()
        tickers_found = [fuzzy_find_ticker(w) for w in words if fuzzy_find_ticker(w)]
        if len(tickers_found) >= 2:
            return 'correlation', tickers_found[:2]
    return 'unknown', None 