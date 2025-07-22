from fastapi import FastAPI, Query, HTTPException, Response
from pydantic import BaseModel
from typing import Optional
import sys
import os
from dotenv import load_dotenv
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from api_utils import get_all_tickers, get_all_companies, fuzzy_find_ticker, parse_user_input
from difflib import get_close_matches
import torch
import pandas as pd
from ddpg_agent import DDPGAgent
from db_config import get_connection
import requests
from fastapi.middleware.cors import CORSMiddleware


# Load environment variables
load_dotenv()

app = FastAPI(title="Trading Bot RESTful API", description="API for trading suggestions and chatbot", version="1.0")

pg_pool = psycopg2.pool.SimpleConnectionPool(
    minconn=1,
    maxconn=10,
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    database=os.getenv("DB_NAME")
)

origins = [
    "http://localhost:3000",               # Local dev frontend
    "http://127.0.0.1:3000",               # Just in case
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,            # For dev, you can use ["*"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ActionRequest(BaseModel):
    ticker: str
    risk_level: str
    capital: float

class ChatRequest(BaseModel):
    message: str
    risk_level: str
    capital: float

#Health API for uptime monitoring
@app.head("/")
@app.get("/")
def health_check():
    return Response(status_code=200)

@app.get("/tickers")
def tickers():
    """Get available tickers and risk levels."""
    return get_all_tickers()

@app.get("/companies")
def companies():
    """Get available companies."""
    return get_all_companies()

@app.get("/find_ticker")
def find_ticker(q: str = Query(...)):
    """Find a ticker by name."""
    return {"ticker": fuzzy_find_ticker(q)}

@app.get("/suggestion")
def suggestion(
    ticker: str = Query(..., description="Stock ticker symbol"),
    risk_level: str = Query(..., description="Risk level: low, medium, high")
):
    """Get trading suggestions for a ticker and risk level."""
    return {"suggestions": get_trading_suggestions(ticker, risk_level)}

@app.post("/action")
def action(req: ActionRequest):
    """Get buy/sell/hold suggestion for a ticker, risk level, and capital."""
    return get_action_for_ticker(req.ticker, req.risk_level, req.capital)

@app.get("/top-tickers")
def top_tickers(
    risk_level: str = Query(..., description="Risk level: low, medium, high"),
    capital: float = Query(..., description="Available capital")
):
    """Get top 3 tickers with the highest buy signals for a risk level and capital."""
    return {"top_tickers": get_top_bottom_tickers(risk_level, capital, top=True)}

@app.get("/bottom-tickers")
def bottom_tickers(
    risk_level: str = Query(..., description="Risk level: low, medium, high"),
    capital: float = Query(..., description="Available capital")
):
    """Get bottom 3 tickers with the strongest sell signals for a risk level and capital."""
    return {"bottom_tickers": get_top_bottom_tickers(risk_level, capital, top=False)}

@app.post("/chat")
def chat(req: ChatRequest):
    """Chatbot endpoint: ask for buy/sell/hold or top/bottom tickers."""
    return chatbot_response(req.message, req.risk_level, req.capital) 

@app.get("/prediction")
def get_prediction(
    ticker: str = Query(..., description="Stock ticker symbol"),
    risk_level: str = Query(None, description="Risk level: low, medium, high (optional)")
):
    """
    Get the latest LSTM prediction for a ticker (and optional risk level).
    """
    conn = get_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed.")
    try:
        if risk_level:
            sql = """
                SELECT date, prediction, risk_level
                FROM lstm_predictions
                WHERE ticker = %s AND risk_level = %s
                ORDER BY date DESC LIMIT 1
            """
            params = (ticker, risk_level)
        else:
            sql = """
                SELECT date, prediction, risk_level
                FROM lstm_predictions
                WHERE ticker = %s
                ORDER BY date DESC LIMIT 1
            """
            params = (ticker,)
        df = pd.read_sql(sql, conn, params=params)
        if df.empty:
            raise HTTPException(status_code=404, detail="No prediction found for the given ticker and risk level.")
        result = df.iloc[0].to_dict()
        return {"prediction": result}
    finally:
        conn.close()

def groq_fallback(user_input):
    # Simple keyword check for finance/trading context
    finance_keywords = [
        'stock', 'trade', 'buy', 'sell', 'hold', 'price', 'market', 'portfolio', 'investment', 'invest',
        'ticker', 'company', 'volume', 'rsi', 'sma', 'moving average', 'trend', 'risk', 'recommend',
        'prediction', 'volatility', 'gainers', 'losers', 'correlation', 'dividend', 'sharpe', 'capital',
        'sector', 'industry', 'profit', 'loss', 'return', 'yield', 'fundamental', 'technical', 'chart',
        'signal', 'analysis', 'exchange', 'nasdaq', 'nyse', 'equity', 'option', 'future', 'bond', 'etf',
        'index', 'indices', 'finance', 'financial', 'earnings', 'revenue', 'balance sheet', 'cash flow',
        'asset', 'liability', 'valuation', 'pe ratio', 'eps', 'dividend', 'split', 'ipo', 'buyback', 'merger',
        'acquisition', 'share', 'quote', 'bid', 'ask', 'spread', 'order', 'broker', 'account', 'margin', 'leverage'
    ]
    user_input_lower = user_input.lower()
    if not any(word in user_input_lower for word in finance_keywords):
        return "I'm just a trading bot AI."
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "API key not configured. Please set GROQ_API_KEY in your .env file."
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {
        "model": "llama3-70b-8192",
        "messages": [{"role": "user", "content": user_input}],
        "max_tokens": 150
    }
    try:
        response = requests.post(url, headers=headers, json=data, timeout=10)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return "Sorry, I couldn't answer that right now."
    except Exception as e:
        return "Sorry, I couldn't answer that right now."

@app.post("/chatbot")
def chatbot(request: ChatRequest):
    user_input = request.message
    risk_level = request.risk_level
    capital = request.capital

    # Model directory and tickers
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
    tickers = sorted({f.split('_')[0] for f in os.listdir(model_dir) if f.endswith('_ddpg_actor_low.pth')})

    # Load ticker-company mapping
    conn = get_connection()
    ticker_to_company = {}
    if conn:
        sql = "SELECT ticker, company FROM tickers"
        ticker_df = pd.read_sql(sql, conn)
        conn.close()
        ticker_to_company = dict(zip(ticker_df['ticker'], ticker_df['company']))

    # Use your intent parser
    intent, ticker = parse_user_input(user_input)

    # --- Buy/Sell/Hold/Recommend/Query ---
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
            return {"response": f"For {user_ticker} ({risk_level} risk, {ticker_to_company.get(user_ticker, user_ticker)}), the model suggests: **{suggestion}**"}
        else:
            return {"response": f"No model found for {user_ticker} with {risk_level} risk level."}

    # --- Last 5 Recommendations ---
    elif intent == 'last_recommendations' and ticker:
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
                return {"response": f"Last 5 recommendations for {ticker} ({ticker_to_company.get(ticker, ticker)}):", "data": df.to_dict(orient='records')}
            else:
                return {"response": f"No recent recommendations found for {ticker}."}

    # --- Volatility for Ticker ---
    elif intent == 'volatility' and ticker:
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
                return {"response": f"Volatility (std dev) for {ticker} ({ticker_to_company.get(ticker, ticker)}): {volatility:.4f}"}
            else:
                return {"response": f"No data to compute volatility for {ticker}."}

    # --- Most Volatile Tickers ---
    elif intent == 'most_volatile':
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
            return {"response": response}
        else:
            return {"response": "No volatility data available."}

    # --- Top 3 Buy Signals ---
    elif intent == 'top':
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
            return {"response": response}
        else:
            return {"response": "No buy recommendations found for any ticker at the selected risk level."}

    # --- Bottom 3 Sell Signals ---
    elif intent == 'bottom':
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
            return {"response": response}
        else:
            return {"response": "No strong sell recommendations found for any ticker at the selected risk level."}

    # --- Current Price ---
    elif intent == 'current_price' and ticker:
        conn = get_connection()
        if conn:
            sql = "SELECT close FROM stock_data WHERE ticker = %s ORDER BY date DESC LIMIT 1"
            df = pd.read_sql(sql, conn, params=(ticker,))
            conn.close()
            if not df.empty:
                price = df['close'].iloc[0]
                return {"response": f"The current price of {ticker} is ${price:.2f}."}
            else:
                return {"response": f"No price data found for {ticker}."}

    # --- Highest Price ---
    elif intent == 'highest_price' and ticker:
        conn = get_connection()
        if conn:
            sql = "SELECT MAX(high) as max_high FROM stock_data WHERE ticker = %s AND date >= NOW() - INTERVAL '30 days'"
            df = pd.read_sql(sql, conn, params=(ticker,))
            conn.close()
            if not df.empty and pd.notnull(df['max_high'].iloc[0]):
                return {"response": f"The highest price of {ticker} in the last 30 days was ${df['max_high'].iloc[0]:.2f}."}
            else:
                return {"response": f"No high price data found for {ticker}."}

    # --- Lowest Price ---
    elif intent == 'lowest_price' and ticker:
        conn = get_connection()
        if conn:
            sql = "SELECT MIN(low) as min_low FROM stock_data WHERE ticker = %s AND date >= NOW() - INTERVAL '30 days'"
            df = pd.read_sql(sql, conn, params=(ticker,))
            conn.close()
            if not df.empty and pd.notnull(df['min_low'].iloc[0]):
                return {"response": f"The lowest price of {ticker} in the last 30 days was ${df['min_low'].iloc[0]:.2f}."}
            else:
                return {"response": f"No low price data found for {ticker}."}

    # --- Average Volume ---
    elif intent == 'average_volume' and ticker:
        conn = get_connection()
        if conn:
            sql = "SELECT AVG(volume) as avg_vol FROM stock_data WHERE ticker = %s AND date >= NOW() - INTERVAL '30 days'"
            df = pd.read_sql(sql, conn, params=(ticker,))
            conn.close()
            if not df.empty and pd.notnull(df['avg_vol'].iloc[0]):
                return {"response": f"The average volume of {ticker} in the last 30 days was {int(df['avg_vol'].iloc[0]):,}."}
            else:
                return {"response": f"No volume data found for {ticker}."}

    # --- RSI ---
    elif intent == 'rsi' and ticker:
        conn = get_connection()
        if conn:
            sql = "SELECT rsi FROM stock_data WHERE ticker = %s ORDER BY date DESC LIMIT 1"
            df = pd.read_sql(sql, conn, params=(ticker,))
            conn.close()
            if not df.empty and pd.notnull(df['rsi'].iloc[0]):
                return {"response": f"The latest RSI for {ticker} is {df['rsi'].iloc[0]:.2f}."}
            else:
                return {"response": f"No RSI data found for {ticker}."}

    # --- SMA ---
    elif intent == 'sma' and ticker:
        conn = get_connection()
        if conn:
            sql = "SELECT sma_10, sma_50 FROM stock_data WHERE ticker = %s ORDER BY date DESC LIMIT 1"
            df = pd.read_sql(sql, conn, params=(ticker,))
            conn.close()
            if not df.empty:
                return {"response": f"The latest SMA-10 for {ticker} is {df['sma_10'].iloc[0]:.2f}, SMA-50 is {df['sma_50'].iloc[0]:.2f}."}
            else:
                return {"response": f"No SMA data found for {ticker}."}

    # --- All Tickers ---
    elif intent == 'all_tickers':
        tickers = get_all_tickers()
        return {"response": "Available tickers: " + ', '.join(tickers)}

    # --- All Companies ---
    elif intent == 'all_companies':
        companies = get_all_companies()
        return {"response": "Available companies: " + ', '.join(companies)}

    # --- Correlation ---
    elif intent == 'correlation' and ticker and isinstance(ticker, list) and len(ticker) == 2:
        t1, t2 = ticker
        conn = get_connection()
        if conn:
            sql = """
                SELECT a.date, a.close as close1, b.close as close2
                FROM stock_data a
                JOIN stock_data b ON a.date = b.date
                WHERE a.ticker = %s AND b.ticker = %s AND a.date >= NOW() - INTERVAL '90 days'
                ORDER BY a.date DESC
            """
            df = pd.read_sql(sql, conn, params=(t1, t2))
            conn.close()
            if not df.empty:
                corr = df['close1'].corr(df['close2'])
                return {"response": f"The correlation between {t1} and {t2} over the last 90 days is {corr:.2f}."}
            else:
                return {"response": f"No correlation data found for {t1} and {t2}."}

    # --- Fallback: Groq LLM for unknown or unsupported questions ---
    else:
        llm_response = groq_fallback(user_input)
        return {"response": llm_response} 
    
@app.get("/prediction_by_date")
def prediction_by_date(
    ticker: str = Query(..., description="Stock ticker symbol"),
    date: str = Query(..., description="Prediction date in YYYY-MM-DD format")
):
    """
    Get the LSTM prediction(s) for a ticker on a specific date, returning all available risk levels.
    """
    conn = get_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed.")
    try:
        sql = """
            SELECT date, prediction, risk_level
            FROM lstm_predictions
            WHERE ticker = %s AND date = %s
        """
        params = (ticker, date)
        df = pd.read_sql(sql, conn, params=params)
        if df.empty:
            raise HTTPException(status_code=404, detail="No prediction found for the given ticker and date.")
        results = df.to_dict(orient='records')
        return {"predictions": results}
    finally:
        conn.close()