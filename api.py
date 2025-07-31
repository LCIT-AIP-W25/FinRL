import os
import pandas as pd
import numpy as np
import torch
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from functools import lru_cache
from dotenv import load_dotenv

# Global cache for models and data
_model_cache = {}
_data_cache = {}
_cache_lock = threading.Lock()

def normalize_risk_level(risk_level: str) -> str:
    """Normalize risk level to lowercase for case-insensitive comparison."""
    return risk_level.lower() if risk_level else "medium"

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="FinRL Trading API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

def load_model_cached(ticker: str, risk_level: str):
    """Load model with caching to avoid repeated disk I/O."""
    cache_key = f"{ticker}_{risk_level}"
    
    with _cache_lock:
        if cache_key in _model_cache:
            return _model_cache[cache_key]
        
        try:
            model_dir = os.path.join(os.path.dirname(__file__), 'models')
            model_path = os.path.join(model_dir, f'{ticker}_ddpg_actor_{risk_level}.pth')
            
            if os.path.exists(model_path):
                agent = DDPGAgent(state_dim=2, action_dim=1)
                agent.actor.load_state_dict(torch.load(model_path, map_location='cpu'))
                agent.actor.eval()
                _model_cache[cache_key] = agent
                return agent
            else:
                return None
        except Exception as e:
            print(f"Error loading model for {ticker}_{risk_level}: {e}")
            return None

def get_batch_historical_data(tickers: List[str], days: int = 252):
    """Fetch historical data for multiple tickers in one query."""
    conn = get_connection()
    if not conn:
        return {}
    
    try:
        # Create placeholders for SQL IN clause
        placeholders = ','.join(['%s'] * len(tickers))
        sql = f"""
            SELECT ticker, close, date 
            FROM stock_data 
            WHERE ticker IN ({placeholders})
            ORDER BY ticker, date DESC
        """
        
        df = pd.read_sql(sql, conn, params=tickers)
        
        # Group by ticker and process
        historical_data = {}
        for ticker in tickers:
            ticker_data = df[df['ticker'] == ticker].head(days)
            if len(ticker_data) >= 60:  # Minimum data requirement
                ticker_data['date'] = pd.to_datetime(ticker_data['date'])
                ticker_data = ticker_data.sort_values('date')
                ticker_data['return'] = ticker_data['close'].pct_change()
                
                # Calculate monthly returns
                ticker_data.set_index('date', inplace=True)
                monthly_returns = ticker_data['return'].resample('ME').apply(lambda x: (1 + x).prod() - 1)
                monthly_returns = monthly_returns.dropna()
                
                if len(monthly_returns) >= 3:
                    historical_data[ticker] = {
                        'avg_monthly_return': monthly_returns.mean(),
                        'monthly_volatility': monthly_returns.std(),
                        'months_of_data': len(monthly_returns)
                    }
        
        return historical_data
        
    except Exception as e:
        print(f"Error fetching batch historical data: {e}")
        return {}
    finally:
        return_connection(conn)

def get_batch_company_names(tickers: List[str]):
    """Fetch company names for multiple tickers in one query."""
    conn = get_connection()
    if not conn:
        print("❌ No database connection available")
        return {}
    
    # Fallback company names for common stocks
    fallback_companies = {
        'AAPL': 'Apple Inc.',
        'MSFT': 'Microsoft Corporation',
        'GOOG': 'Alphabet Inc.',
        'AMZN': 'Amazon.com Inc.',
        'META': 'Meta Platforms Inc.',
        'NVDA': 'NVIDIA Corporation',
        'TSLA': 'Tesla Inc.',
        'NFLX': 'Netflix Inc.',
        'CRM': 'Salesforce Inc.',
        'ADBE': 'Adobe Inc.',
        'BAC': 'Bank of America Corp.',
        'C': 'Citigroup Inc.',
        'BLK': 'BlackRock Inc.',
        'AXP': 'American Express Co.',
        'COF': 'Capital One Financial Corp.',
        'ABBV': 'AbbVie Inc.',
        'ABT': 'Abbott Laboratories',
        'BMY': 'Bristol-Myers Squibb Co.',
        'AMGN': 'Amgen Inc.',
        'COST': 'Costco Wholesale Corp.',
        'AMD': 'Advanced Micro Devices Inc.',
        'AVGO': 'Broadcom Inc.'
    }
    
    try:
        placeholders = ','.join(['%s'] * len(tickers))
        sql = f"SELECT ticker, company FROM tickers WHERE ticker IN ({placeholders})"
        df = pd.read_sql(sql, conn, params=tickers)
        company_dict = dict(zip(df['ticker'], df['company']))
        
        # Add fallback names for any missing companies
        for ticker in tickers:
            if ticker not in company_dict or company_dict[ticker] == ticker:
                company_dict[ticker] = fallback_companies.get(ticker, f"{ticker} Corporation")
        
        # Debug: Print the results
        print(f"get_batch_company_names: Processed {len(tickers)} tickers, found {len(company_dict)} companies")
        for ticker in tickers[:5]:  # Show first 5 for debugging
            company = company_dict.get(ticker, ticker)
            print(f"  {ticker}: {company}")
        
        return company_dict
    except Exception as e:
        print(f"Error fetching company names: {e}")
        # Return fallback names if database query fails
        return {ticker: fallback_companies.get(ticker, f"{ticker} Corporation") for ticker in tickers}
    finally:
        return_connection(conn)

def process_ticker_batch_optimized(ticker_batch: List[str], risk_level: str, amount: float, months: int, historical_data: Dict, ticker_to_company: Dict):
    """Process a batch of tickers with optimized logic."""
    results = []
    
    for ticker in ticker_batch:
        try:
            # Load model with caching
            agent = load_model_cached(ticker, risk_level)
            if agent is None:
                continue
            
            # Get latest prediction
            conn = get_connection()
            latest_state = None
            if conn:
                try:
                    sql = """
                        SELECT prediction, risk_level
                        FROM lstm_predictions
                        WHERE ticker = %s AND risk_level = %s
                        ORDER BY date DESC LIMIT 1
                    """
                    params = (ticker, risk_level)
                    df = pd.read_sql(sql, conn, params=params)
                    if not df.empty:
                        risk_level_mapping = {'low': 0, 'medium': 1, 'high': 2}
                        risk_level_value = risk_level_mapping.get(df.iloc[0]['risk_level'], 0)
                        latest_state = [df.iloc[0]['prediction'], risk_level_value]
                except Exception as e:
                    print(f"Error getting prediction for {ticker}: {e}")
                finally:
                    return_connection(conn)
            
            if latest_state is None:
                risk_level_mapping = {'low': 0, 'medium': 1, 'high': 2}
                risk_level_value = risk_level_mapping.get(risk_level, 1)
                latest_state = [0.0, risk_level_value]
            
            # Get action from model
            action = float(agent.select_action(latest_state, add_noise=False))
            normalized_action = (action + 1) / 2
            
            # Use pre-fetched historical data
            ticker_history = historical_data.get(ticker)
            
            if ticker_history:
                # Use historical data
                avg_monthly_return = ticker_history['avg_monthly_return']
                volatility = ticker_history['monthly_volatility']
                sharpe_ratio = avg_monthly_return / (volatility + 1e-8)
                
                # Adjust based on model action
                if risk_level == 'low':
                    model_adjustment = (normalized_action - 0.5) * 0.02  # ±1% adjustment
                    expected_monthly_return = avg_monthly_return + model_adjustment
                elif risk_level == 'medium':
                    model_adjustment = (normalized_action - 0.5) * 0.03  # ±1.5% adjustment
                    expected_monthly_return = avg_monthly_return + model_adjustment
                else:  # high risk
                    model_adjustment = (normalized_action - 0.5) * 0.05  # ±2.5% adjustment
                    expected_monthly_return = avg_monthly_return + model_adjustment
                
                # Apply bounds
                if risk_level == 'low':
                    expected_monthly_return = np.clip(expected_monthly_return, 0.002, 0.026)
                elif risk_level == 'medium':
                    expected_monthly_return = np.clip(expected_monthly_return, -0.002, 0.024)
                else:
                    expected_monthly_return = np.clip(expected_monthly_return, -0.004, 0.035)
                
                expected_total_return = expected_monthly_return * months
                expected_profit = amount * expected_total_return
                confidence = min(0.95, max(0.3, abs(normalized_action - 0.5) * 2 + 0.5))
                
            else:
                # Use default values
                if risk_level == 'low':
                    avg_monthly_return = 0.013
                    volatility = 0.04
                elif risk_level == 'medium':
                    avg_monthly_return = 0.008
                    volatility = 0.06
                else:  # high risk
                    avg_monthly_return = 0.013
                    volatility = 0.10
                
                sharpe_ratio = avg_monthly_return / (volatility + 1e-8)
                model_adjustment = (normalized_action - 0.5) * 0.02
                expected_monthly_return = avg_monthly_return + model_adjustment
                
                if risk_level == 'low':
                    expected_monthly_return = max(0.005, expected_monthly_return)
                
                # Apply bounds
                if risk_level == 'low':
                    expected_monthly_return = np.clip(expected_monthly_return, 0.002, 0.026)
                elif risk_level == 'medium':
                    expected_monthly_return = np.clip(expected_monthly_return, -0.002, 0.024)
                else:
                    expected_monthly_return = np.clip(expected_monthly_return, -0.004, 0.035)
                
                expected_total_return = expected_monthly_return * months
                expected_profit = amount * expected_total_return
                confidence = min(0.7, max(0.3, abs(normalized_action - 0.5) * 1.5 + 0.3))
            
            # Add to results
            company_name = ticker_to_company.get(ticker, ticker)
            # Ensure we don't return ticker as company name
            if company_name == ticker:
                # Try to get from fallback companies
                fallback_companies = {
                    'AAPL': 'Apple Inc.',
                    'MSFT': 'Microsoft Corporation',
                    'GOOG': 'Alphabet Inc.',
                    'AMZN': 'Amazon.com Inc.',
                    'META': 'Meta Platforms Inc.',
                    'NVDA': 'NVIDIA Corporation',
                    'TSLA': 'Tesla Inc.',
                    'NFLX': 'Netflix Inc.',
                    'CRM': 'Salesforce Inc.',
                    'ADBE': 'Adobe Inc.',
                    'BAC': 'Bank of America Corp.',
                    'C': 'Citigroup Inc.',
                    'BLK': 'BlackRock Inc.',
                    'AXP': 'American Express Co.',
                    'COF': 'Capital One Financial Corp.',
                    'ABBV': 'AbbVie Inc.',
                    'ABT': 'Abbott Laboratories',
                    'BMY': 'Bristol-Myers Squibb Co.',
                    'AMGN': 'Amgen Inc.',
                    'COST': 'Costco Wholesale Corp.',
                    'AMD': 'Advanced Micro Devices Inc.',
                    'AVGO': 'Broadcom Inc.'
                }
                company_name = fallback_companies.get(ticker, f"{ticker} Corporation")
            
            results.append({
                'ticker': ticker,
                'company': company_name,
                'action': normalized_action,
                'expected_monthly_return': expected_monthly_return,
                'expected_total_return': expected_total_return,
                'expected_profit': expected_profit,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'confidence': confidence
            })
            
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            continue
    
    return results

# Import required modules
from ddpg_agent import DDPGAgent
from db_config import get_connection, return_connection

class ActionRequest(BaseModel):
    ticker: str
    risk_level: str
    capital: float

class ChatRequest(BaseModel):
    message: str
    risk_level: str
    capital: float

class PortfolioRequest(BaseModel):
    amount: float
    risk_level: str
    months: int

class PortfolioSuggestion(BaseModel):
    ticker: str
    company: str
    allocation_percentage: float
    allocation_amount: float
    expected_return_percentage: float
    expected_profit: float
    risk_score: float
    confidence: float

class PortfolioResponse(BaseModel):
    total_investment: float
    risk_level: str
    investment_period_months: int
    expected_total_return: float
    expected_total_profit: float
    portfolio_suggestions: List[PortfolioSuggestion]
    risk_analysis: Dict[str, float]
    investment_strategy: str

@app.get("/tickers")
def tickers():
    """Get all available tickers."""
    return {"tickers": get_all_tickers()}

@app.get("/companies")
def companies():
    """Get all available companies."""
    try:
        companies_list = get_all_companies()
        return companies_list
    except Exception as e:
        print(f"❌ Companies endpoint error: {e}")
        # Return fallback companies on any error
        return [
            "Apple Inc.", "Microsoft Corporation", "Alphabet Inc.", "Amazon.com Inc.",
            "Meta Platforms Inc.", "NVIDIA Corporation", "Tesla Inc.", "Netflix Inc.",
            "Salesforce Inc.", "Adobe Inc.", "Bank of America Corp.", "Citigroup Inc.",
            "BlackRock Inc.", "American Express Co.", "Capital One Financial Corp.",
            "AbbVie Inc.", "Abbott Laboratories", "Bristol-Myers Squibb Co.",
            "Amgen Inc.", "Costco Wholesale Corp.", "Advanced Micro Devices Inc.",
            "Broadcom Inc."
        ]

@app.get("/health")
def health_check():
    """Health check endpoint to test database connection."""
    from db_config import check_connection_health
    
    try:
        # Test database connection
        db_healthy = check_connection_health()
        
        # Test basic queries
        conn = get_connection()
        ticker_count = 0
        company_count = 0
        
        if conn:
            try:
                # Test tickers table
                df_tickers = pd.read_sql("SELECT COUNT(*) as count FROM tickers", conn)
                ticker_count = df_tickers.iloc[0]['count']
                
                # Test companies
                df_companies = pd.read_sql("SELECT COUNT(DISTINCT company) as count FROM tickers", conn)
                company_count = df_companies.iloc[0]['count']
                
            except Exception as e:
                print(f"❌ Health check query error: {e}")
            finally:
                return_connection(conn)
        
        return {
            "status": "healthy" if db_healthy else "unhealthy",
            "database_connection": "connected" if db_healthy else "disconnected",
            "tickers_count": ticker_count,
            "companies_count": company_count,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/find_ticker")
def find_ticker(q: str = Query(...)):
    """Find ticker by company name."""
    return {"ticker": find_ticker_by_company(q)}

@app.get("/suggestion")
def suggestion(
    ticker: str = Query(..., description="Stock ticker symbol"),
    risk_level: str = Query(..., description="Risk level: low, medium, high")
):
    """Get trading suggestions for a ticker and risk level."""
    normalized_risk = normalize_risk_level(risk_level)
    return {"suggestions": get_trading_suggestions(ticker, normalized_risk)}

@app.post("/action")
def action(req: ActionRequest):
    """Get trading action for a ticker."""
    normalized_risk = normalize_risk_level(req.risk_level)
    return {"action": get_trading_action(req.ticker, normalized_risk, req.capital)}

@app.get("/top-tickers")
def top_tickers(
    risk_level: str = Query(..., description="Risk level: low, medium, high"),
    capital: float = Query(..., description="Available capital")
):
    """Get top performing tickers for a risk level."""
    normalized_risk = normalize_risk_level(risk_level)
    return {"top_tickers": get_top_tickers(normalized_risk, capital)}

@app.get("/bottom-tickers")
def bottom_tickers(
    risk_level: str = Query(..., description="Risk level: low, medium, high"),
    capital: float = Query(..., description="Available capital")
):
    """Get bottom performing tickers for a risk level."""
    normalized_risk = normalize_risk_level(risk_level)
    return {"bottom_tickers": get_bottom_tickers(normalized_risk, capital)}

@app.post("/chat")
def chat(req: ChatRequest):
    """Chatbot endpoint: ask for buy/sell/hold or top/bottom tickers."""
    normalized_risk = normalize_risk_level(req.risk_level)
    return chatbot_response(req.message, normalized_risk, req.capital)

@app.get("/prediction")
def get_prediction(
    ticker: str = Query(..., description="Stock ticker symbol"),
    date: str = Query(..., description="Date in YYYY-MM-DD format")
):
    """
    Get the LSTM prediction for a ticker on a specific date.
    """
    conn = get_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed.")
    try:
        sql = """
            SELECT date, prediction, risk_level
            FROM lstm_predictions
            WHERE ticker = %s AND date = %s
            ORDER BY date DESC LIMIT 1
        """
        params = (ticker, date)
        df = pd.read_sql(sql, conn, params=params)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No prediction found for {ticker} on {date}.")
        result = df.iloc[0].to_dict()
        return {"prediction": result}
    finally:
        return_connection(conn)

@app.post("/portfolio-suggestion")
def portfolio_suggestion(req: PortfolioRequest):
    """Get portfolio suggestions based on amount, risk level, and investment period."""
    result = get_portfolio_suggestions(req.amount, req.risk_level, req.months)
    
    if "error" in result:
        return result
    
    # Create simplified response
    simplified_suggestions = []
    for suggestion in result.portfolio_suggestions:
        simplified_suggestions.append({
            "ticker": suggestion.ticker,
            "company": suggestion.company,
            "expected_return": f"{suggestion.expected_return_percentage:.1f}%",
            "expected_profit": f"${suggestion.expected_profit:,.0f}"
        })
    
    return {
        "summary": {
            "total_investment": f"${result.total_investment:,.0f}",
            "risk_level": result.risk_level.capitalize(),
            "period": f"{result.investment_period_months} months",
            "expected_return": f"{result.expected_total_return:.1f}%",
            "expected_profit": f"${result.expected_total_profit:,.0f}"
        },
        "stocks": simplified_suggestions,
        "strategy": result.investment_strategy
    }



def get_portfolio_suggestions(amount: float, risk_level: str, months: int):
    """Generate portfolio suggestions with profit predictions using optimized batch processing."""
    
    # Validate inputs
    if amount <= 0:
        return {"error": "Amount must be positive"}
    if risk_level not in ['low', 'medium', 'high']:
        return {"error": "Risk level must be low, medium, or high"}
    if months <= 0 or months > 60:
        return {"error": "Investment period must be between 1 and 60 months"}
    
    # Get available tickers and models
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'models'))
    available_tickers = []
    
    # Check which tickers have models for the given risk level
    for file in os.listdir(model_dir):
        if file.endswith(f'_ddpg_actor_{risk_level}.pth'):
            ticker = file.split('_')[0]
            available_tickers.append(ticker)
    
    # Sort tickers alphabetically to get better distribution
    available_tickers.sort()
    
    print(f"Found {len(available_tickers)} available tickers for risk level {risk_level}")
    print(f"Processing all {len(available_tickers)} tickers with optimized batch processing...")
    
    if not available_tickers:
        return {"error": f"No models available for risk level: {risk_level}"}
    
    # Batch fetch company names and historical data
    ticker_to_company = get_batch_company_names(available_tickers)
    historical_data = get_batch_historical_data(available_tickers)
    
    # Process tickers in parallel batches
    ticker_analysis = []
    batch_size = 20  # Process 20 tickers at a time
    batches = [available_tickers[i:i + batch_size] for i in range(0, len(available_tickers), batch_size)]
    
    print(f"Processing {len(batches)} batches of {batch_size} tickers each...")
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Submit batch processing tasks
        future_to_batch = {
            executor.submit(process_ticker_batch_optimized, batch, risk_level, amount, months, historical_data, ticker_to_company): batch 
            for batch in batches
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_batch):
            batch_results = future.result()
            ticker_analysis.extend(batch_results)
            print(f"Completed batch with {len(batch_results)} results")
    
    print(f"Total processed tickers: {len(ticker_analysis)}")
    
    if not ticker_analysis:
        return {"error": "No valid ticker analysis available"}
    
    # Sort by expected return and add some randomization for variety
    ticker_analysis.sort(key=lambda x: x['expected_total_return'], reverse=True)
    
    # Select top performers based on risk level
    if risk_level == 'low':
        # For low risk: prioritize positive returns, low volatility, and stable returns
        ticker_analysis.sort(key=lambda x: (x['expected_total_return'] > 0, x['sharpe_ratio'], -x['volatility'], x['expected_total_return']), reverse=True)
    elif risk_level == 'medium':
        # For medium risk: balance between return and risk
        ticker_analysis.sort(key=lambda x: (x['expected_total_return'], x['sharpe_ratio']), reverse=True)
    else:  # high risk
        # For high risk: prioritize high returns, accept higher volatility
        ticker_analysis.sort(key=lambda x: (x['expected_total_return'], x['volatility']), reverse=True)
    
    top_tickers = ticker_analysis[:5]  # Top 5 performers for the risk level
    
    # Calculate portfolio allocation
    total_expected_return = sum(t['expected_total_return'] for t in top_tickers)
    portfolio_suggestions = []
    
    # Calculate portfolio allocation based on risk level
    portfolio_suggestions = []
    
    if risk_level == 'low':
        # Conservative: equal allocation for stability
        equal_weight = 1.0 / len(top_tickers)
        for i, ticker_data in enumerate(top_tickers):
            allocation_amount = amount * equal_weight
            expected_profit = allocation_amount * ticker_data['expected_total_return']
            
            portfolio_suggestions.append(PortfolioSuggestion(
                ticker=ticker_data['ticker'],
                company=ticker_data['company'],
                allocation_percentage=equal_weight * 100,
                allocation_amount=allocation_amount,
                expected_return_percentage=ticker_data['expected_total_return'] * 100,
                expected_profit=expected_profit,
                risk_score=ticker_data['volatility'],
                confidence=ticker_data['confidence']
            ))
    elif risk_level == 'medium':
        # Balanced: weight by expected return but cap at 30%
        total_score = sum(t['expected_total_return'] * t['sharpe_ratio'] for t in top_tickers)
        weights = []
        for ticker_data in top_tickers:
            weight = (ticker_data['expected_total_return'] * ticker_data['sharpe_ratio']) / (total_score + 1e-8)
            weight = min(0.3, max(0.1, weight))  # Between 10% and 30%
            weights.append(weight)
        
        # Normalize weights to sum to 1
        total_weight = sum(weights)
        for i, ticker_data in enumerate(top_tickers):
            normalized_weight = weights[i] / total_weight
            allocation_amount = amount * normalized_weight
            expected_profit = allocation_amount * ticker_data['expected_total_return']
            
            portfolio_suggestions.append(PortfolioSuggestion(
                ticker=ticker_data['ticker'],
                company=ticker_data['company'],
                allocation_percentage=normalized_weight * 100,
                allocation_amount=allocation_amount,
                expected_return_percentage=ticker_data['expected_total_return'] * 100,
                expected_profit=expected_profit,
                risk_score=ticker_data['volatility'],
                confidence=ticker_data['confidence']
            ))
    else:  # high risk
        # Aggressive: weight by expected return, allow up to 40%
        total_score = sum(t['expected_total_return'] for t in top_tickers)
        weights = []
        for ticker_data in top_tickers:
            weight = ticker_data['expected_total_return'] / (total_score + 1e-8)
            weight = min(0.4, max(0.1, weight))  # Between 10% and 40%
            weights.append(weight)
        
        # Normalize weights to sum to 1
        total_weight = sum(weights)
        for i, ticker_data in enumerate(top_tickers):
            normalized_weight = weights[i] / total_weight
            allocation_amount = amount * normalized_weight
            expected_profit = allocation_amount * ticker_data['expected_total_return']
            
            portfolio_suggestions.append(PortfolioSuggestion(
                ticker=ticker_data['ticker'],
                company=ticker_data['company'],
                allocation_percentage=normalized_weight * 100,
                allocation_amount=allocation_amount,
                expected_return_percentage=ticker_data['expected_total_return'] * 100,
                expected_profit=expected_profit,
                risk_score=ticker_data['volatility'],
                confidence=ticker_data['confidence']
            ))
    
    # Calculate portfolio-level metrics correctly
    total_allocation = sum(s.allocation_amount for s in portfolio_suggestions)
    total_expected_profit = sum(s.expected_profit for s in portfolio_suggestions)
    weighted_return = total_expected_profit / amount if amount > 0 else 0
    
    # Ensure reasonable bounds for portfolio metrics
    weighted_return = np.clip(weighted_return, -0.5, 0.5)  # Between -50% and +50%
    total_expected_profit = np.clip(total_expected_profit, -amount * 0.5, amount * 0.5)
    
    # Risk analysis
    portfolio_volatility = np.mean([s.risk_score for s in portfolio_suggestions])
    portfolio_sharpe = weighted_return / (portfolio_volatility + 1e-8)
    
    risk_analysis = {
        "portfolio_volatility": portfolio_volatility,
        "portfolio_sharpe_ratio": portfolio_sharpe,
        "max_drawdown_estimate": portfolio_volatility * np.sqrt(months),
        "var_95_percentile": portfolio_volatility * 1.645 * np.sqrt(months)
    }
    
    # Investment strategy description
    if risk_level == 'low':
        strategy = "Conservative portfolio focused on stable, dividend-paying stocks with lower volatility"
    elif risk_level == 'medium':
        strategy = "Balanced portfolio with mix of growth and value stocks, moderate risk tolerance"
    else:
        strategy = "Aggressive portfolio targeting high-growth stocks with higher potential returns"
    
    return PortfolioResponse(
        total_investment=amount,
        risk_level=risk_level,
        investment_period_months=months,
        expected_total_return=weighted_return * 100,
        expected_total_profit=total_expected_profit,
        portfolio_suggestions=portfolio_suggestions,
        risk_analysis=risk_analysis,
        investment_strategy=strategy
    )

# Helper functions (these would need to be implemented or imported)
def get_all_tickers():
    """Get all available tickers."""
    conn = get_connection()
    if not conn:
        return []
    try:
        df = pd.read_sql("SELECT DISTINCT ticker FROM stock_data ORDER BY ticker", conn)
        return df['ticker'].tolist()
    finally:
        return_connection(conn)

def get_all_companies():
    """Get all available companies."""
    conn = get_connection()
    if not conn:
        print("❌ No database connection available for get_all_companies")
        # Return fallback companies if connection fails
        return [
            "Apple Inc.", "Microsoft Corporation", "Alphabet Inc.", "Amazon.com Inc.",
            "Meta Platforms Inc.", "NVIDIA Corporation", "Tesla Inc.", "Netflix Inc.",
            "Salesforce Inc.", "Adobe Inc.", "Bank of America Corp.", "Citigroup Inc.",
            "BlackRock Inc.", "American Express Co.", "Capital One Financial Corp.",
            "AbbVie Inc.", "Abbott Laboratories", "Bristol-Myers Squibb Co.",
            "Amgen Inc.", "Costco Wholesale Corp.", "Advanced Micro Devices Inc.",
            "Broadcom Inc."
        ]
    try:
        # Get companies directly with one efficient query
        df = pd.read_sql("SELECT DISTINCT company FROM tickers WHERE company IS NOT NULL AND company != '' ORDER BY company", conn)
        companies = df['company'].tolist()
        print(f"✅ get_all_companies: Found {len(companies)} companies")
        return companies
    except Exception as e:
        print(f"❌ Error in get_all_companies: {e}")
        # Return fallback companies if query fails
        return [
            "Apple Inc.", "Microsoft Corporation", "Alphabet Inc.", "Amazon.com Inc.",
            "Meta Platforms Inc.", "NVIDIA Corporation", "Tesla Inc.", "Netflix Inc.",
            "Salesforce Inc.", "Adobe Inc.", "Bank of America Corp.", "Citigroup Inc.",
            "BlackRock Inc.", "American Express Co.", "Capital One Financial Corp.",
            "AbbVie Inc.", "Abbott Laboratories", "Bristol-Myers Squibb Co.",
            "Amgen Inc.", "Costco Wholesale Corp.", "Advanced Micro Devices Inc.",
            "Broadcom Inc."
        ]
    finally:
        return_connection(conn)

def find_ticker_by_company(company_name):
    """Find ticker by company name."""
    conn = get_connection()
    if not conn:
        return None
    try:
        sql = "SELECT ticker FROM tickers WHERE LOWER(company) LIKE LOWER(%s) LIMIT 1"
        df = pd.read_sql(sql, conn, params=(f'%{company_name}%',))
        return df['ticker'].iloc[0] if not df.empty else None
    finally:
        return_connection(conn)

def get_trading_suggestions(ticker, risk_level):
    """Get trading suggestions for a ticker."""
    # Implementation would go here
    return f"Trading suggestions for {ticker} with {risk_level} risk"

def get_trading_action(ticker, risk_level, capital):
    """Get trading action for a ticker."""
    # Implementation would go here
    return "Hold"

def get_top_tickers(risk_level, capital):
    """Get top performing tickers."""
    # Implementation would go here
    return ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

def get_bottom_tickers(risk_level, capital):
    """Get bottom performing tickers."""
    # Implementation would go here
    return ["TICKER1", "TICKER2", "TICKER3", "TICKER4", "TICKER5"]

def chatbot_response(message, risk_level, capital):
    """Chatbot response handler."""
    # Implementation would go here
    return {"response": f"Chatbot response for {message} with {risk_level} risk and {capital} capital"} 
