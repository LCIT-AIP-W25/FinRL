
import numpy as np

def calculate_risk_metrics(stock_data, model, env):
    """
    Calculate risk metrics like Sharpe ratio, volatility, drawdown, etc.
    """
    portfolio_value = model.predict(env)
    
    # Calculate returns and volatility
    returns = np.diff(portfolio_value) / portfolio_value[:-1]
    volatility = np.std(returns)
    sharpe_ratio = np.mean(returns) / volatility
    
    # Calculate drawdown
    drawdown = calculate_drawdown(portfolio_value)
    
    return {
        "sharpe_ratio": sharpe_ratio,
        "volatility": volatility,
        "drawdown": drawdown,
    }

def calculate_drawdown(portfolio_value):
    """
    Calculate maximum drawdown of the portfolio.
    """
    running_max = np.maximum.accumulate(portfolio_value)
    drawdown = (portfolio_value - running_max) / running_max
    return np.min(drawdown)
