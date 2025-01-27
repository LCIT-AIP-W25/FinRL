
# FinRL

## Overview

The **Stock Market Simulator** is a command-line Python application that allows users to simulate buying stocks, managing portfolios, and viewing stock prices using live data from Yahoo Finance (`yfinance`). 

Future updates will integrate **FinRL** for algorithmic trading, a **MongoDB** backend for storing user data, and a **React** or **HTML/JS** frontend for an interactive user experience.

## Features

- **Sign Up / Login**: Create an account or log in securely.
- **Buy Stocks**: Purchase stocks with a starting balance of $10,000. Stock prices are fetched live from Yahoo Finance.
- **View Portfolio**: View your portfolio and remaining balance.

## Requirements

- Python 3.x
- `yfinance` library

Install dependencies using:

```bash
pip install yfinance
```

## Running the Application

1. Clone or download the repo.
2. Navigate to the folder with the Python script.
3. Run:

   ```bash
   python stock_market_simulator.py
   ```

## Usage

1. **Sign Up / Login**: Sign up with a unique username or log in with your credentials.
2. **Buy Stocks**: Select stocks (AAPL, GOOGL, AMZN, TSLA) and buy based on available funds.
3. **View Portfolio**: Check your stocks and remaining balance.
4. **Logout**: Log out anytime.

## Future Updates

- **FinRL Integration**: Algorithmic trading and stock data analysis.
- **MongoDB Backend**: Storing user data and transactions.
- **Frontend**: React or HTML/JS for an interactive web interface.

## Troubleshooting

- **Stock data not loading**: Ensure an active internet connection and that `yfinance` is installed.
- **Invalid login**: Double-check your credentials.

## License

Open-source under the [MIT License](LICENSE).
```

This version keeps the essential information while being more concise.