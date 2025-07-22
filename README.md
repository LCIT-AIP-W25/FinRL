# 🤖 FinRL AI Trading Bot

A sophisticated AI-powered trading bot that combines Deep Deterministic Policy Gradient (DDPG) reinforcement learning with LSTM neural networks to provide intelligent stock trading recommendations and interactive chatbot functionality.

## 🚀 Features

- **AI-Powered Trading Recommendations**: Uses DDPG agents trained for different risk levels (low, medium, high)
- **LSTM Price Prediction**: Neural network models for stock price forecasting
- **Interactive Chatbot**: Natural language interface for trading queries with Groq API integration
- **Multi-Risk Level Support**: Personalized recommendations based on risk tolerance
- **Real-time Stock Data**: Integration with financial data sources and sentiment analysis
- **RESTful API**: FastAPI-based backend for easy integration
- **Sentiment Analysis**: News sentiment integration for enhanced predictions
- **Fuzzy Matching**: Intelligent ticker and company name recognition

## 📊 Supported Stocks

The bot supports 500+ S&P 500 stocks including:
- **Tech**: AAPL, MSFT, GOOG, META, NVDA, TSLA, AMZN, NFLX
- **Finance**: JPM, BAC, WFC, GS, AXP
- **Healthcare**: JNJ, PFE, UNH, ABBV
- **And many more...**

## 🛠️ Technology Stack

### Backend
- **Framework**: FastAPI, Python 3.8+
- **Machine Learning**: PyTorch, TensorFlow 2.10, Scikit-learn
- **Reinforcement Learning**: Custom DDPG implementation
- **Data Processing**: Pandas, NumPy
- **Database**: PostgreSQL
- **AI Chat**: Groq API (Llama3-70B)

### Frontend (In Development)
- **Framework**: React 19
- **Styling**: Styled Components
- **Routing**: React Router DOM
- **Icons**: FontAwesome

## 📋 Prerequisites

- Python 3.8 or higher
- PostgreSQL database
- Groq API key (for chatbot functionality)
- Node.js (for frontend development)

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd FinRL
```

### 2. Install Backend Dependencies

```bash
cd backendnew
pip install -r requirements.txt
```

### 3. Environment Setup

1. Create a `.env` file in the `backendnew` directory:
```bash
# Groq API Configuration
GROQ_API_KEY=your_groq_api_key_here

# Database Configuration (if needed)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=your_database_name
DB_USER=your_username
DB_PASSWORD=your_password
```

### 4. Database Setup

- PostgreSQL database is required
- Ensure proper database credentials in `db_config.py`
- Run database initialization scripts before first use
- Required tables: `tickers`, `lstm_predictions`, `stock_data`, `finance_news`


### 5. Start the API Server

```bash
cd backendnew
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at: `http://localhost:8000`

### 6. (Optional) Start Frontend Development

```bash
cd backendnew/stockTradingbotUI
npm install
npm start
```

## 📖 API Documentation

### Interactive Documentation
Visit `http://localhost:8000/docs` for interactive API documentation.

### Key Endpoints

#### Get Available Data

```http
GET /tickers                    # Get all available tickers
GET /companies                  # Get all available companies
GET /find_ticker?q=AAPL        # Find ticker by name/company
```

#### Trading Recommendations
```http
POST /action
Content-Type: application/json

{
  "ticker": "AAPL",
  "risk_level": "medium",
  "capital": 10000
}
```

#### Trading Suggestions (Historical)
```http
GET /suggestion?ticker=AAPL&risk_level=medium
```

#### Top/Bottom Tickers
```http
GET /top-tickers?risk_level=medium&capital=10000
GET /bottom-tickers?risk_level=medium&capital=10000
```


#### Advanced Chatbot (with Groq API)
```http
POST /chatbot
Content-Type: application/json

{
  "message": "What's the correlation between AAPL and MSFT?",
  "risk_level": "medium",
  "capital": 10000
}
```


#### Trading Recommendations
```http
POST /action
Content-Type: application/json

{
  "ticker": "AAPL",
  "risk_level": "medium",
  "capital": 10000
}
```

#### Trading Suggestions (Historical)
```http
GET /suggestion?ticker=AAPL&risk_level=medium
```

#### Top/Bottom Tickers
```http
GET /top-tickers?risk_level=medium&capital=10000
GET /bottom-tickers?risk_level=medium&capital=10000
```

#### Chatbot
```http
POST /chat
Content-Type: application/json

{
  "message": "Should I buy AAPL?",
  "risk_level": "medium",
  "capital": 10000
}
```

#### Advanced Chatbot (with Groq API)
```http
POST /chatbot
Content-Type: application/json

{
  "message": "What's the correlation between AAPL and MSFT?",
  "risk_level": "medium",
  "capital": 10000
}
```

## 💬 Chatbot Examples

The chatbot supports various types of queries with intelligent intent parsing:

### Trading Recommendations
- "Should I buy AAPL?"
- "What's your recommendation for TSLA?"
- "Should I sell MSFT?"
- "Hold recommendation for GOOG"

### Market Analysis
- "Show me the top 3 buy signals"
- "What are the most volatile stocks?"
- "Get bottom 3 sell signals"
- "Top gainers today"

### Stock Information
- "What's the current price of GOOG?"
- "Show me the RSI for NVDA"
- "What's the correlation between AAPL and MSFT?"
- "Market cap of TSLA"

### Historical Data
- "Show me last 5 recommendations for TSLA"
- "What's the volatility of META?"
- "Highest price of AAPL in the last 30 days"

### Technical Analysis
- "SMA for MSFT"
- "Moving average for GOOG"
- "Trend analysis for NVDA"
- "Sharpe ratio of AAPL"

## 🧠 AI Models

### DDPG Agents
- **Purpose**: Generate buy/sell/hold actions based on market state
- **Architecture**: Actor-Critic networks with target networks
- **Training**: Reinforcement learning with historical stock data
- **Risk Levels**: Separate models for low, medium, and high risk tolerance
- **Output**: Continuous action values (-1 to 1) representing sell to buy signals
- **Features**: State includes LSTM predictions and risk level encoding

### LSTM Networks
- **Purpose**: Predict stock prices and market trends
- **Architecture**: 2-layer LSTM with dropout and dense layers
- **Features**: Historical price data, technical indicators (SMA, RSI, Volume)
- **Output**: 10-day price predictions
- **Training**: Uses MinMaxScaler for data normalization

## 📈 Model Training

### Training DDPG Agents
```bash
cd backendnew
python train_ddpg.py --ticker AAPL --risk_level medium
```

### Training LSTM Models
```bash
cd backendnew
python train_lstm_from_news_tickers.py
```

### Data Preprocessing
```bash
cd backendnew
python data_preprocessing.py
python lstm_data_prep.py
```

### Resume Training
```bash
cd backendnew
python resume_lstm_training.py
```

## 🔧 Configuration

### Risk Levels
- **Low**: Conservative approach, minimal risk (risk_level = 0)
- **Medium**: Balanced risk-reward strategy (risk_level = 1)
- **High**: Aggressive approach, higher potential returns (risk_level = 2)


### Model Files
- Trained models are large files and may not be included in the repository
- DDPG models are stored in the `models/` directory with naming pattern: `{TICKER}_ddpg_actor_{RISK_LEVEL}.pth`
- LSTM models are stored in `backendnew/models/` with naming pattern: `{TICKER}_lstm_model.h5` and `{TICKER}_scaler.pkl`

## 📁 Project Structure

```
FinRL/
├── backendnew/                    # Main backend application
│   ├── api.py                    # FastAPI server and endpoints
│   ├── api_utils.py              # API utility functions and chatbot logic
│   ├── ddpg_agent.py             # DDPG agent implementation
│   ├── lstm_model.py             # LSTM model implementation
│   ├── data_preprocessing.py     # Data preprocessing scripts
│   ├── train_ddpg.py             # DDPG training script
│   ├── sentiment_analysis.py     # News sentiment analysis
│   ├── db_config.py              # Database configuration
│   ├── requirements.txt          # Python dependencies
│   ├── models/                   # Trained LSTM models and scalers
│   └── stockTradingbotUI/        # React frontend (in development)
│       ├── package.json          # Frontend dependencies
│       └── node_modules/         # Frontend packages
├── models/                       # Trained DDPG models
│   ├── AAPL_ddpg_actor_low.pth
│   ├── AAPL_ddpg_actor_medium.pth
│   ├── AAPL_ddpg_actor_high.pth
│   └── ... (for all tickers)
├── data/                         # Data storage
├── .env                          # Environment variables (not in git)
├── .gitignore                    # Git ignore rules
└── README.md                     # This file

## 🚨 Important Notes

### Security
- **Never commit your `.env` file** - it contains sensitive API keys
- The `.env` file should be automatically ignored by git
- Use environment variables for all sensitive configuration

### Model Files
- Trained models are large files and may not be included in the repository
- DDPG models are stored in the `models/` directory with naming pattern: `{TICKER}_ddpg_actor_{RISK_LEVEL}.pth`
- LSTM models are stored in `backendnew/models/` with naming pattern: `{TICKER}_lstm_model.h5` and `{TICKER}_scaler.pkl`

### Database Requirements
- PostgreSQL database is required
- Ensure proper database credentials in `db_config.py`
- Run database initialization scripts before first use
- Required tables: `tickers`, `lstm_predictions`, `stock_data`, `finance_news`

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

**This software is for educational and research purposes only. It is not intended to provide financial advice. Always do your own research and consult with financial professionals before making investment decisions.**

The authors are not responsible for any financial losses incurred through the use of this software.

## 🆘 Support

If you encounter any issues:

1. Check the [Issues](https://github.com/your-repo/issues) page
2. Ensure all dependencies are installed correctly
3. Verify your environment variables are set properly
4. Check that your database is running and accessible
5. Ensure you have sufficient data for model training

## 🔄 Development Status

- ✅ Backend API (FastAPI) - Complete
- ✅ DDPG Agent Implementation - Complete
- ✅ LSTM Model Implementation - Complete
- ✅ Chatbot with Intent Parsing - Complete
- ✅ Database Integration - Complete
- 🔄 Frontend UI (React) - In Development
- 🔄 Docker Containerization - Planned
- 🔄 Model Performance Optimization - Ongoing

## 📊 Performance Metrics

The system provides:
- Real-time trading signals based on AI models
- Historical performance tracking
- Risk-adjusted recommendations
- Multi-timeframe analysis
- Sentiment-enhanced predictions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Contact

For questions or support, please open an issue on the GitHub repository. 
