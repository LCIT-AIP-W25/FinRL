# 🤖 AI Trading Bot with FinRL

A sophisticated AI-powered trading bot that combines Deep Deterministic Policy Gradient (DDPG) reinforcement learning with LSTM neural networks to provide intelligent stock trading recommendations and chatbot functionality.

## 🚀 Features

- **AI-Powered Trading Recommendations**: Uses DDPG agents trained for different risk levels (low, medium, high)
- **LSTM Price Prediction**: Neural network models for stock price forecasting
- **Interactive Chatbot**: Natural language interface for trading queries
- **Multi-Risk Level Support**: Personalized recommendations based on risk tolerance
- **Real-time Stock Data**: Integration with financial data sources
- **RESTful API**: FastAPI-based backend for easy integration
- **Sentiment Analysis**: News sentiment integration for enhanced predictions

## 📊 Supported Stocks

The bot supports 500+ S&P 500 stocks including:
- **Tech**: AAPL, MSFT, GOOG, META, NVDA, TSLA, AMZN
- **Finance**: JPM, BAC, WFC, GS
- **Healthcare**: JNJ, PFE, UNH
- **And many more...**

## 🛠️ Technology Stack

- **Backend**: FastAPI, Python 3.8+
- **Machine Learning**: PyTorch, TensorFlow, Scikit-learn
- **Reinforcement Learning**: FinRL, Stable-Baselines3
- **Data Processing**: Pandas, NumPy
- **Database**: PostgreSQL
- **AI Chat**: Groq API (Llama3-70B)


## 📋 Prerequisites

- Python 3.8 or higher
- PostgreSQL database
- Groq API key (for chatbot functionality)

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd BACKENDNNEW
```

### 2. Install Dependencies

```bash
cd FinRL
pip install -r requirements.txt
```

### 3. Environment Setup

1. Copy the environment template:
```bash
cp .env.example .env
```

2. Edit `.env` and add your API keys:
```env
# Groq API Configuration
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Database Setup

1. Create a PostgreSQL database
2. Update `db_config.py` with your database credentials
3. Run the database initialization scripts

### 5. Start the API Server

```bash
cd FinRL
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at: `http://localhost:8000`

## 📖 API Documentation

### Interactive Documentation
Visit `http://localhost:8000/docs` for interactive API documentation.

### Key Endpoints

#### Chatbot
```http
POST /chatbot
Content-Type: application/json

{
  "message": "Should I buy AAPL?",
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

#### Top/Bottom Tickers
```http
GET /top-tickers?risk_level=medium&capital=10000
GET /bottom-tickers?risk_level=medium&capital=10000
```

## 💬 Chatbot Examples

The chatbot supports various types of queries:

### Trading Recommendations
- "Should I buy AAPL?"
- "What's your recommendation for TSLA?"
- "Should I sell MSFT?"

### Market Analysis
- "Show me the top 3 buy signals"
- "What are the most volatile stocks?"
- "Get bottom 3 sell signals"

### Stock Information
- "What's the current price of GOOG?"
- "Show me the RSI for NVDA"
- "What's the correlation between AAPL and MSFT?"

### Historical Data
- "Show me last 5 recommendations for TSLA"
- "What's the volatility of META?"

## 🧠 AI Models

### DDPG Agents
- **Purpose**: Generate buy/sell/hold actions based on market state
- **Training**: Reinforcement learning with historical stock data
- **Risk Levels**: Separate models for low, medium, and high risk tolerance
- **Output**: Continuous action values (-1 to 1) representing sell to buy signals

### LSTM Networks
- **Purpose**: Predict stock prices and market trends
- **Architecture**: Long Short-Term Memory neural networks
- **Features**: Historical price data, technical indicators, sentiment data
- **Output**: Price predictions and trend analysis

## 📈 Model Training

### Training DDPG Agents
```bash
python train_ddpg.py --ticker AAPL --risk_level medium
```

### Training LSTM Models
```bash
python train_lstm_from_news_tickers.py --ticker AAPL
```

### Data Preprocessing
```bash
python data_preprocessing.py
python lstm_data_prep.py
```

## 🔧 Configuration

### Risk Levels
- **Low**: Conservative approach, minimal risk
- **Medium**: Balanced risk-reward strategy
- **High**: Aggressive approach, higher potential returns

### Model Parameters
- **State Dimension**: 2 (prediction + risk level)
- **Action Dimension**: 1 (buy/sell signal)
- **Learning Rate**: 0.001
- **Batch Size**: 64
- **Memory Size**: 100000

## 📁 Project Structure

```
SCRAPING/
├── FinRL/
│   ├── api.py                 # FastAPI server
│   ├── api_utils.py           # API utility functions
│   ├── ddpg_agent.py          # DDPG agent implementation
│   ├── lstm_model.py          # LSTM model implementation
│   ├── data_preprocessing.py  # Data preprocessing scripts
│   ├── train_ddpg.py          # DDPG training script
│   ├── sentiment_analysis.py  # News sentiment analysis
│   ├── db_config.py           # Database configuration
│   ├── requirements.txt       # Python dependencies
│   └── models/                # Trained model files
├── models/                    # Additional model files
├── data/                      # Data storage
├── .env                       # Environment variables (not in git)
├── .env.example              # Environment template
├── .gitignore                # Git ignore rules
└── README.md                 # This file
```

## 🚨 Important Notes

### Security
- **Never commit your `.env` file** - it contains sensitive API keys
- The `.env` file is automatically ignored by git
- Use `.env.example` as a template for your own setup

### Model Files
- Trained models are large files and may not be included in the repository
- Models are stored in the `models/` directory
- Each stock has separate models for different risk levels

### Database Requirements
- PostgreSQL database is required
- Ensure proper database credentials in `db_config.py`
- Run database initialization scripts before first use


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

## 📊 Performance Metrics

The models are evaluated on:
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Worst peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Ratio of gross profit to gross loss

---

**Built with ❤️ using cutting-edge AI and machine learning technologies** 
