# Stock Market Predictor

A machine learning-powered stock market prediction tool that analyzes tech stocks and provides daily predictions.

## Features

### 🚀 Daily Top Tech Stocks
- Identifies the top 10 tech stocks most likely to rise each day
- Uses advanced ML models to analyze market patterns
- Updates predictions daily based on latest market data

### 🔍 Individual Stock Analysis
- Search any stock by ticker symbol (e.g., AAPL, GOOGL, MSFT)
- Get detailed prediction analysis with confidence scores
- View historical performance and trend analysis

## Tech Stack

- **Backend**: Python with FastAPI
- **ML Models**: Scikit-learn, TensorFlow/Keras
- **Data Sources**: Yahoo Finance, Alpha Vantage API
- **Frontend**: React.js with TypeScript
- **Database**: SQLite for caching and user data
- **Deployment**: Docker containerization

## Project Structure

```
Market-Predictor/
├── backend/                 # FastAPI backend
│   ├── app/
│   │   ├── models/         # ML prediction models
│   │   ├── data/           # Data collection & processing
│   │   ├── api/            # API endpoints
│   │   └── utils/          # Utility functions
│   ├── requirements.txt
│   └── main.py
├── frontend/               # React frontend
│   ├── src/
│   │   ├── components/     # UI components
│   │   ├── pages/          # Page components
│   │   └── services/       # API services
│   └── package.json
├── data/                   # Data storage
├── models/                 # Trained ML models
└── docker-compose.yml      # Container orchestration
```

## Getting Started

1. Clone the repository
2. Set up environment variables for API keys
3. Run `docker-compose up` to start the application
4. Access the web interface at `http://localhost:3000`

## API Endpoints

- `GET /api/top-stocks` - Get daily top tech stocks
- `GET /api/stock/{ticker}` - Get prediction for specific stock
- `GET /api/stock/{ticker}/history` - Get historical data

## License

MIT License - see LICENSE file for details