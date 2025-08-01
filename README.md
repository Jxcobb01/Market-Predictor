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
- **Frontend**: Angular 20 with TypeScript
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
├── frontend/               # Angular frontend
│   ├── src/app/
│   │   ├── components/     # UI components
│   │   ├── pages/          # Page components
│   │   ├── services/       # API services
│   │   └── app.routes.ts   # Routing configuration
│   ├── package.json
│   └── angular.json
├── data/                   # Data storage
├── models/                 # Trained ML models
└── docker-compose.yml      # Container orchestration
```

## Getting Started

### Prerequisites
- Node.js (v20.19.0 or higher)
- Python 3.9+
- Docker and Docker Compose (optional)

### Option 1: Quick Start with Docker
1. Clone the repository
2. Set up environment variables for API keys (if needed)
3. Run `docker-compose up --build`
4. Access the web interface at `http://localhost:3000`

### Option 2: Local Development

#### Backend Setup
```bash
cd backend
pip install -r requirements.txt
python main.py
```
Backend will be available at `http://localhost:8000`

#### Frontend Setup
```bash
cd frontend
npm install
npx ng serve
```
Frontend will be available at `http://localhost:4200`

### Option 3: Automated Setup
```bash
chmod +x setup.sh
./setup.sh
```

## API Endpoints

- `GET /api/top-stocks` - Get daily top tech stocks
- `GET /api/stock/{ticker}` - Get prediction for specific stock
- `GET /api/stock/{ticker}/history` - Get historical data
- `POST /api/model/train` - Train the prediction model
- `GET /api/model/info` - Get model information
- `GET /api/health` - Health check

## Frontend Pages

- **Home** (`/`) - Landing page with features overview
- **Top Stocks** (`/top-stocks`) - Daily top tech stock predictions
- **Stock Search** (`/search`) - Search individual stocks by ticker

## Development

### Angular Commands
```bash
# Generate new component
npx ng generate component components/my-component

# Generate new service
npx ng generate service services/my-service

# Build for production
npx ng build

# Run tests
npx ng test
```

### Backend Commands
```bash
# Train the model
curl -X POST http://localhost:8000/api/model/train

# Get top stocks
curl http://localhost:8000/api/top-stocks

# Search specific stock
curl http://localhost:8000/api/stock/AAPL
```

## Model Training

The ML model uses:
- **Gradient Boosting Classifier** for predictions
- **Technical Indicators**: RSI, MACD, Moving Averages, Bollinger Bands
- **Feature Engineering**: Price changes, volume analysis, volatility
- **Data Sources**: Yahoo Finance historical data

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Disclaimer

This application provides stock market predictions based on machine learning analysis of technical indicators. These predictions are for informational purposes only and should not be considered as financial advice. Always conduct your own research and consult with a financial advisor before making investment decisions. Past performance does not guarantee future results.