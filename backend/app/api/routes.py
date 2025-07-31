from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import logging
from datetime import datetime

from ..data.stock_data import StockDataCollector
from ..models.prediction_model import StockPredictionModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Stock Market Predictor API",
    description="AI-powered stock market prediction tool for tech stocks",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
data_collector = StockDataCollector()
prediction_model = StockPredictionModel()

# Pydantic models for request/response
class StockPrediction(BaseModel):
    ticker: str
    prediction: str
    confidence: float
    probability_rise: float
    probability_fall: float
    current_price: float
    technical_insights: Dict
    timestamp: str

class TopStocksResponse(BaseModel):
    predictions: List[StockPrediction]
    total_count: int
    generated_at: str

class StockSearchResponse(BaseModel):
    ticker: str
    prediction: str
    confidence: float
    probability_rise: float
    probability_fall: float
    current_price: float
    technical_insights: Dict
    timestamp: str
    company_info: Optional[Dict] = None

class ErrorResponse(BaseModel):
    error: str
    message: str
    timestamp: str

@app.get("/", response_model=Dict)
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Stock Market Predictor API",
        "version": "1.0.0",
        "endpoints": {
            "top_stocks": "/api/top-stocks",
            "stock_prediction": "/api/stock/{ticker}",
            "model_info": "/api/model/info",
            "health": "/api/health"
        }
    }

@app.get("/api/health", response_model=Dict)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": prediction_model.model is not None
    }

@app.get("/api/top-stocks", response_model=TopStocksResponse)
async def get_top_stocks():
    """
    Get the top 10 tech stocks most likely to rise today
    """
    try:
        logger.info("Fetching top stocks prediction")
        
        # Load model if not already loaded
        if prediction_model.model is None:
            prediction_model.load_model()
        
        if prediction_model.model is None:
            raise HTTPException(
                status_code=503,
                detail="Prediction model not available. Please train the model first."
            )
        
        # Get data for all tech stocks
        stock_data = data_collector.get_all_tech_stocks_data()
        
        if not stock_data:
            raise HTTPException(
                status_code=503,
                detail="Unable to fetch stock data"
            )
        
        # Get predictions
        predictions = prediction_model.predict_top_stocks(stock_data)
        
        # Convert to response format
        stock_predictions = []
        for pred in predictions:
            stock_predictions.append(StockPrediction(
                ticker=pred["ticker"],
                prediction=pred["prediction"],
                confidence=pred["confidence"],
                probability_rise=pred["probability_rise"],
                probability_fall=pred["probability_fall"],
                current_price=pred["current_price"],
                technical_insights=pred["technical_insights"],
                timestamp=pred["timestamp"]
            ))
        
        return TopStocksResponse(
            predictions=stock_predictions,
            total_count=len(stock_predictions),
            generated_at=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting top stocks: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/api/stock/{ticker}", response_model=StockSearchResponse)
async def get_stock_prediction(ticker: str):
    """
    Get prediction for a specific stock by ticker symbol
    """
    try:
        logger.info(f"Getting prediction for {ticker}")
        
        # Validate ticker
        ticker = ticker.upper().strip()
        if not ticker:
            raise HTTPException(
                status_code=400,
                detail="Invalid ticker symbol"
            )
        
        # Load model if not already loaded
        if prediction_model.model is None:
            prediction_model.load_model()
        
        if prediction_model.model is None:
            raise HTTPException(
                status_code=503,
                detail="Prediction model not available. Please train the model first."
            )
        
        # Get stock data
        stock_data = data_collector.get_stock_data(ticker)
        
        if stock_data is None or stock_data.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for ticker {ticker}"
            )
        
        # Get prediction
        prediction = prediction_model.predict_stock(stock_data)
        
        if "error" in prediction:
            raise HTTPException(
                status_code=500,
                detail=f"Prediction error: {prediction['error']}"
            )
        
        # Get company info
        company_info = None
        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            info = stock.info
            company_info = {
                "name": info.get("longName", "Unknown"),
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE"),
                "dividend_yield": info.get("dividendYield")
            }
        except Exception as e:
            logger.warning(f"Could not fetch company info for {ticker}: {str(e)}")
        
        return StockSearchResponse(
            ticker=ticker,
            prediction=prediction["prediction"],
            confidence=prediction["confidence"],
            probability_rise=prediction["probability_rise"],
            probability_fall=prediction["probability_fall"],
            current_price=prediction["current_price"],
            technical_insights=prediction["technical_insights"],
            timestamp=prediction["timestamp"],
            company_info=company_info
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting stock prediction for {ticker}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/api/stock/{ticker}/history")
async def get_stock_history(ticker: str, period: str = Query("1y", description="Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)")):
    """
    Get historical data for a specific stock
    """
    try:
        ticker = ticker.upper().strip()
        
        stock_data = data_collector.get_stock_data(ticker, period=period)
        
        if stock_data is None or stock_data.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No historical data found for {ticker}"
            )
        
        # Convert to JSON-serializable format
        history_data = []
        for index, row in stock_data.iterrows():
            history_data.append({
                "date": index.isoformat(),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": int(row["Volume"]),
                "rsi": float(row["RSI"]) if not pd.isna(row["RSI"]) else None,
                "macd": float(row["MACD"]) if not pd.isna(row["MACD"]) else None,
                "sma_20": float(row["SMA_20"]) if not pd.isna(row["SMA_20"]) else None,
                "sma_50": float(row["SMA_50"]) if not pd.isna(row["SMA_50"]) else None
            })
        
        return {
            "ticker": ticker,
            "period": period,
            "data": history_data,
            "count": len(history_data)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting history for {ticker}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/api/model/info")
async def get_model_info():
    """
    Get information about the current prediction model
    """
    try:
        return prediction_model.get_model_info()
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.post("/api/model/train")
async def train_model():
    """
    Train the prediction model with current market data
    """
    try:
        logger.info("Starting model training")
        
        # Get all tech stocks data
        stock_data = data_collector.get_all_tech_stocks_data()
        
        if not stock_data:
            raise HTTPException(
                status_code=503,
                detail="Unable to fetch stock data for training"
            )
        
        # Train model
        success = prediction_model.train_model(stock_data)
        
        if success:
            return {
                "message": "Model trained successfully",
                "timestamp": datetime.now().isoformat(),
                "stocks_processed": len(stock_data)
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Model training failed"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

# Import pandas for history endpoint
import pandas as pd 