import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import ta
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockDataCollector:
    """Collects and processes stock data for prediction models"""
    
    def __init__(self):
        # Top tech stocks to track
        self.tech_stocks = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX',
            'ADBE', 'CRM', 'ORCL', 'INTC', 'AMD', 'QCOM', 'AVGO', 'TXN',
            'CSCO', 'IBM', 'HPQ', 'DELL', 'ZM', 'SHOP', 'SQ', 'PYPL'
        ]
        
    def get_stock_data(self, ticker: str, period: str = "1y") -> Optional[pd.DataFrame]:
        """Fetch stock data for a given ticker"""
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period=period)
            
            if data.empty:
                logger.warning(f"No data found for {ticker}")
                return None
                
            return self._add_technical_indicators(data)
            
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {str(e)}")
            return None
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the stock data"""
        df = data.copy()
        
        # Moving averages
        df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df['EMA_12'] = ta.trend.ema_indicator(df['Close'], window=12)
        df['EMA_26'] = ta.trend.ema_indicator(df['Close'], window=26)
        
        # MACD
        df['MACD'] = ta.trend.macd_diff(df['Close'])
        df['MACD_Signal'] = ta.trend.macd_signal(df['Close'])
        
        # RSI
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        
        # Bollinger Bands
        df['BB_Upper'] = ta.volatility.bollinger_hband(df['Close'])
        df['BB_Lower'] = ta.volatility.bollinger_lband(df['Close'])
        df['BB_Middle'] = ta.volatility.bollinger_mavg(df['Close'])
        
        # Volume indicators
        df['Volume_SMA'] = ta.volume.volume_sma(df['Close'], df['Volume'])
        
        # Price changes
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Change_5d'] = df['Close'].pct_change(periods=5)
        df['Price_Change_20d'] = df['Close'].pct_change(periods=20)
        
        # Volatility
        df['Volatility'] = df['Price_Change'].rolling(window=20).std()
        
        return df
    
    def get_all_tech_stocks_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch data for all tech stocks"""
        stock_data = {}
        
        for ticker in self.tech_stocks:
            logger.info(f"Fetching data for {ticker}")
            data = self.get_stock_data(ticker)
            if data is not None:
                stock_data[ticker] = data
                
        return stock_data
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ML model"""
        df = data.copy()
        
        # Target variable: 1 if price increases next day, 0 otherwise
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        # Feature columns
        feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
            'MACD', 'MACD_Signal', 'RSI',
            'BB_Upper', 'BB_Lower', 'BB_Middle',
            'Volume_SMA', 'Price_Change', 'Price_Change_5d',
            'Price_Change_20d', 'Volatility'
        ]
        
        # Remove rows with NaN values
        df = df.dropna()
        
        return df[feature_columns + ['Target']]
    
    def get_latest_data(self, ticker: str) -> Optional[Dict]:
        """Get the latest data point for a stock"""
        data = self.get_stock_data(ticker, period="5d")
        if data is None or data.empty:
            return None
            
        latest = data.iloc[-1]
        
        return {
            'ticker': ticker,
            'price': latest['Close'],
            'change': latest['Price_Change'],
            'volume': latest['Volume'],
            'rsi': latest['RSI'],
            'macd': latest['MACD'],
            'sma_20': latest['SMA_20'],
            'sma_50': latest['SMA_50'],
            'volatility': latest['Volatility']
        } 