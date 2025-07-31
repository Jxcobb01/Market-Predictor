import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import os

logger = logging.getLogger(__name__)

class StockPredictionModel:
    """Machine learning model for stock price prediction"""
    
    def __init__(self, model_path: str = "models/stock_predictor.pkl"):
        self.model_path = model_path
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
            'MACD', 'MACD_Signal', 'RSI',
            'BB_Upper', 'BB_Lower', 'BB_Middle',
            'Volume_SMA', 'Price_Change', 'Price_Change_5d',
            'Price_Change_20d', 'Volatility'
        ]
        
    def train_model(self, stock_data: Dict[str, pd.DataFrame]) -> bool:
        """Train the prediction model on historical stock data"""
        try:
            # Combine all stock data
            all_data = []
            for ticker, data in stock_data.items():
                if data is not None and not data.empty:
                    # Add ticker as a feature
                    data_copy = data.copy()
                    data_copy['Ticker'] = ticker
                    all_data.append(data_copy)
            
            if not all_data:
                logger.error("No valid stock data found for training")
                return False
                
            combined_data = pd.concat(all_data, ignore_index=True)
            
            # Prepare features
            features = combined_data[self.feature_columns].fillna(0)
            target = combined_data['Target']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features, target, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model (using Gradient Boosting for better performance)
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
            
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            logger.info(f"Model trained successfully. Accuracy: {accuracy:.3f}")
            logger.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")
            
            # Save model
            self.save_model()
            
            return True
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return False
    
    def predict_stock(self, stock_data: pd.DataFrame) -> Dict:
        """Predict if a stock will rise or fall"""
        try:
            if self.model is None:
                self.load_model()
            
            if self.model is None:
                return {"error": "Model not available"}
            
            # Get latest data point
            latest_data = stock_data.iloc[-1]
            
            # Prepare features
            features = latest_data[self.feature_columns].fillna(0).values.reshape(1, -1)
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            probability = self.model.predict_proba(features_scaled)[0]
            
            # Calculate confidence score
            confidence = max(probability) * 100
            
            # Determine trend
            trend = "RISE" if prediction == 1 else "FALL"
            
            # Get technical analysis insights
            technical_insights = self._analyze_technical_indicators(latest_data)
            
            return {
                "prediction": trend,
                "confidence": round(confidence, 2),
                "probability_rise": round(probability[1] * 100, 2),
                "probability_fall": round(probability[0] * 100, 2),
                "technical_insights": technical_insights,
                "current_price": latest_data['Close'],
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return {"error": str(e)}
    
    def predict_top_stocks(self, stock_data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """Predict top stocks most likely to rise"""
        predictions = []
        
        for ticker, data in stock_data.items():
            if data is not None and not data.empty:
                prediction = self.predict_stock(data)
                if "error" not in prediction:
                    prediction["ticker"] = ticker
                    predictions.append(prediction)
        
        # Sort by confidence and probability of rise
        predictions.sort(key=lambda x: (x["probability_rise"], x["confidence"]), reverse=True)
        
        # Return top 10
        return predictions[:10]
    
    def _analyze_technical_indicators(self, data: pd.Series) -> Dict:
        """Analyze technical indicators for insights"""
        insights = {}
        
        # RSI Analysis
        rsi = data['RSI']
        if rsi < 30:
            insights['rsi'] = "Oversold - potential buy signal"
        elif rsi > 70:
            insights['rsi'] = "Overbought - potential sell signal"
        else:
            insights['rsi'] = "Neutral"
        
        # MACD Analysis
        macd = data['MACD']
        macd_signal = data['MACD_Signal']
        if macd > macd_signal:
            insights['macd'] = "Bullish - MACD above signal line"
        else:
            insights['macd'] = "Bearish - MACD below signal line"
        
        # Moving Average Analysis
        sma_20 = data['SMA_20']
        sma_50 = data['SMA_50']
        current_price = data['Close']
        
        if current_price > sma_20 > sma_50:
            insights['moving_averages'] = "Strong uptrend"
        elif current_price > sma_20:
            insights['moving_averages'] = "Moderate uptrend"
        elif current_price < sma_20 < sma_50:
            insights['moving_averages'] = "Strong downtrend"
        else:
            insights['moving_averages'] = "Mixed signals"
        
        # Volume Analysis
        volume = data['Volume']
        volume_sma = data['Volume_SMA']
        if volume > volume_sma * 1.5:
            insights['volume'] = "High volume - strong conviction"
        elif volume < volume_sma * 0.5:
            insights['volume'] = "Low volume - weak conviction"
        else:
            insights['volume'] = "Normal volume"
        
        return insights
    
    def save_model(self):
        """Save the trained model"""
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'trained_at': datetime.now().isoformat()
            }
            
            joblib.dump(model_data, self.model_path)
            logger.info(f"Model saved to {self.model_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
    
    def load_model(self):
        """Load the trained model"""
        try:
            if os.path.exists(self.model_path):
                model_data = joblib.load(self.model_path)
                self.model = model_data['model']
                self.scaler = model_data['scaler']
                self.feature_columns = model_data['feature_columns']
                logger.info("Model loaded successfully")
            else:
                logger.warning("No saved model found")
                
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
    
    def get_model_info(self) -> Dict:
        """Get information about the current model"""
        if self.model is None:
            return {"status": "No model loaded"}
        
        return {
            "status": "Model loaded",
            "model_type": type(self.model).__name__,
            "feature_count": len(self.feature_columns),
            "features": self.feature_columns
        } 