import uvicorn
import logging
from app.api.routes import app
from app.data.stock_data import StockDataCollector
from app.models.prediction_model import StockPredictionModel
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def initialize_app():
    """Initialize the application components"""
    try:
        logger.info("Initializing Stock Market Predictor...")
        
        # Create necessary directories
        os.makedirs("models", exist_ok=True)
        os.makedirs("data", exist_ok=True)
        
        # Initialize components
        data_collector = StockDataCollector()
        prediction_model = StockPredictionModel()
        
        # Try to load existing model
        prediction_model.load_model()
        
        if prediction_model.model is None:
            logger.warning("No trained model found. Use /api/model/train to train a new model.")
        else:
            logger.info("Model loaded successfully")
        
        logger.info("Application initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing application: {str(e)}")

if __name__ == "__main__":
    # Initialize the application
    initialize_app()
    
    # Run the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 