"""
Database and Logging System for NegaBot API
Handles prediction logging using SQLite database
"""
import sqlite3
import json
import logging
from datetime import datetime
from typing import List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
DB_PATH = "negabot_predictions.db"

class PredictionLogger:
    def __init__(self, db_path: str = DB_PATH):
        """
        Initialize the prediction logger with SQLite database
        
        Args:
            db_path (str): Path to SQLite database file
        """
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create predictions table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        text TEXT NOT NULL,
                        sentiment TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        predicted_class INTEGER NOT NULL,
                        timestamp TEXT NOT NULL,
                        metadata TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create index for faster queries
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_sentiment ON predictions(sentiment)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_timestamp ON predictions(timestamp)
                """)
                
                conn.commit()
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise e
    
    def log_prediction(self, text: str, sentiment: str, confidence: float, 
                      predicted_class: int = None, metadata: Dict = None):
        """
        Log a prediction to the database
        
        Args:
            text (str): Input text
            sentiment (str): Predicted sentiment
            confidence (float): Prediction confidence
            predicted_class (int): Predicted class (0 or 1)
            metadata (dict): Optional metadata
        """
        try:
            # Infer predicted_class if not provided
            if predicted_class is None:
                predicted_class = 1 if sentiment == "Negative" else 0
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO predictions (text, sentiment, confidence, predicted_class, timestamp, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    text,
                    sentiment,
                    confidence,
                    predicted_class,
                    datetime.now().isoformat(),
                    json.dumps(metadata) if metadata else None
                ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error logging prediction: {str(e)}")
            raise e
    
    def get_all_predictions(self, limit: int = None) -> List[Dict]:
        """
        Get all predictions from the database
        
        Args:
            limit (int): Maximum number of records to return
            
        Returns:
            List of prediction dictionaries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                query = """
                    SELECT id, text, sentiment, confidence, predicted_class, timestamp, metadata, created_at
                    FROM predictions 
                    ORDER BY created_at DESC
                """
                
                if limit:
                    query += f" LIMIT {limit}"
                
                cursor.execute(query)
                rows = cursor.fetchall()
                
                predictions = []
                for row in rows:
                    prediction = {
                        "id": row[0],
                        "text": row[1],
                        "sentiment": row[2],
                        "confidence": row[3],
                        "predicted_class": row[4],
                        "timestamp": row[5],
                        "metadata": json.loads(row[6]) if row[6] else None,
                        "created_at": row[7]
                    }
                    predictions.append(prediction)
                
                return predictions
                
        except Exception as e:
            logger.error(f"Error getting predictions: {str(e)}")
            return []
    
    def get_predictions_by_sentiment(self, sentiment: str) -> List[Dict]:
        """
        Get predictions filtered by sentiment
        
        Args:
            sentiment (str): Sentiment to filter by ("Positive" or "Negative")
            
        Returns:
            List of prediction dictionaries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT id, text, sentiment, confidence, predicted_class, timestamp, metadata, created_at
                    FROM predictions 
                    WHERE sentiment = ?
                    ORDER BY created_at DESC
                """, (sentiment,))
                
                rows = cursor.fetchall()
                
                predictions = []
                for row in rows:
                    prediction = {
                        "id": row[0],
                        "text": row[1],
                        "sentiment": row[2],
                        "confidence": row[3],
                        "predicted_class": row[4],
                        "timestamp": row[5],
                        "metadata": json.loads(row[6]) if row[6] else None,
                        "created_at": row[7]
                    }
                    predictions.append(prediction)
                
                return predictions
                
        except Exception as e:
            logger.error(f"Error getting predictions by sentiment: {str(e)}")
            return []
    
    def get_stats(self) -> Dict:
        """
        Get prediction statistics
        
        Returns:
            Dictionary with statistics
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Total count
                cursor.execute("SELECT COUNT(*) FROM predictions")
                total_count = cursor.fetchone()[0]
                
                if total_count == 0:
                    return {
                        "total_predictions": 0,
                        "positive_count": 0,
                        "negative_count": 0,
                        "average_confidence": 0
                    }
                
                # Sentiment counts
                cursor.execute("SELECT sentiment, COUNT(*) FROM predictions GROUP BY sentiment")
                sentiment_counts = dict(cursor.fetchall())
                
                # Average confidence
                cursor.execute("SELECT AVG(confidence) FROM predictions")
                avg_confidence = cursor.fetchone()[0]
                
                return {
                    "total_predictions": total_count,
                    "positive_count": sentiment_counts.get("Positive", 0),
                    "negative_count": sentiment_counts.get("Negative", 0),
                    "average_confidence": round(avg_confidence, 4) if avg_confidence else 0
                }
                
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {}

# Global logger instance
_logger_instance = None

def get_logger():
    """Get the global logger instance"""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = PredictionLogger()
    return _logger_instance

def log_prediction(text: str, sentiment: str, confidence: float, metadata: Dict = None):
    """Convenience function to log a prediction"""
    logger_instance = get_logger()
    logger_instance.log_prediction(text, sentiment, confidence, metadata=metadata)

def get_all_predictions(limit: int = None) -> List[Dict]:
    """Convenience function to get all predictions"""
    logger_instance = get_logger()
    return logger_instance.get_all_predictions(limit=limit)

def get_predictions_by_sentiment(sentiment: str) -> List[Dict]:
    """Convenience function to get predictions by sentiment"""
    logger_instance = get_logger()
    return logger_instance.get_predictions_by_sentiment(sentiment)

def get_prediction_stats() -> Dict:
    """Convenience function to get prediction statistics"""
    logger_instance = get_logger()
    return logger_instance.get_stats()

if __name__ == "__main__":
    # Test the logging system
    logger_instance = PredictionLogger()
    
    # Test logging
    test_predictions = [
        ("This product is amazing!", "Positive", 0.95),
        ("Terrible quality, waste of money", "Negative", 0.89),
        ("It's okay, nothing special", "Positive", 0.67),
        ("Awful customer service", "Negative", 0.92)
    ]
    
    print("Testing prediction logging...")
    for text, sentiment, confidence in test_predictions:
        logger_instance.log_prediction(text, sentiment, confidence)
        print(f"Logged: {sentiment} - {text}")
    
    # Test retrieval
    print("\nRetrieving all predictions:")
    predictions = logger_instance.get_all_predictions()
    for pred in predictions:
        print(f"ID: {pred['id']}, Sentiment: {pred['sentiment']}, Text: {pred['text'][:50]}...")
    
    # Test stats
    print("\nPrediction statistics:")
    stats = logger_instance.get_stats()
    print(json.dumps(stats, indent=2))
