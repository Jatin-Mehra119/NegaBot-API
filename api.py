"""
NegaBot API - FastAPI application for tweet sentiment classification
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import logging
from datetime import datetime
import json
import os
from model import get_model
from database import log_prediction, get_all_predictions

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="NegaBot API",
    description="Tweet Sentiment Classification API using NegaBot model",
    version="1.0.0"
)

# Pydantic models for request/response validation
class TweetRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000, description="Tweet text to analyze")
    metadata: Optional[dict] = Field(default=None, description="Optional metadata")

class TweetResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    predicted_class: int
    probabilities: dict
    timestamp: str
    request_id: Optional[str] = None

class BatchTweetRequest(BaseModel):
    tweets: List[str] = Field(..., min_items=1, max_items=50, description="List of tweets to analyze")
    metadata: Optional[dict] = Field(default=None, description="Optional metadata")

class BatchTweetResponse(BaseModel):
    results: List[TweetResponse]
    total_processed: int
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str

# Global variables
model = None

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup"""
    global model
    try:
        logger.info("Starting NegaBot API...")
        model = get_model()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise e

@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to NegaBot API",
        "version": "1.0.0",
        "description": "Tweet Sentiment Classification using NegaBot model",
        "endpoints": {
            "predict": "/predict - Single tweet prediction",
            "batch_predict": "/batch_predict - Multiple tweets prediction",
            "health": "/health - API health check",
            "stats": "/stats - Prediction statistics"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        timestamp=datetime.now().isoformat()
    )

@app.post("/predict", response_model=TweetResponse)
async def predict_sentiment(request: TweetRequest):
    """
    Predict sentiment for a single tweet
    
    Args:
        request: TweetRequest containing the tweet text
        
    Returns:
        TweetResponse with prediction results
    """
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Get prediction from model
        result = model.predict(request.text)
        
        # Create response
        response = TweetResponse(
            text=result["text"],
            sentiment=result["sentiment"],
            confidence=result["confidence"],
            predicted_class=result["predicted_class"],
            probabilities=result["probabilities"],
            timestamp=datetime.now().isoformat()
        )
        
        # Log the prediction
        log_prediction(
            text=request.text,
            sentiment=result["sentiment"],
            confidence=result["confidence"],
            metadata=request.metadata
        )
        
        logger.info(f"Prediction made: {result['sentiment']} (confidence: {result['confidence']:.2%})")
        return response
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/batch_predict", response_model=BatchTweetResponse)
async def batch_predict_sentiment(request: BatchTweetRequest):
    """
    Predict sentiment for multiple tweets
    
    Args:
        request: BatchTweetRequest containing list of tweets
        
    Returns:
        BatchTweetResponse with all prediction results
    """
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Get predictions for all tweets
        results = model.batch_predict(request.tweets)
        
        # Create response objects
        responses = []
        for result in results:
            response = TweetResponse(
                text=result["text"],
                sentiment=result["sentiment"],
                confidence=result["confidence"],
                predicted_class=result["predicted_class"],
                probabilities=result["probabilities"],
                timestamp=datetime.now().isoformat()
            )
            responses.append(response)
            
            # Log each prediction
            log_prediction(
                text=result["text"],
                sentiment=result["sentiment"],
                confidence=result["confidence"],
                metadata=request.metadata
            )
        
        batch_response = BatchTweetResponse(
            results=responses,
            total_processed=len(responses),
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"Batch prediction completed: {len(responses)} tweets processed")
        return batch_response
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/stats", response_model=dict)
async def get_prediction_stats():
    """
    Get prediction statistics
    
    Returns:
        Dictionary with prediction statistics
    """
    try:
        predictions = get_all_predictions()
        
        if not predictions:
            return {
                "total_predictions": 0,
                "positive_count": 0,
                "negative_count": 0,
                "average_confidence": 0,
                "message": "No predictions found"
            }
        
        total = len(predictions)
        positive_count = sum(1 for p in predictions if p["sentiment"] == "Positive")
        negative_count = total - positive_count
        avg_confidence = sum(p["confidence"] for p in predictions) / total
        
        stats = {
            "total_predictions": total,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "positive_percentage": round((positive_count / total) * 100, 2),
            "negative_percentage": round((negative_count / total) * 100, 2),
            "average_confidence": round(avg_confidence, 4),
            "last_updated": datetime.now().isoformat()
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
