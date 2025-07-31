"""
Test suite for NegaBot API
"""
import pytest
import httpx
import asyncio
import json
from fastapi.testclient import TestClient
from api import app

# Test client
client = TestClient(app)

class TestNegaBotAPI:
    """Test cases for NegaBot API endpoints"""
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "NegaBot API" in data["message"]
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "timestamp" in data
    
    def test_predict_endpoint_positive(self):
        """Test prediction endpoint with positive text"""
        test_data = {
            "text": "This product is absolutely amazing! Best purchase ever!"
        }
        response = client.post("/predict", json=test_data)
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert "text" in data
        assert "sentiment" in data
        assert "confidence" in data
        assert "predicted_class" in data
        assert "probabilities" in data
        assert "timestamp" in data
        
        # Check data types
        assert isinstance(data["confidence"], float)
        assert data["confidence"] >= 0 and data["confidence"] <= 1
        assert data["sentiment"] in ["Positive", "Negative"]
        assert data["predicted_class"] in [0, 1]
    
    def test_predict_endpoint_negative(self):
        """Test prediction endpoint with negative text"""
        test_data = {
            "text": "Terrible quality, broke after one day. Complete waste of money."
        }
        response = client.post("/predict", json=test_data)
        assert response.status_code == 200
        data = response.json()
        
        assert data["text"] == test_data["text"]
        assert data["sentiment"] in ["Positive", "Negative"]
        assert isinstance(data["confidence"], float)
    
    def test_predict_endpoint_with_metadata(self):
        """Test prediction endpoint with metadata"""
        test_data = {
            "text": "Pretty good value for money",
            "metadata": {"source": "test", "user_id": "123"}
        }
        response = client.post("/predict", json=test_data)
        assert response.status_code == 200
        data = response.json()
        assert data["text"] == test_data["text"]
    
    def test_predict_endpoint_empty_text(self):
        """Test prediction endpoint with empty text"""
        test_data = {"text": ""}
        response = client.post("/predict", json=test_data)
        assert response.status_code == 422  # Validation error
    
    def test_predict_endpoint_long_text(self):
        """Test prediction endpoint with very long text"""
        long_text = "This is a test. " * 100  # 1500+ characters
        test_data = {"text": long_text}
        response = client.post("/predict", json=test_data)
        # Should either work or return 422 for too long text
        assert response.status_code in [200, 422]
    
    def test_batch_predict_endpoint(self):
        """Test batch prediction endpoint"""
        test_data = {
            "tweets": [
                "Amazing product, highly recommend!",
                "Terrible experience, waste of money",
                "It's okay, nothing special"
            ]
        }
        response = client.post("/batch_predict", json=test_data)
        assert response.status_code == 200
        data = response.json()
        
        assert "results" in data
        assert "total_processed" in data
        assert "timestamp" in data
        assert len(data["results"]) == len(test_data["tweets"])
        assert data["total_processed"] == len(test_data["tweets"])
        
        # Check each result
        for i, result in enumerate(data["results"]):
            assert result["text"] == test_data["tweets"][i]
            assert result["sentiment"] in ["Positive", "Negative"]
            assert isinstance(result["confidence"], float)
    
    def test_batch_predict_empty_list(self):
        """Test batch prediction with empty list"""
        test_data = {"tweets": []}
        response = client.post("/batch_predict", json=test_data)
        assert response.status_code == 422  # Validation error
    
    def test_batch_predict_too_many_tweets(self):
        """Test batch prediction with too many tweets"""
        test_data = {"tweets": ["test tweet"] * 100}  # More than max limit
        response = client.post("/batch_predict", json=test_data)
        assert response.status_code == 422  # Validation error
    
    def test_stats_endpoint(self):
        """Test statistics endpoint"""
        # First make some predictions to have data
        test_tweets = [
            "Amazing product!",
            "Terrible quality",
            "Pretty good"
        ]
        
        for tweet in test_tweets:
            client.post("/predict", json={"text": tweet})
        
        # Now test stats
        response = client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        
        if data["total_predictions"] > 0:
            assert "total_predictions" in data
            assert "positive_count" in data
            assert "negative_count" in data
            assert "positive_percentage" in data
            assert "negative_percentage" in data
            assert "average_confidence" in data
            assert "last_updated" in data
            
            # Check data consistency
            assert data["positive_count"] + data["negative_count"] == data["total_predictions"]
            assert abs(data["positive_percentage"] + data["negative_percentage"] - 100) < 0.1

class TestModelPredictions:
    """Test cases for model prediction accuracy and consistency"""
    
    def test_prediction_consistency(self):
        """Test that same input gives same output"""
        test_text = "This product is amazing!"
        
        # Make multiple predictions
        responses = []
        for _ in range(3):
            response = client.post("/predict", json={"text": test_text})
            responses.append(response.json())
        
        # All responses should be identical
        for i in range(1, len(responses)):
            assert responses[i]["sentiment"] == responses[0]["sentiment"]
            assert abs(responses[i]["confidence"] - responses[0]["confidence"]) < 0.001
    
    def test_edge_cases(self):
        """Test edge cases and special inputs"""
        edge_cases = [
            "😊😊😊",  # Only emojis
            "123456789",  # Only numbers
            "!!!!!",  # Only punctuation
            "a",  # Single character
            "This is a normal sentence.",  # Normal case
        ]
        
        for text in edge_cases:
            response = client.post("/predict", json={"text": text})
            # Should not crash, even if prediction quality varies
            assert response.status_code == 200
            data = response.json()
            assert data["sentiment"] in ["Positive", "Negative"]

def test_database_integration():
    """Test database logging functionality"""
    from database import get_all_predictions, get_prediction_stats
    
    # Get initial count
    initial_predictions = get_all_predictions()
    initial_count = len(initial_predictions)
    
    # Make a prediction
    test_text = "Test tweet for database integration"
    response = client.post("/predict", json={"text": test_text})
    assert response.status_code == 200
    
    # Check if prediction was logged
    new_predictions = get_all_predictions()
    assert len(new_predictions) == initial_count + 1
    
    # Check if the new prediction is in the database
    latest_prediction = new_predictions[0]  # Most recent first
    assert latest_prediction["text"] == test_text
    
    # Test stats
    stats = get_prediction_stats()
    assert stats["total_predictions"] >= 1

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
