"""
NegaBot Model - Tweet Sentiment Classification
Uses the SmolLM 360M V2 model for product criticism detection
"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NegaBotModel:
    def __init__(self, model_name="jatinmehra/NegaBot-Product-Criticism-Catcher"):
        """
        Initialize the NegaBot model for sentiment classification
        
        Args:
            model_name (str): HuggingFace model identifier
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.load_model()
    
    def load_model(self):
        """Load the model and tokenizer from HuggingFace"""
        try:
            logger.info(f"Loading model: {self.model_name}")
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Set model to evaluation mode
            self.model.eval()
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise e
    
    def predict(self, text: str) -> dict:
        """
        Predict sentiment for a given text
        
        Args:
            text (str): Input text to classify
            
        Returns:
            dict: Prediction result with sentiment and confidence
        """
        try:
            # Tokenize input text
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # Apply softmax to get probabilities
                probabilities = torch.softmax(logits, dim=1)
                predicted_class = torch.argmax(logits, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            # Map prediction to sentiment
            sentiment = "Negative" if predicted_class == 1 else "Positive"
            
            return {
                "text": text,
                "sentiment": sentiment,
                "confidence": round(confidence, 4),
                "predicted_class": predicted_class,
                "probabilities": {
                    "positive": round(probabilities[0][0].item(), 4),
                    "negative": round(probabilities[0][1].item(), 4)
                }
            }
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise e
    
    def batch_predict(self, texts: list) -> list:
        """
        Predict sentiment for multiple texts
        
        Args:
            texts (list): List of texts to classify
            
        Returns:
            list: List of prediction results
        """
        results = []
        for text in texts:
            results.append(self.predict(text))
        return results

# Global model instance (singleton pattern)
_model_instance = None

def get_model():
    """Get the global model instance"""
    global _model_instance
    if _model_instance is None:
        _model_instance = NegaBotModel()
    return _model_instance

if __name__ == "__main__":
    # Test the model
    model = NegaBotModel()
    
    test_texts = [
        "This product is awful and broke within a week!",
        "Amazing quality, highly recommend this product!",
        "The service was okay, nothing special.",
        "Terrible customer support, waste of money!"
    ]
    
    print("Testing NegaBot Model:")
    print("=" * 50)
    
    for text in test_texts:
        result = model.predict(text)
        print(f"Text: {text}")
        print(f"Sentiment: {result['sentiment']} (Confidence: {result['confidence']:.2%})")
        print("-" * 30)
