#!/usr/bin/env python3
"""
NegaBot Demo Script
Demonstrates all the features of the NegaBot API
"""
import requests
import time
import random

# API Configuration
API_BASE = "http://localhost:8000"

# Sample tweets for testing
SAMPLE_TWEETS = [
    # Positive sentiment
    "This product exceeded my expectations! Absolutely fantastic quality.",
    "Amazing customer service, they went above and beyond!",
    "Love this purchase, best decision I made this year!",
    "Outstanding quality and fast delivery, highly recommend!",
    "Perfect product, exactly what I was looking for!",
    
    # Negative sentiment
    "Terrible quality, broke within a week of purchase.",
    "Worst customer service ever, completely unhelpful staff.",
    "Overpriced garbage, total waste of money.",
    "Poor quality materials, feels cheap and flimsy.",
    "Awful experience, would never buy from them again.",
    
    # Mixed/Neutral sentiment
    "It's okay, nothing special but does the job.",
    "Average product, could be better for the price.",
    "Not bad but not great either, mediocre quality.",
    "Pretty decent, meets basic expectations.",
    "Satisfactory purchase, no complaints but no praise either."
]

def test_health_endpoint():
    """Test the health endpoint"""
    print("🔍 Testing Health Endpoint...")
    try:
        response = requests.get(f"{API_BASE}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API Health: {data['status']}")
            print(f"✅ Model Loaded: {data['model_loaded']}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check error: {e}")
    print()

def test_single_predictions():
    """Test single predictions with various sentiments"""
    print("🎯 Testing Single Predictions...")
    
    test_cases = [
        "This product is absolutely amazing!",
        "Terrible quality, complete waste of money.",
        "It's pretty decent for the price."
    ]
    
    for i, text in enumerate(test_cases, 1):
        try:
            response = requests.post(
                f"{API_BASE}/predict",
                json={"text": text}
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"{i}. Text: {text}")
                print(f"   Sentiment: {data['sentiment']} (Confidence: {data['confidence']:.2%})")
                print(f"   Probabilities: +{data['probabilities']['positive']:.2%} / -{data['probabilities']['negative']:.2%}")
            else:
                print(f"❌ Prediction failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Prediction error: {e}")
        print()

def test_batch_predictions():
    """Test batch predictions"""
    print("📦 Testing Batch Predictions...")
    
    batch_tweets = random.sample(SAMPLE_TWEETS, 5)
    
    try:
        response = requests.post(
            f"{API_BASE}/batch_predict",
            json={"tweets": batch_tweets}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Processed {data['total_processed']} tweets:")
            
            for i, result in enumerate(data['results'], 1):
                print(f"{i}. {result['sentiment']} ({result['confidence']:.2%}): {result['text'][:60]}...")
        else:
            print(f"❌ Batch prediction failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Batch prediction error: {e}")
    print()

def generate_demo_data():
    """Generate demo data for the dashboard"""
    print("📊 Generating Demo Data...")
    
    # Add some random predictions to populate the dashboard
    random_tweets = random.sample(SAMPLE_TWEETS, 10)
    
    for i, tweet in enumerate(random_tweets, 1):
        try:
            response = requests.post(
                f"{API_BASE}/predict",
                json={
                    "text": tweet,
                    "metadata": {"demo": True, "batch": "demo_data"}
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"{i:2d}. {data['sentiment']:8s} | {tweet[:50]}...")
            
            # Small delay to simulate real usage
            time.sleep(0.1)
            
        except Exception as e:
            print(f"❌ Demo data error: {e}")
    print()

def show_final_stats():
    """Show final statistics"""
    print("📈 Final Statistics...")
    
    try:
        response = requests.get(f"{API_BASE}/stats")
        
        if response.status_code == 200:
            stats = response.json()
            print(f"📊 Total Predictions: {stats['total_predictions']}")
            print(f"😊 Positive: {stats['positive_count']} ({stats['positive_percentage']:.1f}%)")
            print(f"😞 Negative: {stats['negative_count']} ({stats['negative_percentage']:.1f}%)")
            print(f"🎯 Average Confidence: {stats['average_confidence']:.2%}")
        else:
            print(f"❌ Stats retrieval failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Stats error: {e}")
    print()

def main():
    """Main demo function"""
    print("🤖 NegaBot API Demo")
    print("=" * 50)
    print()
    
    # Test health
    test_health_endpoint()
    
    # Test single predictions
    test_single_predictions()
    
    # Test batch predictions
    test_batch_predictions()
    
    # Generate demo data
    generate_demo_data()
    
    # Show final stats
    show_final_stats()
    
    print("🎉 Demo Complete!")
    print()
    print("📋 Next Steps:")
    print("1. View API documentation: http://localhost:8000/docs")
    print("2. Check analytics dashboard: http://localhost:8501")
    print("3. Explore the prediction database: negabot_predictions.db")
    print("4. Run tests: python -m pytest test_api.py -v")

if __name__ == "__main__":
    main()
