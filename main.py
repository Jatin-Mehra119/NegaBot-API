"""
NegaBot API - Main Entry Point
Tweet Sentiment Classification using SmolLM 360M V2 Model
"""
import sys
import argparse
import time
from multiprocessing import Process

def run_api():
    """Run the FastAPI application"""
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

def run_dashboard():
    """Run the Streamlit dashboard"""
    import streamlit.web.cli as stcli
    sys.argv = ["streamlit", "run", "dashboard.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
    stcli.main()

def run_both():
    """Run both API and dashboard concurrently"""
    print("🚀 Starting NegaBot API and Dashboard...")
    
    # Start API process
    api_process = Process(target=run_api)
    api_process.start()
    print("✅ API started on http://localhost:8000")
    
    # Wait a moment for API to start
    time.sleep(3)
    
    # Start dashboard process
    dashboard_process = Process(target=run_dashboard)
    dashboard_process.start()
    print("✅ Dashboard started on http://localhost:8501")
    
    print("\n🎉 NegaBot is ready!")
    print("📊 API Documentation: http://localhost:8000/docs")
    print("📈 Analytics Dashboard: http://localhost:8501")
    print("\nPress Ctrl+C to stop both services")
    
    try:
        # Wait for processes
        api_process.join()
        dashboard_process.join()
    except KeyboardInterrupt:
        print("\n🛑 Stopping services...")
        api_process.terminate()
        dashboard_process.terminate()
        api_process.join()
        dashboard_process.join()
        print("✅ Services stopped")

def test_model():
    """Test the model with sample tweets"""
    print("🧪 Testing NegaBot model...")
    
    from model import NegaBotModel
    
    model = NegaBotModel()
    
    test_tweets = [
        "This product is absolutely amazing! Best purchase ever!",
        "Terrible quality, broke after one day. Complete waste of money.",
        "It's okay, nothing special but does the job.",
        "Outstanding customer service and fast delivery!",
        "Awful experience, would not recommend to anyone.",
        "Pretty good value for money, satisfied with purchase."
    ]
    
    print("\n📊 Test Results:")
    print("=" * 60)
    
    for i, tweet in enumerate(test_tweets, 1):
        result = model.predict(tweet)
        print(f"\n{i}. Tweet: {tweet}")
        print(f"   Sentiment: {result['sentiment']} (Confidence: {result['confidence']:.2%})")
        print(f"   Probabilities: Positive: {result['probabilities']['positive']:.2%}, "
              f"Negative: {result['probabilities']['negative']:.2%}")

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="NegaBot - Tweet Sentiment Classification")
    
    parser.add_argument("--mode", "-m", 
                       choices=["api", "dashboard", "both", "test"],
                       default="both",
                       help="Run mode: api only, dashboard only, both, or test model")
    
    parser.add_argument("--host", 
                       default="0.0.0.0",
                       help="Host to run services on (default: 0.0.0.0)")
    
    parser.add_argument("--api-port", 
                       type=int,
                       default=8000,
                       help="Port for API service (default: 8000)")
    
    parser.add_argument("--dashboard-port", 
                       type=int,
                       default=8501,
                       help="Port for dashboard service (default: 8501)")
    
    args = parser.parse_args()
    
    print("🤖 NegaBot - Tweet Sentiment Classification")
    print("=" * 50)
    
    if args.mode == "test":
        test_model()
    elif args.mode == "api":
        print(f"🚀 Starting API on {args.host}:{args.api_port}")
        import uvicorn
        uvicorn.run("api:app", host=args.host, port=args.api_port, reload=True)
    elif args.mode == "dashboard":
        print(f"📈 Starting Dashboard on {args.host}:{args.dashboard_port}")
        import streamlit.web.cli as stcli
        sys.argv = ["streamlit", "run", "dashboard.py", 
                   "--server.port", str(args.dashboard_port), 
                   "--server.address", args.host]
        stcli.main()
    else:  # both
        run_both()

if __name__ == "__main__":
    main()