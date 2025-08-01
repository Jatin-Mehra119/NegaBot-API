<div align="center">

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Jatin-Mehra119/NegaBot-API)
[![Model Card](https://img.shields.io/badge/Model%20Card-NegaBot-%23FF6600?style=flat&logo=huggingface&logoColor=yellow)](https://huggingface.co/jatinmehra/NegaBot-Product-Criticism-Catcher)

</div>

# NegaBot API

**Tweet Sentiment Classification using [NegaBot Model SmolLM2-360M V2](https://github.com/Jatin-Mehra119/NegaBot-Product-Criticism-Catcher) | [Model Card](https://huggingface.co/jatinmehra/NegaBot-Product-Criticism-Catcher)**

NegaBot is a complete sentiment analysis solution that detects positive and negative sentiment in tweets, particularly focusing on product criticism detection. Built with FastAPI, Streamlit, and the powerful `jatinmehra/NegaBot-Product-Criticism-Catcher` model.

**Note**: This API is now deployed and available for public testing!

## 🚀 Live Demo

**Try it now**: The API is deployed on Hugging Face Spaces and available for testing at:
https://jatinmehra-negabot-api.hf.space/docs#/

The live deployment features:
- **Unified Interface**: Both API and dashboard accessible on a single port
- **Interactive Documentation**: Full Swagger UI for testing endpoints
- **Built-in Analytics**: HTML dashboard integrated with the API
- **Ready to Use**: No setup required, just start making requests!

## Features

- **Advanced AI Model**: Uses SmolLM 360M V2 for accurate sentiment classification
- **Fast API**: RESTful API built with FastAPI for high-performance predictions
- **Analytics Dashboard**: Streamlit dashboard for real-time analytics
- **Data Logging**: SQLite database for storing and analyzing predictions
- **Batch Processing**: Support for single and batch predictions
- **Visualizations**: Charts, word clouds, and trend analysis
- **Docker Ready**: Containerized deployment ready
- **Testing**: Demo script included

## 🌐 Deployment

### Hugging Face Spaces
The API is currently deployed on Hugging Face Spaces for demo and testing purposes:

**🔗 Live API**: https://jatinmehra-negabot-api.hf.space/docs#/

**Key Features of the Deployment:**
- **Single Port Architecture**: Both API endpoints and analytics dashboard run on the same port
- **Interactive Documentation**: Full Swagger UI available for testing all endpoints
- **Built-in Dashboard**: Analytics dashboard accessible at `/dashboard` endpoint
- **No Setup Required**: Ready to use immediately for testing and integration
- **Public Access**: Available for demonstration and development purposes

**Deployment Differences:**
- Unified port structure (dashboard accessible via `/dashboard` instead of separate port)
- HTML-based dashboard instead of Streamlit (for better integration)
- Optimized for serverless deployment on Hugging Face infrastructure

## Quick Start

### Option 1: Direct Python Execution

1. **Clone and Setup**
```bash
git clone https://github.com/Jatin-Mehra119/NegaBot-API.git
cd NegaBot-API
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2. **Run Everything**
```bash
python main.py --mode both
```

3. **Access the Services**
- API Documentation: http://localhost:8000/docs
- Analytics Dashboard: http://localhost:8501

### Option 2: Docker Deployment

1. **Build and Run**
```bash
docker build -t negabot .
docker run -p 8000:8000 -p 8501:8501 negabot
```

## Usage Examples

### Live API Usage (Hugging Face Deployment)

#### Single Prediction
```bash
curl -X POST "https://jatinmehra-negabot-api.hf.space/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "This product is amazing! Best purchase ever!"}'
```

#### Batch Prediction
```bash
curl -X POST "https://jatinmehra-negabot-api.hf.space/batch_predict" \
     -H "Content-Type: application/json" \
     -d '{
       "tweets": [
         "Amazing product, highly recommend!",
         "Terrible quality, waste of money",
         "Its okay, nothing special"
       ]
     }'
```

#### Python Client Example (Live API)
```python
import requests

# Single prediction
response = requests.post(
    "https://jatinmehra-negabot-api.hf.space/predict",
    json={"text": "This product broke after one week!"}
)
result = response.json()
print(f"Sentiment: {result['sentiment']} (Confidence: {result['confidence']:.2%})")

# Batch prediction
response = requests.post(
    "https://jatinmehra-negabot-api.hf.space/batch_predict",
    json={
        "tweets": [
            "Love this product!",
            "Terrible experience",
            "Pretty decent quality"
        ]
    }
)
results = response.json()
for result in results['results']:
    print(f"'{result['text']}' -> {result['sentiment']}")
```

### Local API Usage

#### Single Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "This product is amazing! Best purchase ever!"}'
```

#### Batch Prediction
```bash
curl -X POST "http://localhost:8000/batch_predict" \
     -H "Content-Type: application/json" \
     -d '{
       "tweets": [
         "Amazing product, highly recommend!",
         "Terrible quality, waste of money",
         "Its okay, nothing special"
       ]
     }'
```

#### Python Client Example
```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "This product broke after one week!"}
)
result = response.json()
print(f"Sentiment: {result['sentiment']} (Confidence: {result['confidence']:.2%})")

# Batch prediction
response = requests.post(
    "http://localhost:8000/batch_predict",
    json={
        "tweets": [
            "Love this product!",
            "Terrible experience",
            "Pretty decent quality"
        ]
    }
)
results = response.json()
for result in results['results']:
    print(f"'{result['text']}' -> {result['sentiment']}")
```

### Model Usage (Direct)

```python
from model import NegaBotModel

# Initialize model
model = NegaBotModel()

# Single prediction
result = model.predict("This product is awful and broke within a week!")
print(f"Sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Probabilities: {result['probabilities']}")

# Batch prediction
texts = [
    "Amazing quality, highly recommend!",
    "Terrible customer service",
    "Pretty good value for money"
]
results = model.batch_predict(texts)
for result in results:
    print(f"{result['text']} -> {result['sentiment']}")
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information and available endpoints |
| `/health` | GET | Health check and model status |
| `/predict` | POST | Single tweet sentiment prediction |
| `/batch_predict` | POST | Batch tweet sentiment prediction |
| `/stats` | GET | Prediction statistics and analytics |

### Request/Response Schemas

#### Predict Request
```json
{
  "text": "string (1-1000 chars)",
  "metadata": {
    "optional": "metadata object"
  }
}
```

#### Predict Response
```json
{
  "text": "input text",
  "sentiment": "Positive|Negative",
  "confidence": 0.95,
  "predicted_class": 0,
  "probabilities": {
    "positive": 0.95,
    "negative": 0.05
  },
  "timestamp": "2024-01-01T12:00:00"
}
```

## Dashboard Features

### Live Dashboard
Access the analytics dashboard at: https://jatinmehra-negabot-api.hf.space/dashboard

### Local Dashboard Features
The built-in analytics dashboard provides comprehensive analytics:

- **Real-time Metrics**: Total predictions, sentiment distribution, average confidence
- **Interactive Charts**: Visual representation of sentiment trends and distributions
- **Recent Predictions**: View latest prediction results with confidence scores
- **Data Export**: Download prediction data as CSV or JSON
- **Unified Interface**: Dashboard accessible through the same port as the API (for Hugging Face deployment)
- **Auto-refresh**: Real-time updates as new predictions are made

## Testing

Run the demo script to test the functionality:

```bash
# Test model directly
python demo.py

# Test with different modes
python main.py --mode test
```

## Project Structure

```
NegaBot-API/
├── main.py              # Main entry point and CLI
├── api.py               # FastAPI application
├── model.py             # NegaBot model wrapper
├── database.py          # SQLite database and logging
├── dashboard.py         # Streamlit analytics dashboard
├── demo.py              # Feature demonstration script
├── requirements.txt     # Python dependencies
├── Dockerfile           # Docker configuration
├── README.md           # This file
└── negabot_predictions.db # Database (created at runtime)
```

## Configuration

### Command Line Options

```bash
python main.py --help
```

Options:
- `--mode`: `api`, `dashboard`, `both`, or `test`
- `--host`: Host address (default: 0.0.0.0)
- `--api-port`: API port (default: 8000)
- `--dashboard-port`: Dashboard port (default: 8501)

### Environment Variables

- `PYTHONPATH`: Set to project root
- `PYTHONUNBUFFERED`: Set to 1 for better logging

## Model Information

- **Model**: `jatinmehra/NegaBot-Product-Criticism-Catcher`
- **Base Architecture**: SmolLM 360M V2
- **Task**: Binary sentiment classification
- **Classes**: 
  - 0: Positive sentiment
  - 1: Negative sentiment (criticism/complaints)
- **Input**: Text (max 512 tokens)
- **Output**: Sentiment label + confidence scores

### Scaling Considerations

- **API Scaling**: Use multiple worker processes with Gunicorn
- **Database**: Consider PostgreSQL for high-volume production use
- **Caching**: Add Redis for frequently requested predictions
- **Load Balancing**: Use nginx or HAProxy for multiple instances

## Logging and Monitoring

### Database Schema

```sql
CREATE TABLE predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    sentiment TEXT NOT NULL,
    confidence REAL NOT NULL,
    predicted_class INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    metadata TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

### Log Files

- Application logs: Console output
- Prediction logs: SQLite database
- Access logs: Uvicorn/Gunicorn logs

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the Apache-2.0 License - see the [LICENSE](LICENSE) file for details.

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Ensure internet connection for downloading the model
   - Check disk space (model is ~1.5GB)
   - Verify transformers library version

2. **Port Conflicts**
   - Change ports using command line arguments
   - Check if ports 8000/8501 are already in use

3. **Database Permissions**
   - Ensure write permissions in the project directory
   - Check SQLite installation

4. **Memory Issues**
   - Model requires ~8GB RAM minimum
   - Consider using CPU-only inference for smaller systems

### Getting Help

- Open an issue on GitHub
- Check the documentation
- Search existing issues

---

**Built with FastAPI, Streamlit, and the powerful NegaBot model.**
