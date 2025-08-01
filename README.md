# NegaBot API

**Tweet Sentiment Classification using SmolLM 360M V2 Model**

NegaBot is a complete sentiment analysis solution that detects positive and negative sentiment in tweets, particularly focusing on product criticism detection. Built with FastAPI, Streamlit, and the powerful `jatinmehra/NegaBot-Product-Criticism-Catcher` model.

**Note**: This API will be deployed to production in the future for public access.

## Features

- **Advanced AI Model**: Uses SmolLM 360M V2 for accurate sentiment classification
- **Fast API**: RESTful API built with FastAPI for high-performance predictions
- **Analytics Dashboard**: Streamlit dashboard for real-time analytics
- **Data Logging**: SQLite database for storing and analyzing predictions
- **Batch Processing**: Support for single and batch predictions
- **Visualizations**: Charts, word clouds, and trend analysis
- **Docker Ready**: Containerized deployment ready
- **Testing**: Demo script included

## Quick Start

### Option 1: Direct Python Execution

1. **Clone and Setup**
```bash
git clone <repository-url>
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

### API Usage

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

The Streamlit dashboard provides comprehensive analytics:

- **Real-time Metrics**: Total predictions, sentiment distribution, average confidence
- **Trend Analysis**: Time-series charts showing sentiment trends over time
- **Word Clouds**: Visual representation of common words in positive/negative tweets
- **Advanced Filtering**: Filter by sentiment, date range, and search terms
- **Data Export**: Download prediction data as CSV or JSON
- **Auto-refresh**: Optional auto-refresh for real-time monitoring

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