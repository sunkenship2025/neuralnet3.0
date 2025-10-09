# IoT Device Identification API

REST API for predicting IoT device types from network traffic features using trained deep learning models.

## 🚀 Features

- **Multiple Model Support**: CNN+LSTM, LSTM, CNN, MLP, Histogram Gradient Boosting
- **Flexible Input**: Accept JSON feature vectors or CSV files
- **Batch Processing**: Predict multiple samples in a single request
- **Confidence Scores**: Get probability distributions for top predicted classes
- **Feature Importance**: Query the most important features used by models

## 📋 Prerequisites

```bash
Python 3.8+
pip install -r requirements.txt
```

## 🛠️ Setup

### 1. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 2. Export Trained Models

Before running the API, you need to export your trained models:

```bash
# Export CNN+LSTM model (best performer)
python export_model.py --model cnn_lstm --model-dir outputs/cnn_lstm_full

# Export Histogram Gradient Boosting (fastest)
python export_model.py --model hgb --model-dir outputs/hgb_run1

# Export LSTM model
python export_model.py --model lstm --model-dir outputs/lstm_cosine_long_latest
```

This creates an `api_models/` directory with:
- `preprocessing.pkl`: Scaler and label encoder
- `metadata.json`: Model configuration
- `model_weights.pt` or `model.pkl`: Model weights

### 3. Start the API Server

```bash
python api.py
```

Or with uvicorn directly:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

## 📚 API Documentation

### Interactive Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Endpoints

#### `GET /`
Health check and API information

**Response:**
```json
{
  "status": "online",
  "message": "IoT Device Identification API",
  "available_models": ["cnn_lstm", "lstm", "cnn", "mlp", "hgb"],
  "endpoints": {
    "/predict": "POST - Predict device from features",
    "/predict/csv": "POST - Predict devices from CSV file",
    "/health": "GET - API health status"
  }
}
```

#### `GET /health`
Detailed health status

**Response:**
```json
{
  "status": "healthy",
  "feature_ranking_loaded": true,
  "veto_file_exists": true,
  "models_cached": ["cnn_lstm"]
}
```

#### `GET /features?top_k=50`
Get top K most important features

**Parameters:**
- `top_k` (int): Number of features to return (default: 50)

**Response:**
```json
{
  "top_k": 50,
  "features": ["feature1", "feature2", ...],
  "importance_scores": [
    {"Feature": "feature1", "Average": 0.95},
    {"Feature": "feature2", "Average": 0.89}
  ]
}
```

#### `POST /predict`
Predict device types from feature vectors

**Request Body:**
```json
{
  "features": [
    {
      "feature1": 0.5,
      "feature2": 1.2,
      "feature3": 0.8,
      ...
    },
    {
      "feature1": 0.3,
      "feature2": 0.9,
      "feature3": 1.1,
      ...
    }
  ],
  "top_k_features": 50,
  "model_type": "cnn_lstm"
}
```

**Response:**
```json
{
  "predictions": ["Smart_Lock", "Security_Camera"],
  "probabilities": [
    {
      "Smart_Lock": 0.85,
      "Security_Camera": 0.10,
      "Smart_Plug": 0.05
    },
    {
      "Security_Camera": 0.78,
      "Smart_Lock": 0.12,
      "Smart_Thermostat": 0.08
    }
  ],
  "model_used": "cnn_lstm"
}
```

#### `POST /predict/csv`
Predict device types from uploaded CSV file

**Parameters:**
- `file` (file): CSV file with network traffic features
- `model_type` (str): Model to use (default: "cnn_lstm")

**Response:**
```json
{
  "num_samples": 1000,
  "predictions": ["Smart_Lock", "Security_Camera", ...],
  "confidences": [0.85, 0.78, ...],
  "avg_confidence": 0.82,
  "min_confidence": 0.45,
  "max_confidence": 0.98,
  "class_distribution": {
    "Smart_Lock": 250,
    "Security_Camera": 180,
    "Smart_Plug": 320
  },
  "model_used": "cnn_lstm"
}
```

## 🧪 Usage Examples

### Python Client

```python
import requests
import pandas as pd

# API base URL
API_URL = "http://localhost:8000"

# Example 1: Predict from feature vectors
features = [
    {"feature1": 0.5, "feature2": 1.2, ...},
    {"feature1": 0.3, "feature2": 0.9, ...}
]

response = requests.post(
    f"{API_URL}/predict",
    json={
        "features": features,
        "model_type": "cnn_lstm",
        "top_k_features": 50
    }
)

result = response.json()
print(f"Predictions: {result['predictions']}")
print(f"Probabilities: {result['probabilities']}")

# Example 2: Predict from CSV file
with open("Aalto_test_IoTDevID (1).csv", "rb") as f:
    response = requests.post(
        f"{API_URL}/predict/csv",
        files={"file": f},
        params={"model_type": "cnn_lstm"}
    )

result = response.json()
print(f"Processed {result['num_samples']} samples")
print(f"Average confidence: {result['avg_confidence']:.2%}")
print(f"Class distribution: {result['class_distribution']}")

# Example 3: Get top features
response = requests.get(f"{API_URL}/features?top_k=20")
features = response.json()
print(f"Top 20 features: {features['features']}")
```

### cURL Examples

```bash
# Health check
curl http://localhost:8000/health

# Get top features
curl "http://localhost:8000/features?top_k=10"

# Predict from JSON
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [{"feature1": 0.5, "feature2": 1.2}],
    "model_type": "cnn_lstm"
  }'

# Predict from CSV
curl -X POST http://localhost:8000/predict/csv \
  -F "file=@Aalto_test_IoTDevID (1).csv" \
  -F "model_type=cnn_lstm"
```

## 🎯 Model Selection

| Model | Speed | Accuracy | Use Case |
|-------|-------|----------|----------|
| `hgb` | ⚡⚡⚡ Fast | ~89% | Production, low latency |
| `cnn_lstm` | ⚡⚡ Medium | ~66% | Best neural model, balanced |
| `lstm` | ⚡ Slow | ~65% | Sequence modeling |
| `cnn` | ⚡⚡ Medium | ~64% | Local patterns |
| `mlp` | ⚡⚡⚡ Fast | ~62% | Baseline |

**Recommendation**: Use `hgb` for production (fastest + highest accuracy), or `cnn_lstm` if you need neural network features.

## 📊 Input Data Format

### CSV File Requirements
- Must contain all top-K features (default: top 50 from veto ranking)
- Feature names must match training data
- Missing features will cause a 400 error

### Feature Vector Requirements
- JSON object with feature names as keys
- All required features must be present
- Values should be numeric (float/int)

## 🔧 Configuration

### Environment Variables

```bash
# API settings
export API_HOST="0.0.0.0"
export API_PORT=8000

# Model settings
export DEFAULT_MODEL="cnn_lstm"
export DEFAULT_TOP_K=50
export MODEL_DIR="api_models"
```

## 🐛 Troubleshooting

### Model Not Found Error
```
ValueError: Model directory not found: api_models/cnn_lstm
```
**Solution**: Run `export_model.py` to export your trained models first.

### Missing Features Error
```
Missing required features: ['feature1', 'feature2', ...]
```
**Solution**: Ensure your input data contains all required features from the veto ranking.

### Import Errors
```
ImportError: No module named 'fastapi'
```
**Solution**: Install dependencies: `pip install -r requirements.txt`

## 📦 Deployment

### Docker (Coming Soon)

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Production Considerations

1. **Use HTTPS** in production (add SSL certificates)
2. **Add authentication** (API keys, OAuth2)
3. **Enable CORS** if needed for web clients
4. **Add rate limiting** to prevent abuse
5. **Use a process manager** like gunicorn with multiple workers
6. **Monitor with logging** and error tracking
7. **Cache models** in memory (already implemented)

## 📝 License

Same as parent project.

## 🤝 Contributing

Contributions welcome! Please test with the provided examples before submitting PRs.
