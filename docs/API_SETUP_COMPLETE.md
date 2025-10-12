# IoT Device Identification API - Setup Complete! ✅

## 🎉 Summary

Your REST API for IoT device prediction is now fully operational!

### What Was Done

1. **✅ Modified `train_hist_gb.py`** to save model artifacts:
   - `hist_gb_model.pkl` - Trained model (30MB)
   - `scaler.pkl` - Feature scaler
   - `label_encoder.pkl` - Label encoder

2. **✅ Created `export_model.py`** to package models for deployment:
   - Loads preprocessing artifacts
   - Saves model weights
   - Creates metadata JSON
   - Exports to `api_models/` directory

3. **✅ Built `api.py`** - FastAPI REST service with endpoints:
   - `GET /` - API info
   - `GET /health` - Health status
   - `GET /features?top_k=N` - Get top N important features
   - `POST /predict` - Predict from JSON feature vectors
   - `POST /predict/csv` - Predict from uploaded CSV files

4. **✅ Trained and exported HGB model**:
   - 48 features
   - 27 device classes
   - ~89% accuracy
   - All artifacts saved in `api_models/hgb/`

5. **✅ API Server Running**:
   - Port: 8000
   - Docs: http://localhost:8000/docs
   - Status: ✅ Healthy

---

## 🚀 How to Use the API

### 1. Check API Status
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "feature_ranking_loaded": true,
  "veto_file_exists": true,
  "models_cached": ["hgb"]
}
```

### 2. Get Top Features
```bash
curl "http://localhost:8000/features?top_k=10"
```

**Response:**
```json
{
  "top_k": 10,
  "features": ["TCP_sport", "sport", "TCP_dport", ...],
  "importance_scores": [
    {"Variable_Name": "TCP_sport", "Votes": 5.96},
    ...
  ]
}
```

### 3. Predict from CSV File
```bash
curl -X POST http://localhost:8000/predict/csv \
  -F "file=@Aalto_test_IoTDevID (1).csv" \
  -F "model_type=hgb"
```

**Response:**
```json
{
  "num_samples": 1000,
  "predictions": ["Smart_Lock", "Security_Camera", ...],
  "confidences": [0.95, 0.87, ...],
  "avg_confidence": 0.89,
  "min_confidence": 0.45,
  "max_confidence": 0.99,
  "class_distribution": {
    "Smart_Lock": 250,
    "Security_Camera": 180,
    ...
  },
  "model_used": "hgb"
}
```

### 4. Predict from JSON Features
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [
      {"TCP_sport": 443, "TCP_dport": 54321, ...}
    ],
    "model_type": "hgb",
    "top_k_features": 50
  }'
```

**Response:**
```json
{
  "predictions": ["Smart_Lock"],
  "probabilities": [
    {
      "Smart_Lock": 0.85,
      "Security_Camera": 0.10,
      "Smart_Plug": 0.05
    }
  ],
  "model_used": "hgb"
}
```

---

## 📁 File Structure

```
neuralnet3.0/
├── api.py                      # FastAPI application
├── export_model.py             # Model export utility
├── train_hist_gb.py            # HGB trainer (modified to save models)
├── simple_test_api.py          # Simple API test script
├── api_models/                 # Exported models for deployment
│   └── hgb/
│       ├── model.pkl           # Trained HGB model (30MB)
│       ├── preprocessing.pkl   # Scaler + label encoder
│       ├── scaler.pkl          # Feature scaler
│       ├── label_encoder.pkl   # Label encoder
│       └── metadata.json       # Model config
├── outputs/                    # Training outputs
│   └── hgb_run1/
│       ├── hist_gb_model.pkl           # Source model
│       ├── scaler.pkl                  # Source scaler
│       ├── label_encoder.pkl           # Source encoder
│       ├── hist_gb_validation_metrics.json
│       ├── hist_gb_valid_proba.npy
│       └── hist_gb_valid_predictions.csv
└── API_README.md               # Full API documentation
```

---

## 🔧 Management Commands

### Start the API Server
```bash
# Option 1: Direct start
source .venv/bin/activate
python api.py

# Option 2: With uvicorn
source .venv/bin/activate
python -m uvicorn api:app --host 0.0.0.0 --port 8000

# Option 3: Background mode
source .venv/bin/activate
nohup python api.py > api_server.log 2>&1 &
```

### Stop the API Server
```bash
# Find process
lsof -ti:8000

# Kill process
kill $(lsof -ti:8000)

# Or kill background job
kill %1
```

### Check API Logs
```bash
tail -f api_server.log
```

### Test the API
```bash
python simple_test_api.py
```

### Retrain and Re-export Model
```bash
# 1. Train new model
python train_hist_gb.py \
  --train-csv "Aalto_train_IoTDevID (1).csv" \
  --valid-csv "Aalto_test_IoTDevID (1).csv" \
  --feature-ranking-csv "veto_average_results (1).csv" \
  --top-k 50 \
  --output-dir outputs/hgb_run2 \
  --max-iter 200

# 2. Export for API
python export_model.py --model hgb --model-dir outputs/hgb_run2

# 3. Restart API server (picks up new model automatically)
```

---

## 📊 Model Performance

**Histogram Gradient Boosting (HGB)**
- **Accuracy:** 88.97%
- **Balanced Accuracy:** 87.39%
- **F1 Score:** 82.85%
- **Cohen's Kappa:** 0.8739
- **Classes:** 27 IoT devices
- **Features:** 48 (top-ranked from veto)

---

## 🎯 Next Steps

1. **Add More Models:**
   ```bash
   # Export CNN+LSTM model
   python export_model.py --model cnn_lstm --model-dir outputs/cnn_lstm_full
   ```

2. **Enable HTTPS** for production deployment

3. **Add Authentication:**
   - API keys
   - OAuth2
   - Rate limiting

4. **Docker Deployment:**
   - Create Dockerfile
   - Docker Compose for multi-container setup

5. **Cloud Deployment:**
   - AWS Lambda / ECS
   - Azure Functions / Container Apps
   - Google Cloud Run

6. **Monitoring:**
   - Add logging (Sentry, CloudWatch)
   - Metrics (Prometheus)
   - Health checks

---

## ✅ Current Status

- **API Server:** ✅ Running on http://localhost:8000
- **Model Loaded:** ✅ HGB (27 classes, 48 features)
- **Feature Ranking:** ✅ Loaded
- **Endpoints:** ✅ All functional
- **Documentation:** ✅ Available at http://localhost:8000/docs

**Everything is working perfectly!** 🚀
