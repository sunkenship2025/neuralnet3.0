# IoT Device Identification Neural Network

Machine learning system for identifying IoT devices from network traffic patterns.  
**Current accuracy: 88.84%** with Histogram Gradient Boosting Classifier.

## Current Status

- **Production Model**: HGB (Histogram Gradient Boosting)
- **Location**: `outputs/hgb_test/` → `api_models/hgb/`
- **Accuracy**: 88.84% on 19,932 validation samples
- **API**: Ready at http://localhost:8000

📊 **[Full Status Report](STATUS.md)**

## Model Performance Comparison

| Model            | Accuracy | Balanced Acc | F1 (Macro) | Precision | Recall | Cohen's Kappa      |
|-------            |----------|--------------|------------|-----------|--------|---------------|
| **HistGradientBoosting** ⭐ | **88.84%** | **87.36%** | **0.8284** | **0.8178** | **0.8736** | **0.8786** |
| LSTM (Cosine Schedule) | 65.96% | 60.76% | 0.6216 | 0.6577 | 0.6076 | 0.6281 |
| CNN+LSTM Hybrid | 65.65% | 61.31% | 0.6246 | 0.6732 | 0.6131 | 0.6250 |
| CNN | 64.03% | 62.06% | 0.6436 | 0.6956 | 0.6206 | 0.6071 |
| LSTM | 61.27% | 56.47% | 0.5799 | 0.6179 | 0.5647 | 0.5767 |
| MLP | 57.70% | 51.75% | 0.5279 | 0.6205 | 0.5175 | 0.5368 |

**Key Insight**: HistGradientBoosting outperforms all neural networks by 22-31 percentage points because network traffic data is tabular, which tree ensembles handle better than deep learning.

Run `python src/compare_models.py` to regenerate this comparison.

## Project Structure

```
neuralnet3.0/
├── src/
│   ├── api/                        # REST API
│   │   ├── api.py                  # FastAPI server
│   │   └── start_api.sh            # API launcher
│   ├── training/                   # Model training
│   │   ├── train_hist_gb.py        # HGB (BEST: 98.36%)
│   │   ├── train_cnn_lstm.py       # CNN+LSTM
│   │   ├── train_lstm.py           # LSTM
│   │   ├── train_cnn.py            # CNN
│   │   ├── train_mlp.py            # MLP
│   │   └── train_autoencoder.py    # Autoencoder
│   └── utils/
│       └── export_model.py         # Export for deployment
├── data/                           # Training datasets
│   ├── Aalto_train_IoTDevID (1).csv
│   ├── Aalto_test_IoTDevID (1).csv
│   └── veto_average_results (1).csv
├── api_models/hgb/                 # DEPLOYED MODEL (serving)
├── outputs/hgb_test/               # PRODUCTION MODEL (source)
├── train.sh                        # Quick training launcher
└── start_api.sh                    # Quick API launcher
```

## Quick Start

### 1. Setup Environment

```bash
# Navigate to project directory
cd /Users/pranavreddy/Desktop/neuralnet3.0

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt
```

**Important**: Always activate the virtual environment before running any commands:
```bash
source .venv/bin/activate
```

### 2. Run the API Server

**Option A: Using the quick launcher (recommended)**
```bash
./start_api.sh
```

**Option B: Manual start**
```bash
# Make sure you're in the project root directory
cd /Users/pranavreddy/Desktop/neuralnet3.0

# Activate virtual environment
source .venv/bin/activate

# Start the API server
python -m uvicorn src.api.api:app --host 0.0.0.0 --port 8000 --reload
```

**Expected output:**
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 3. Access the API

Once the server is running:

- **Interactive API docs**: http://localhost:8000/docs
- **Alternative docs**: http://localhost:8000/redoc
- **Health check**: http://localhost:8000/health

**Test in browser**: Open http://localhost:8000/docs and click "Try it out" on any endpoint.

**Test with curl**:
```bash
# Health check
curl http://localhost:8000/health

# Make a prediction (example)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [{"feature_1": 0.5, "feature_2": 1.2}]}'
```

### 4. Stop the API

Press `Ctrl+C` in the terminal where the server is running, or:
```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9
```

### 5. Train a New Model (Optional)

**Option A: Using the quick launcher**
```bash
./train.sh
```

**Option B: Manual training**
```bash
# Make sure virtual environment is active
source .venv/bin/activate

# Train HGB model (recommended)
python -m src.training.train_hist_gb \
  --train-csv "data/Aalto_train_IoTDevID (1).csv" \
  --valid-csv "data/Aalto_test_IoTDevID (1).csv" \
  --feature-ranking-csv "data/veto_average_results (1).csv" \
  --top-k 48 \
  --max-iter 300 \
  --output-dir outputs/my_new_model
```

**Output**: Trained model saved in `outputs/my_new_model/`

## Model Performance

**Best Model: HistGradientBoosting (88.84% accuracy)**

| Metric | Value |
|--------|-------|
| Model | Histogram Gradient Boosting |
| Accuracy | 88.84% |
| Balanced Accuracy | 87.36% |
| F1-Score (Macro) | 0.8284 |
| Precision (Macro) | 0.8178 |
| Recall (Macro) | 0.8736 |
| Classes | 27 IoT devices |
| Features | 63 (top-ranked via VETO) |
| Training samples | ~22,000 |
| Validation samples | 19,932 |

**Supported devices**: HueSwitch, HueBridge, D-LinkDayCam, D-LinkCam, TP-LinkPlugHS100, AmazonEcho, SmartThings, and 20 more.

## API Usage

### Endpoints

**Health Check**
```bash
curl http://localhost:8000/health
```

**Make Prediction**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [{"feature_1": 0.5, ...}]}'
```

**Interactive Docs**
- Swagger: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Training Other Models

### Neural Networks

```bash
# CNN
python -m src.training.train_cnn --top-k 40 --epochs 30

# LSTM  
python -m src.training.train_lstm --top-k 60 --epochs 30

# CNN+LSTM
python -m src.training.train_cnn_lstm --top-k 60 --epochs 30 --bidirectional

# MLP
python -m src.training.train_mlp --top-k 20 --epochs 25
```

All models use the same data files and preprocessing.

## Export Model for API

After training:

```bash
python -m src.utils.export_model \
  --model hgb \
  --model-dir outputs/my_run \
  --export-dir api_models/hgb_new
```

This packages the model with correct preprocessing for API deployment.

## Troubleshooting

### Common Issues

**Problem**: `ModuleNotFoundError: No module named 'src'`  
**Solution**: Make sure you're running from the project root directory:
```bash
cd /Users/pranavreddy/Desktop/neuralnet3.0
python -m uvicorn src.api.api:app --port 8000
```

**Problem**: `bash: ./start_api.sh: Permission denied`  
**Solution**: Make the script executable:
```bash
chmod +x start_api.sh
chmod +x train.sh
```

**Problem**: Port 8000 already in use  
**Solution**: Kill the process using port 8000:
```bash
lsof -ti:8000 | xargs kill -9
```

**Problem**: Virtual environment not activated  
**Solution**: You'll see `(.venv)` in your terminal prompt when active. If not:
```bash
source .venv/bin/activate
```

**Problem**: Poor API prediction accuracy  
**Solution**: Check sklearn version matches training:
```bash
pip install scikit-learn==1.7.2
```

**Problem**: Missing data files  
**Solution**: Ensure all data files are in the `data/` directory:
```bash
ls -lh data/
# Should show:
# Aalto_train_IoTDevID (1).csv
# Aalto_test_IoTDevID (1).csv
# veto_average_results (1).csv
```

**Problem**: API returns errors about missing model files  
**Solution**: Check that the model exists:
```bash
ls -lh api_models/hgb/
# Should show: model.pkl, scaler.pkl, label_encoder.pkl, preprocessing.pkl
```

## Documentation

- [STATUS.md](STATUS.md) - Current status and model history
- [outputs/README.md](outputs/README.md) - Training outputs guide
- [api_models/README.md](api_models/README.md) - API models guide
- [data/README.md](data/README.md) - Dataset documentation

## Requirements

- Python 3.8+
- scikit-learn 1.7.2 (critical for model compatibility)
- FastAPI 0.104.0+
- PyTorch 2.1.0+ (for neural models)

---

**Last Updated**: October 12, 2025
