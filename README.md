# IoT Device Identification Neural Network

Machine learning system for identifying IoT devices from network traffic patterns.  
**Current accuracy: 98.36%** with Histogram Gradient Boosting Classifier.

## Current Status

- **Production Model**: HGB (Histogram Gradient Boosting)
- **Location**: `outputs/hgb_test/` → `api_models/hgb/`
- **Accuracy**: 98.36% on 19,932 validation samples
- **API**: Ready at http://localhost:8000

� **[Full Status Report](STATUS.md)**

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

### 1. Setup

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Start API

```bash
./start_api.sh
```

Access at: http://localhost:8000/docs

### 3. Train Model

```bash
./train.sh
```

Or manually:
```bash
python -m src.training.train_hist_gb \
  --train-csv "data/Aalto_train_IoTDevID (1).csv" \
  --valid-csv "data/Aalto_test_IoTDevID (1).csv" \
  --feature-ranking-csv "data/veto_average_results (1).csv" \
  --top-k 48 \
  --max-iter 300
```

## Model Performance

| Metric | Value |
|--------|-------|
| Model | Histogram Gradient Boosting |
| Accuracy | 98.36% |
| Classes | 27 IoT devices |
| Features | 48 (top-ranked) |
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

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'src'` | Run from project root |
| Poor API accuracy | Check sklearn version: `pip install scikit-learn==1.7.2` |
| Port 8000 in use | `lsof -ti:8000 \| xargs kill -9` |

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
