# NeuralNet 3.0 – IoT Device Identification

This project trains neural classifiers on the Aalto IoT Device Identification dataset using
PyTorch. It includes multiple architectures (MLP, CNN, LSTM, CNN+LSTM, Autoencoder) and tree-based
models (Histogram Gradient Boosting) that can capture patterns in the top-*K* veto-ranked features.
All models evaluate on a held-out validation split with comprehensive metrics.

**🆕 NEW: REST API for inference!** Deploy trained models as a FastAPI service. See [API Documentation](API_README.md).

> **Note**: The repository currently ships `Aalto_train_IoTDevID (1).csv` and
> `Aalto_test_IoTDevID (1).csv`. The latter is treated as the validation split referenced in the
> original instructions (`Aalto_validation_IoTDevID.csv`).

## Environment setup

1. Create a virtual environment (recommended) and activate it.

  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

2. Install dependencies inside the environment:

  ```bash
  pip install -r requirements.txt
  ```

PyTorch automatically detects Apple Metal (MPS) on Apple Silicon. No additional steps are required
beyond installing the `torch` wheel that supports MPS (version ≥ 2.1 is recommended).

## Training – MLP

Run the training script from the repository root:

```bash
python train_mlp.py \
  --train-csv "Aalto_train_IoTDevID (1).csv" \
  --valid-csv "Aalto_test_IoTDevID (1).csv" \
  --feature-ranking-csv "veto_average_results (1).csv" \
  --top-k 20 \
  --epochs 25 \
  --batch-size 512 \
  --lr 1e-3
```

Key options:

- `--top-k`: Number of highest-vote features to include (default: 20).
- `--hidden-dims`: Hidden layer widths, e.g. `--hidden-dims 256 128`.
- `--output-dir`: Target directory for metrics artifacts (default: `outputs/`).

The script automatically selects the MPS backend when available, falling back to CUDA or CPU.

## Training – 1D CNN

Run the convolutional baseline to capture short-range feature interactions:

```bash
python train_cnn.py \
  --train-csv "Aalto_train_IoTDevID (1).csv" \
  --valid-csv "Aalto_test_IoTDevID (1).csv" \
  --feature-ranking-csv "veto_average_results (1).csv" \
  --top-k 40 \
  --epochs 30 \
  --batch-size 256 \
  --conv-channels 64 128 128 \
  --lr 5e-4
```

Key options:

- `--top-k`: Number of highest-vote features to include (default: 40).
- `--conv-channels`: Output channels per Conv1d block (default: `64 128 128`).
- `--kernel-size`: Odd-valued kernel size; default 3 keeps the feature length with padding.
- `--dropout`: Shared dropout probability applied after each block and dense layer (default: 0.2).
- `--output-dir`: Target directory for metrics artifacts (default: `outputs/`).

This script uses the same device auto-selection logic, metrics, and preprocessing as the MLP.

## Training – CNN + LSTM Hybrid

Blend local feature extraction with longer-range sequence modelling:

```bash
python train_cnn_lstm.py \
  --train-csv "Aalto_train_IoTDevID (1).csv" \
  --valid-csv "Aalto_test_IoTDevID (1).csv" \
  --feature-ranking-csv "veto_average_results (1).csv" \
  --top-k 60 \
  --epochs 30 \
  --batch-size 256 \
  --conv-channels 64 128 \
  --lstm-hidden-size 128 \
  --bidirectional
```

Highlights:

- Convolutional blocks capture short-range interactions before a bidirectional LSTM models longer
  dependencies.
- Shares the preprocessing pipeline with other trainers, so outputs are comparable and
  ensemble-ready.
- Supports the same learning-rate schedulers as the LSTM script (`none`, `step`, `cosine`) and
  optional gradient clipping via `--max-grad-norm`.

## Training – Feedforward Autoencoder

Learn latent representations and reconstruction errors for the ranked features:

```bash
python train_autoencoder.py \
  --train-csv "Aalto_train_IoTDevID (1).csv" \
  --valid-csv "Aalto_test_IoTDevID (1).csv" \
  --feature-ranking-csv "veto_average_results (1).csv" \
  --top-k 60 \
  --epochs 40 \
  --hidden-dims 256 128 \
  --latent-dim 32 \
  --save-embeddings
```

Highlights:

- Mirrors the same scaling and feature-selection pipeline as the classifiers but optimises MSE.
- Exports per-sample reconstruction errors (`autoencoder_*_errors.csv`) and optional latent
  embeddings (`--save-embeddings`) for downstream anomaly detection or ensembling.
- Writes the trained weights and scaler statistics to `autoencoder_state.pt` for reuse.

## Training – Histogram Gradient Boosting

The tree ensemble baseline mirrors the neural preprocessing while offering interpretable feature
importances and probability outputs suitable for ensembling:

```bash
python train_hist_gb.py \
  --train-csv "Aalto_train_IoTDevID (1).csv" \
  --valid-csv "Aalto_test_IoTDevID (1).csv" \
  --feature-ranking-csv "veto_average_results (1).csv" \
  --top-k 120 \
  --max-iter 300 \
  --output-dir outputs/hgb_run1
```

Highlights:

- Automatically enforces label overlap and drops missing ranked features from each split.
- Saves `hist_gb_validation_metrics.json`, `hist_gb_valid_proba.npy`, and
  `hist_gb_valid_predictions.csv` for downstream ensemble stacking.
- `--class-weight balanced` by default to compensate for skewed device frequencies.

## Outputs

- `outputs/mlp_validation_metrics.json`: Training history and validation metrics
  (accuracy, balanced accuracy, macro precision/recall/F1, Cohen's kappa).
- `outputs/cnn_validation_metrics.json`: Analogous metrics for the Conv1d model.
- `outputs/cnn_lstm_validation_metrics.json`: Metrics history for the CNN+LSTM hybrid (path varies
  with `--output-dir`).
- `outputs/autoencoder_run1/autoencoder_metrics.json`: Reconstruction losses and configuration for
  the autoencoder (path varies with `--output-dir`).
- `outputs/autoencoder_run1/autoencoder_valid_errors.csv`: Per-sample reconstruction errors, useful
  for anomaly detection (path varies with `--output-dir`).
- `outputs/autoencoder_run1/autoencoder_valid_latent.npy`: Optional latent embeddings when
  `--save-embeddings` is enabled (path varies with `--output-dir`).
- `outputs/hgb_run1/hist_gb_validation_metrics.json`: Metrics, per-class report, and configuration
  details for the histogram gradient boosting run (path varies with `--output-dir`).
- `outputs/hgb_run1/hist_gb_valid_proba.npy`: Validation probabilities ready for ensemble
  blending (path varies with `--output-dir`).
- `outputs/hgb_run1/hist_gb_valid_predictions.csv`: Ground-truth vs. predicted labels for the
  validation split.
- Console logs showing epoch loss and validation F1 for quick comparison with DecisionTree baselines.

## Reproducibility

- Deterministic random seeds are configured for NumPy and PyTorch (`--seed`, default 42).
- Feature standardisation uses `StandardScaler`, fit on the training split and applied to validation.

## 🌐 REST API for Inference

Deploy your trained models as a REST API! See the complete [API Documentation](API_README.md) for details.

### Quick Start

1. **Export a trained model:**
   ```bash
   python export_model.py --model cnn_lstm --model-dir outputs/cnn_lstm_full
   ```

2. **Start the API server:**
   ```bash
   ./start_api.sh
   # Or manually:
   # python -m uvicorn api:app --host 0.0.0.0 --port 8000
   ```

3. **Test the API:**
   ```bash
   python test_api.py
   ```

4. **Access documentation:**
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

### API Features

- 📊 **Multiple Models**: CNN+LSTM, LSTM, CNN, MLP, Histogram GB
- 📥 **Flexible Input**: JSON feature vectors or CSV file uploads
- 🚀 **Batch Processing**: Predict multiple samples at once
- 📈 **Confidence Scores**: Get probability distributions for predictions
- 🔍 **Feature Inspection**: Query top-K most important features

### Example Usage

```python
import requests

# Predict from features
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "features": [{"feature1": 0.5, "feature2": 1.2, ...}],
        "model_type": "cnn_lstm"
    }
)
print(response.json())
# Output: {"predictions": ["Smart_Lock"], "probabilities": [...], ...}
```

See [API_README.md](API_README.md) for complete documentation and examples.

## Next steps

- ✅ Tune hyperparameters such as hidden widths, learning rate, and dropout probability.
- ✅ Export trained model weights for downstream evaluation or deployment.
- 🔄 Add confusion matrices or per-class metrics for deeper analysis.
- 🚀 Deploy API to production with Docker and load balancing.
- 📊 Implement ensemble voting across multiple models.
