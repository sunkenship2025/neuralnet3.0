# Outputs Directory

This directory contains training outputs from various model runs.

## 📊 Current Status

### ⭐ **ACTIVE MODEL**: `hgb_test/`
- **Date**: October 12, 2025
- **Accuracy**: 98.36%
- **Status**: ✅ Currently deployed in API
- **Files**:
  - `hist_gb_model.pkl` - Trained model
  - `scaler.pkl` - StandardScaler (fitted on training data)
  - `label_encoder.pkl` - Label encoder for 27 classes
  - `hist_gb_validation_metrics.json` - Performance metrics
  - `hist_gb_valid_predictions.csv` - All predictions
  - `hist_gb_valid_proba.npy` - Probability scores

**👉 Use this model for production!**

---

## 📁 Other Training Runs

### `hgb_sklearn172/`
- **Date**: October 12, 2025
- **Purpose**: Retrained with sklearn 1.7.2
- **Accuracy**: 98.36%
- **Status**: ✅ Successful (base for hgb_test)
- **Note**: Used to verify sklearn compatibility

### `hgb_fixed/`
- **Date**: October 12, 2025
- **Purpose**: Fixed StandardScaler bug
- **Accuracy**: 88.69%
- **Status**: ⚠️ Old - has sklearn version mismatch
- **Issue**: Trained with sklearn 1.7.1, incompatible with 1.7.2

### `hgb_run1/`
- **Date**: October 10, 2025
- **Purpose**: Initial training attempt
- **Status**: ⚠️ Old - sklearn 1.7.1 compatibility issues
- **Issue**: Model predictions don't match saved predictions

---

## 🗑️ Cleanup Recommendation

You can safely delete old runs:
```bash
rm -rf outputs/hgb_run1 outputs/hgb_fixed outputs/hgb_sklearn172
```

**Keep only**: `outputs/hgb_test/` (current active model)

---

## 📝 Training New Models

To create a new training run:
```bash
python -m src.training.train_hist_gb \
    --top-k 50 \
    --max-iter 200 \
    --output-dir outputs/hgb_YYYY-MM-DD
```

After training, export for API:
```bash
python -m src.utils.export_model \
    --model hgb \
    --model-dir outputs/hgb_YYYY-MM-DD \
    --export-dir api_models
```
