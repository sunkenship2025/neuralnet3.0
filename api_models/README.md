# API Models Directory

This directory contains exported models ready for deployment in the API.

## ⭐ ACTIVE MODEL

### `hgb/` - Histogram Gradient Boosting
- **Source**: `outputs/hgb_test/`
- **Accuracy**: 98.36%
- **Status**: ✅ **CURRENTLY SERVING**
- **Classes**: 27 IoT device types
- **Features**: 48 top-ranked network traffic features

**Files:**
- `model.pkl` (30 MB) - Trained HGB classifier
- `scaler.pkl` - StandardScaler fitted on training data
- `label_encoder.pkl` - Label encoder for device names
- `preprocessing.pkl` - Feature list and metadata (48 features in correct order!)
- `metadata.json` - Model configuration and class names

**👉 This is what the API uses for predictions**

---

## 🔍 How It Works

1. API loads model from `api_models/hgb/`
2. Incoming CSV data is read
3. Features are selected in the **exact order** from `preprocessing.pkl`
4. NaN values filled with 0.0
5. Data scaled using `scaler.pkl`
6. Model predicts device class
7. Results returned with confidence scores

---

## 🚀 API Usage

```bash
# Start API
./start_api.sh

# Test prediction
curl -X POST "http://localhost:8000/predict/csv" \
    -F "file=@your_data.csv" \
    -F "model_type=hgb"
```

---

## 📝 Exporting New Models

When you train a new model and want to deploy it:

```bash
python -m src.utils.export_model \
    --model hgb \
    --model-dir outputs/your_new_model \
    --export-dir api_models \
    --top-k 50
```

This will **replace** the current model in `api_models/hgb/`.

**Important**: Restart the API after exporting a new model!

---

## ⚠️ Critical Notes

1. **Feature Order Matters**: The preprocessing.pkl contains features in the exact order used during training. Changing this breaks predictions!

2. **Scaler is Tied to Model**: The scaler.pkl must be from the same training run as model.pkl

3. **sklearn Version**: Model was trained with sklearn 1.7.2 - use the same version

4. **File Size**: Model files are large (30MB) - excluded from git

---

## 🧹 Backup Old Models

Before exporting a new model, backup the current one:
```bash
cp -r api_models/hgb api_models/hgb_backup_$(date +%Y%m%d)
```
