#!/bin/bash
# Launcher script for training models

set -e

echo "🎓 IoT Device Identification - Model Training"
echo "=============================================="
echo ""

# Check if venv exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "   Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate venv
source .venv/bin/activate

# Install requirements if needed
if ! python -c "import sklearn, torch, pandas" 2>/dev/null; then
    echo "📥 Installing dependencies..."
    pip install -q -r requirements.txt
fi

echo ""
echo "Available training scripts:"
echo "  1. Histogram Gradient Boosting (RECOMMENDED - 98.36% accuracy)"
echo "  2. CNN + LSTM"
echo "  3. LSTM"
echo "  4. CNN"
echo "  5. MLP"
echo "  6. Autoencoder"
echo ""
read -p "Select model (1-6): " choice

case $choice in
    1)
        echo "Training Histogram Gradient Boosting..."
        python -m src.training.train_hist_gb --top-k 50 --max-iter 200 --output-dir outputs/hgb_latest
        ;;
    2)
        echo "Training CNN+LSTM..."
        python -m src.training.train_cnn_lstm --top-k 50 --output-dir outputs/cnn_lstm_latest
        ;;
    3)
        echo "Training LSTM..."
        python -m src.training.train_lstm --top-k 50 --output-dir outputs/lstm_latest
        ;;
    4)
        echo "Training CNN..."
        python -m src.training.train_cnn --top-k 50 --output-dir outputs/cnn_latest
        ;;
    5)
        echo "Training MLP..."
        python -m src.training.train_mlp --top-k 50 --output-dir outputs/mlp_latest
        ;;
    6)
        echo "Training Autoencoder..."
        python -m src.training.train_autoencoder --top-k 50 --output-dir outputs/autoencoder_latest
        ;;
    *)
        echo "Invalid choice!"
        exit 1
        ;;
esac

echo ""
echo "✅ Training complete!"
echo ""
echo "Next steps:"
echo "  1. Export model: python -m src.utils.export_model --model hgb --model-dir outputs/hgb_latest"
echo "  2. Start API: ./start_api.sh"
