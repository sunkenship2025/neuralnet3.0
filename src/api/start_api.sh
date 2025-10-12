#!/bin/bash
# Quick Start Script for IoT Device Identification API

set -e

echo "🚀 IoT Device Identification API - Quick Start"
echo "=============================================="
echo ""

# Go to project root
cd "$(dirname "$0")/../.."
PROJECT_ROOT=$(pwd)

# Check if venv exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "   Please run: python -m venv .venv"
    exit 1
fi

# Activate venv
echo "📦 Activating virtual environment..."
# shellcheck disable=SC1091
source .venv/bin/activate

# Check if FastAPI is installed
if ! python -c "import fastapi" 2>/dev/null; then
    echo "📥 Installing API dependencies..."
    pip install -q fastapi 'uvicorn[standard]' python-multipart pydantic
fi

# Check if required files exist
echo "🔍 Checking required files..."
if [ ! -f "data/veto_average_results (1).csv" ]; then
    echo "❌ Feature ranking file not found: data/veto_average_results (1).csv"
    exit 1
fi

# Check if models are exported
if [ ! -d "api_models" ]; then
    echo "⚠️  No exported models found. You need to export models first:"
    echo ""
    echo "   # Export CNN+LSTM model (recommended)"
    echo "   python export_model.py --model cnn_lstm --model-dir outputs/cnn_lstm_full"
    echo ""
    echo "   # Or export HGB model (fastest)"
    echo "   python export_model.py --model hgb --model-dir outputs/hgb_run1"
    echo ""
    echo "   Then re-run this script."
    exit 1
fi

echo "✅ All checks passed!"
echo ""
echo "🌐 Starting API server..."
echo "   Access at: http://localhost:8000"
echo "   Docs at: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Start the API from project root
cd "$PROJECT_ROOT"
python -m uvicorn src.api.api:app --host 0.0.0.0 --port 8000 --reload
