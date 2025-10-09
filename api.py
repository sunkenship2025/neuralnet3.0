"""
FastAPI REST API for IoT Device Identification
Accepts network traffic features and predicts device types using trained models.
"""

import os
import json
import io
from typing import List, Optional, Dict, Any
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Import model architectures
from train_mlp import TabularMLP
from train_cnn import TabularCNN
from train_lstm import TabularLSTM
from train_cnn_lstm import TabularCNNLSTM
from train_autoencoder import FeedforwardAutoencoder

app = FastAPI(
    title="IoT Device Identification API",
    description="Predict IoT device types from network traffic features",
    version="1.0.0"
)

# Global model cache
MODELS = {}
FEATURE_RANKING = None
LABEL_ENCODER = None


class PredictionRequest(BaseModel):
    """Request model for single or batch predictions"""
    features: List[Dict[str, float]]  # List of feature dictionaries
    top_k_features: int = 50  # Number of top features to use
    model_type: str = "cnn_lstm"  # Which model to use


class PredictionResponse(BaseModel):
    """Response model with predictions"""
    predictions: List[str]  # Device type predictions
    probabilities: List[Dict[str, float]]  # Confidence scores per class
    model_used: str


def load_feature_ranking(veto_path: str = "veto_average_results (1).csv") -> pd.DataFrame:
    """Load and cache feature importance ranking"""
    global FEATURE_RANKING
    if FEATURE_RANKING is None:
        if not os.path.exists(veto_path):
            raise FileNotFoundError(f"Feature ranking file not found: {veto_path}")
        FEATURE_RANKING = pd.read_csv(veto_path)
    return FEATURE_RANKING


def get_top_features(veto_df: pd.DataFrame, top_k: int = 50) -> List[str]:
    """Get top K features from veto ranking"""
    return veto_df.nlargest(top_k, 'Average')['Feature'].tolist()


def load_model(model_type: str, model_dir: str = "api_models"):
    """Load a trained model and its artifacts"""
    if model_type in MODELS:
        return MODELS[model_type]
    
    import pickle
    
    model_path = Path(model_dir) / model_type
    
    if not model_path.exists():
        raise ValueError(f"Model directory not found: {model_path}. Run export_model.py first.")
    
    # Load preprocessing artifacts
    with open(model_path / "preprocessing.pkl", "rb") as f:
        preprocessing = pickle.load(f)
    
    # Load metadata
    with open(model_path / "metadata.json", "r") as f:
        metadata = json.load(f)
    
    model = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model based on type
    if model_type == "cnn_lstm":
        from train_cnn_lstm import TabularCNNLSTM
        model = TabularCNNLSTM(
            input_dim=metadata['input_dim'],
            num_classes=metadata['num_classes'],
            conv_channels=[64, 128, 256],
            kernel_sizes=[7, 5, 3],
            lstm_hidden=128,
            lstm_layers=2,
            dropout=0.3
        )
        weights_path = model_path / "model_weights.pt"
        if weights_path.exists():
            model.load_state_dict(torch.load(weights_path, map_location=device))
            model.to(device)
            model.eval()
    
    elif model_type == "hgb":
        weights_path = model_path / "model.pkl"
        if weights_path.exists():
            with open(weights_path, "rb") as f:
                model = pickle.load(f)
    
    else:
        raise NotImplementedError(f"Model type {model_type} not yet supported")
    
    MODELS[model_type] = {
        'model': model,
        'preprocessing': preprocessing,
        'metadata': metadata,
        'device': device
    }
    
    return MODELS[model_type]


@app.on_event("startup")
async def startup_event():
    """Initialize models and feature rankings on startup"""
    try:
        load_feature_ranking()
        print("✓ Feature ranking loaded")
    except Exception as e:
        print(f"⚠ Warning: Could not load feature ranking: {e}")


@app.get("/")
async def root():
    """API health check"""
    return {
        "status": "online",
        "message": "IoT Device Identification API",
        "available_models": ["mlp", "cnn", "lstm", "cnn_lstm", "hgb"],
        "endpoints": {
            "/predict": "POST - Predict device from features",
            "/predict/csv": "POST - Predict devices from CSV file",
            "/health": "GET - API health status"
        }
    }


@app.get("/health")
async def health():
    """Detailed health check"""
    veto_exists = os.path.exists("veto_average_results (1).csv")
    return {
        "status": "healthy",
        "feature_ranking_loaded": FEATURE_RANKING is not None,
        "veto_file_exists": veto_exists,
        "models_cached": list(MODELS.keys())
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict IoT device types from feature vectors
    
    Example request:
    {
        "features": [
            {"feature1": 0.5, "feature2": 1.2, ...},
            {"feature1": 0.3, "feature2": 0.9, ...}
        ],
        "top_k_features": 50,
        "model_type": "cnn_lstm"
    }
    """
    try:
        # Load model artifacts
        model_artifacts = load_model(request.model_type)
        model = model_artifacts['model']
        preprocessing = model_artifacts['preprocessing']
        metadata = model_artifacts['metadata']
        device = model_artifacts.get('device', torch.device('cpu'))
        
        # Convert features to DataFrame
        df = pd.DataFrame(request.features)
        
        # Get required features
        required_features = preprocessing['top_features']
        
        # Check if required features exist
        missing_features = set(required_features) - set(df.columns)
        if missing_features:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required features: {list(missing_features)[:10]}..."
            )
        
        # Select and order features
        X = df[required_features].values
        
        # Preprocess
        X_scaled = preprocessing['scaler'].transform(X)
        
        # Run inference
        if request.model_type == "hgb":
            # Sklearn model
            y_pred = model.predict(X_scaled)
            y_proba = model.predict_proba(X_scaled)
        else:
            # PyTorch model
            X_tensor = torch.FloatTensor(X_scaled).to(device)
            with torch.no_grad():
                logits = model(X_tensor)
                y_proba = torch.softmax(logits, dim=-1).cpu().numpy()
                y_pred = y_proba.argmax(axis=1)
        
        # Decode predictions
        label_encoder = preprocessing['label_encoder']
        predictions = label_encoder.inverse_transform(y_pred).tolist()
        
        # Format probabilities
        probabilities = []
        for proba in y_proba:
            class_probs = {
                label_encoder.classes_[i]: float(proba[i])
                for i in range(len(label_encoder.classes_))
            }
            # Sort by probability and take top 5
            sorted_probs = dict(sorted(class_probs.items(), key=lambda x: x[1], reverse=True)[:5])
            probabilities.append(sorted_probs)
        
        return PredictionResponse(
            predictions=predictions,
            probabilities=probabilities,
            model_used=request.model_type
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/csv")
async def predict_csv(
    file: UploadFile = File(...),
    model_type: str = "cnn_lstm"
):
    """
    Predict IoT device types from uploaded CSV file
    
    CSV should contain network traffic features (same format as Aalto_test_IoTDevID.csv)
    """
    try:
        # Load model artifacts
        model_artifacts = load_model(model_type)
        model = model_artifacts['model']
        preprocessing = model_artifacts['preprocessing']
        device = model_artifacts.get('device', torch.device('cpu'))
        
        # Read CSV
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Get required features
        required_features = preprocessing['top_features']
        
        # Check for required features
        missing_features = set(required_features) - set(df.columns)
        if missing_features:
            raise HTTPException(
                status_code=400,
                detail=f"CSV missing required features: {list(missing_features)[:10]}..."
            )
        
        # Select features
        X = df[required_features].values
        
        # Preprocess
        X_scaled = preprocessing['scaler'].transform(X)
        
        # Run inference
        if model_type == "hgb":
            y_pred = model.predict(X_scaled)
            y_proba = model.predict_proba(X_scaled)
        else:
            X_tensor = torch.FloatTensor(X_scaled).to(device)
            with torch.no_grad():
                logits = model(X_tensor)
                y_proba = torch.softmax(logits, dim=-1).cpu().numpy()
                y_pred = y_proba.argmax(axis=1)
        
        # Decode predictions
        label_encoder = preprocessing['label_encoder']
        predictions = label_encoder.inverse_transform(y_pred).tolist()
        
        # Get confidence scores (max probability per sample)
        confidences = y_proba.max(axis=1).tolist()
        
        # Get class distribution
        unique, counts = np.unique(predictions, return_counts=True)
        class_distribution = dict(zip(unique.tolist(), counts.tolist()))
        
        return {
            "num_samples": len(df),
            "predictions": predictions,
            "confidences": confidences,
            "avg_confidence": float(np.mean(confidences)),
            "min_confidence": float(np.min(confidences)),
            "max_confidence": float(np.max(confidences)),
            "class_distribution": class_distribution,
            "model_used": model_type
        }
        
    except pd.errors.ParserError:
        raise HTTPException(status_code=400, detail="Invalid CSV file")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/features")
async def get_features(top_k: int = 50):
    """Get the top K most important features"""
    try:
        veto_df = load_feature_ranking()
        top_features = get_top_features(veto_df, top_k)
        return {
            "top_k": top_k,
            "features": top_features,
            "importance_scores": veto_df.nlargest(top_k, 'Average')[['Feature', 'Average']].to_dict('records')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
