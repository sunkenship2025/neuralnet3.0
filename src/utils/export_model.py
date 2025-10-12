"""
Export trained models for API deployment
Saves model weights, preprocessing artifacts, and metadata
"""

import os
import json
import argparse
from pathlib import Path
import pickle

import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Import model architectures
try:
    from src.training.train_cnn_lstm import TabularCNNLSTM
except ImportError:
    TabularCNNLSTM = None
try:
    from src.training.train_lstm import TabularLSTM
except ImportError:
    TabularLSTM = None
try:
    from src.training.train_cnn import TabularCNN
except ImportError:
    TabularCNN = None
try:
    from src.training.train_mlp import MLP
except ImportError:
    MLP = None


def load_data(train_path: str, veto_path: str, top_k: int = 50):
    """Load and preprocess data to get label encoder and scaler"""
    # Load datasets
    train_df = pd.read_csv(train_path)
    veto_df = pd.read_csv(veto_path)
    
    # Get target column (case-insensitive)
    target_col = None
    for col in train_df.columns:
        if col.lower() in ['device', 'label']:
            target_col = col
            break
    
    if target_col is None:
        raise ValueError(f"No target column found (expected 'device' or 'label'). Found columns: {list(train_df.columns)}")
    
    # Get top features (check for column name)
    feature_col = 'Variable_Name' if 'Variable_Name' in veto_df.columns else 'Feature'
    score_col = 'Votes' if 'Votes' in veto_df.columns else 'Average'
    top_features = veto_df.nlargest(top_k, score_col)[feature_col].tolist()
    
    # Check features exist
    missing_features = set(top_features) - set(train_df.columns)
    if missing_features:
        print(f"Warning: {len(missing_features)} features missing from training data")
        top_features = [f for f in top_features if f in train_df.columns]
    
    X_train = train_df[top_features].values
    y_train = train_df[target_col].values
    
    # Fit preprocessing
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    
    return {
        'scaler': scaler,
        'label_encoder': label_encoder,
        'top_features': top_features,
        'num_classes': len(label_encoder.classes_),
        'class_names': label_encoder.classes_.tolist(),
        'input_dim': len(top_features)
    }


def export_model(
    model_name: str,
    model_dir: str,
    export_dir: str = "api_models",
    train_path: str = "Aalto_train_IoTDevID (1).csv",
    veto_path: str = "veto_average_results (1).csv",
    top_k: int = 50
):
    """
    Export a trained model with all artifacts needed for inference
    
    Args:
        model_name: Name of model (cnn_lstm, lstm, cnn, mlp)
        model_dir: Directory containing trained model outputs
        export_dir: Directory to save exported artifacts
        train_path: Path to training data for preprocessing
        veto_path: Path to feature ranking file
        top_k: Number of top features to use
    """
    
    # Create export directory
    export_path = Path(export_dir) / model_name
    export_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📦 Exporting {model_name} model...")
    
    # Load preprocessing artifacts
    print("  → Loading preprocessing artifacts...")
    preprocessing = load_data(train_path, veto_path, top_k)
    
    # DON'T save preprocessing yet - need to fix feature order for HGB first!
    
    # Load and save model weights based on type
    model_dir_path = Path(model_dir)
    
    if model_name == "cnn_lstm":
        # Load metrics to get hyperparameters
        metrics_path = model_dir_path / "cnn_lstm_validation_metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Metrics not found: {metrics_path}")
        
        # For this to work, we need to save model state during training
        # Check if model weights exist
        weights_path = model_dir_path / "cnn_lstm_best.pt"
        if not weights_path.exists():
            print(f"  ⚠ Warning: Model weights not found at {weights_path}")
            print(f"  → You need to save model.state_dict() during training")
            # Save placeholder config
            config = {
                'model_type': 'cnn_lstm',
                'input_dim': preprocessing['input_dim'],
                'num_classes': preprocessing['num_classes'],
                'conv_channels': [64, 128, 256],
                'kernel_sizes': [7, 5, 3],
                'lstm_hidden': 128,
                'lstm_layers': 2,
                'dropout': 0.3
            }
        else:
            # Load model state
            state_dict = torch.load(weights_path, map_location='cpu')
            torch.save(state_dict, export_path / "model_weights.pt")
            print(f"  ✓ Saved model_weights.pt")
    
    elif model_name == "hgb":
        # For sklearn models, copy the saved pickle
        model_path = model_dir_path / "hist_gb_model.pkl"
        if model_path.exists():
            import shutil
            shutil.copy(model_path, export_path / "model.pkl")
            print(f"  ✓ Saved model.pkl")
            
            # Load the actual feature list from training metadata
            metrics_path = model_dir_path / "hist_gb_validation_metrics.json"
            if metrics_path.exists():
                with open(metrics_path, 'r') as f:
                    training_metadata = json.load(f)
                
                # Override the feature list with the one used during training
                preprocessing['top_features'] = training_metadata['config']['feature_names']
                preprocessing['input_dim'] = len(preprocessing['top_features'])
                print(f"  ✓ Loaded feature list from training metadata ({len(preprocessing['top_features'])} features)")
            
            # Also copy scaler and label encoder if they exist
            scaler_path = model_dir_path / "scaler.pkl"
            encoder_path = model_dir_path / "label_encoder.pkl"
            if scaler_path.exists():
                shutil.copy(scaler_path, export_path / "scaler.pkl")
                print(f"  ✓ Copied scaler.pkl")
            if encoder_path.exists():
                shutil.copy(encoder_path, export_path / "label_encoder.pkl")
                print(f"  ✓ Copied label_encoder.pkl")
        else:
            print(f"  ⚠ Warning: Model not found at {model_path}")
    
    # NOW save preprocessing with corrected features (for HGB)
    with open(export_path / "preprocessing.pkl", "wb") as f:
        pickle.dump(preprocessing, f)
    print(f"  ✓ Saved preprocessing.pkl")
    
    # Save metadata
    metadata = {
        'model_name': model_name,
        'top_k_features': top_k,
        'num_classes': preprocessing['num_classes'],
        'class_names': preprocessing['class_names'],
        'input_dim': preprocessing['input_dim'],
        'feature_names': preprocessing['top_features']
    }
    
    with open(export_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ Saved metadata.json")
    
    print(f"✅ Model exported to {export_path}")
    print(f"   - preprocessing.pkl ({preprocessing['num_classes']} classes)")
    print(f"   - metadata.json")
    print(f"   - Classes: {', '.join(preprocessing['class_names'][:5])}...")


def main():
    parser = argparse.ArgumentParser(description="Export models for API deployment")
    parser.add_argument("--model", type=str, default="cnn_lstm",
                       choices=["cnn_lstm", "lstm", "cnn", "mlp", "hgb"],
                       help="Model to export")
    parser.add_argument("--model-dir", type=str, required=True,
                       help="Directory containing trained model outputs")
    parser.add_argument("--export-dir", type=str, default="api_models",
                       help="Directory to save exported artifacts")
    parser.add_argument("--train-path", type=str, default="data/Aalto_train_IoTDevID (1).csv",
                       help="Path to training data")
    parser.add_argument("--veto-path", type=str, default="data/veto_average_results (1).csv",
                       help="Path to feature ranking file")
    parser.add_argument("--top-k", type=int, default=50,
                       help="Number of top features")
    
    args = parser.parse_args()
    
    export_model(
        model_name=args.model,
        model_dir=args.model_dir,
        export_dir=args.export_dir,
        train_path=args.train_path,
        veto_path=args.veto_path,
        top_k=args.top_k
    )


if __name__ == "__main__":
    main()
