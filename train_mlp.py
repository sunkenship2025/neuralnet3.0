#!/usr/bin/env python3
"""
Train a simple multi-layer perceptron on the Aalto IoT Device Identification dataset.

This script loads the feature rankings produced in `veto_average_results (1).csv`, selects the
Top-K features, and trains a neural network using PyTorch with MPS acceleration on Apple Silicon
when available. Metrics are computed on the validation split for comparison against existing
DecisionTree baselines.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset


DEFAULT_TRAIN_PATH = "Aalto_train_IoTDevID (1).csv"
DEFAULT_VALID_PATH = "Aalto_test_IoTDevID (1).csv"
DEFAULT_FEATURE_RANKING_PATH = "veto_average_results (1).csv"
DEFAULT_OUTPUT_DIR = "outputs"


@dataclass
class TrainingConfig:
    train_csv: str
    valid_csv: str
    feature_csv: str
    top_k_features: int
    batch_size: int
    epochs: int
    learning_rate: float
    hidden_dims: Sequence[int]
    weight_decay: float
    output_dir: str
    seed: int


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], num_classes: int) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.2))
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Train an MLP classifier on IoT device data")
    parser.add_argument("--train-csv", default=DEFAULT_TRAIN_PATH, help="Path to training CSV file")
    parser.add_argument(
        "--valid-csv", default=DEFAULT_VALID_PATH, help="Path to validation CSV file"
    )
    parser.add_argument(
        "--feature-ranking-csv",
        default=DEFAULT_FEATURE_RANKING_PATH,
        help="Path to Veto aggregated feature ranking CSV",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of top-ranked features to use as model inputs",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[128, 64],
        help="Hidden layer sizes for the MLP",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="L2 weight decay for optimizer",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to store metrics and artifacts",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    return TrainingConfig(
        train_csv=args.train_csv,
        valid_csv=args.valid_csv,
        feature_csv=args.feature_ranking_csv,
        top_k_features=args.top_k,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        hidden_dims=args.hidden_dims,
        weight_decay=args.weight_decay,
        output_dir=args.output_dir,
        seed=args.seed,
    )


def select_features(feature_csv: str, top_k: int) -> List[str]:
    ranking_df = pd.read_csv(feature_csv)
    if "Votes" not in ranking_df.columns:
        raise ValueError("Feature ranking CSV must contain a 'Votes' column")
    if "Variable_Name" not in ranking_df.columns:
        raise ValueError("Feature ranking CSV must contain a 'Variable_Name' column")

    filtered = ranking_df[ranking_df["Votes"] > 0].sort_values("Votes", ascending=False)
    features = filtered["Variable_Name"].tolist()
    if not features:
        raise ValueError("No features with positive vote scores found in ranking CSV")
    if top_k > len(features):
        print(
            f"[warn] Requested top_k={top_k} but only {len(features)} features available with votes. "
            f"Using top_k={len(features)}."
        )
        top_k = len(features)
    selected = features[:top_k]
    print(f"[info] Selected top {len(selected)} features: {selected}")
    return selected


def prepare_tensors(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    label_encoder: LabelEncoder,
    scaler: StandardScaler,
    fit_transform: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing expected feature columns in dataframe: {missing_cols}")

    features_np = df[feature_cols].astype(np.float32).fillna(0.0).values
    labels_np = df["Label"].astype(str).values

    if fit_transform:
        features_np = scaler.fit_transform(features_np)
        label_encoder.fit(labels_np)
    else:
        features_np = scaler.transform(features_np)

    labels_encoded = label_encoder.transform(labels_np)

    features_tensor = torch.tensor(features_np, dtype=torch.float32)
    labels_tensor = torch.tensor(labels_encoded, dtype=torch.long)
    return features_tensor, labels_tensor


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        print("[info] Using Apple Metal Performance Shaders (MPS) backend")
        return torch.device("mps")
    if torch.cuda.is_available():
        print("[info] Using CUDA GPU backend")
        return torch.device("cuda")
    print("[info] Falling back to CPU")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_x.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == batch_y).sum().item()
        total += batch_x.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    preds: List[int] = []
    for batch_x, _ in loader:
        batch_x = batch_x.to(device)
        logits = model(batch_x)
        preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds, axis=0)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    kappa = cohen_kappa_score(y_true, y_pred)
    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "cohen_kappa": kappa,
    }


def main() -> None:
    config = parse_args()
    set_seed(config.seed)

    device = get_device()

    features = select_features(config.feature_csv, config.top_k_features)

    train_df = pd.read_csv(config.train_csv, low_memory=False)
    valid_df = pd.read_csv(config.valid_csv, low_memory=False)
    for col in ["Label"]:
        if col not in train_df.columns or col not in valid_df.columns:
            raise KeyError(f"Expected column '{col}' not found in both train and validation CSVs")

    feature_cols = [f for f in features if f in train_df.columns]
    missing_from_train = [f for f in features if f not in train_df.columns]
    if missing_from_train:
        print(
            f"[warn] Dropping {len(missing_from_train)} features missing from training data: "
            f"{missing_from_train}"
        )
    if not feature_cols:
        raise ValueError("No overlapping features between ranking file and training CSV")

    label_encoder = LabelEncoder()
    scaler = StandardScaler()

    train_features, train_labels = prepare_tensors(
        train_df, feature_cols, label_encoder, scaler, fit_transform=True
    )
    valid_features, valid_labels = prepare_tensors(
        valid_df, feature_cols, label_encoder, scaler, fit_transform=False
    )

    train_dataset = TensorDataset(train_features, train_labels)
    valid_dataset = TensorDataset(valid_features, valid_labels)

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=False
    )
    train_eval_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=False, drop_last=False
    )
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False)

    model = MLP(
        input_dim=train_features.shape[1],
        hidden_dims=config.hidden_dims,
        num_classes=len(label_encoder.classes_),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    history = []
    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        train_preds = predict(model, train_eval_loader, device)
        train_metrics = compute_metrics(train_labels.numpy(), train_preds)

        valid_preds = predict(model, valid_loader, device)
        valid_metrics = compute_metrics(valid_labels.numpy(), valid_preds)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "train_metrics": train_metrics,
                "valid_metrics": valid_metrics,
            }
        )

        print(
            f"Epoch {epoch:03d}/{config.epochs}: loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} val_acc={valid_metrics['accuracy']:.4f} "
            f"val_f1_macro={valid_metrics['f1_macro']:.4f}"
        )

    os.makedirs(config.output_dir, exist_ok=True)
    metrics_path = os.path.join(config.output_dir, "mlp_validation_metrics.json")
    report = {
        "config": {
            "top_k_features": config.top_k_features,
            "hidden_dims": list(config.hidden_dims),
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "device": device.type,
            "num_features": train_features.shape[1],
            "classes": label_encoder.classes_.tolist(),
        },
        "history": history,
        "final_validation_metrics": history[-1]["valid_metrics"],
    }
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"[info] Saved validation metrics to {metrics_path}")


if __name__ == "__main__":
    main()
