#!/usr/bin/env python3
"""
Train a 1D convolutional neural network on the Aalto IoT Device Identification dataset.

This script mirrors the preprocessing, feature selection, and metric reporting used in
`train_mlp.py`, but replaces the classifier with a small Conv1D stack that can capture local
feature interactions along the ranked feature axis. The model automatically leverages Apple's
Metal (MPS) backend when available.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence

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
    conv_channels: Sequence[int]
    kernel_size: int
    dropout: float
    weight_decay: float
    label_smoothing: float
    max_grad_norm: Optional[float]
    output_dir: str
    seed: int


class TabularCNN(nn.Module):
    def __init__(
        self,
        input_length: int,
        num_classes: int,
        conv_channels: Sequence[int],
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if input_length <= 0:
            raise ValueError("input_length must be positive")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size should be odd to preserve length with same padding")

        layers: List[nn.Module] = []
        in_channels = 1
        current_length = input_length
        for idx, out_channels in enumerate(conv_channels):
            layers.append(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            if idx < len(conv_channels) - 1 and current_length >= 4:
                layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
                current_length = math.floor((current_length + 1) / 2)

            in_channels = out_channels

        self.feature_extractor = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        hidden_dim = max(128, in_channels // 2)
        classifier_layers: List[nn.Module] = [
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            classifier_layers.append(nn.Dropout(dropout))
        classifier_layers.append(nn.Linear(hidden_dim, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.feature_extractor(x)
        z = self.global_pool(z).squeeze(-1)
        return self.classifier(z)


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Train a 1D CNN classifier on IoT device data")
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
    default=40,
        help="Number of top-ranked features to use as model inputs",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument(
    "--conv-channels",
    type=int,
    nargs="+",
    default=[64, 128, 128],
        help="Output channels for each Conv1d block",
    )
    parser.add_argument(
        "--kernel-size",
        type=int,
        default=3,
        help="Kernel size for Conv1d blocks (must be odd to keep length with padding)",
    )
    parser.add_argument(
    "--dropout",
    type=float,
    default=0.2,
        help="Dropout probability applied after each block and dense layer",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="L2 weight decay for optimizer",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.05,
        help="Label smoothing factor for cross-entropy loss",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm (set <=0 to disable)",
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
        conv_channels=args.conv_channels,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        max_grad_norm=args.max_grad_norm if args.max_grad_norm > 0 else None,
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
    max_grad_norm: Optional[float] = None,
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
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
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

    train_features = train_features.unsqueeze(1)
    valid_features = valid_features.unsqueeze(1)

    train_dataset = TensorDataset(train_features, train_labels)
    valid_dataset = TensorDataset(valid_features, valid_labels)

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=False
    )
    train_eval_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=False, drop_last=False
    )
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False)

    model = TabularCNN(
        input_length=train_features.shape[-1],
        num_classes=len(label_encoder.classes_),
        conv_channels=config.conv_channels,
        kernel_size=config.kernel_size,
        dropout=config.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=max(0.0, config.label_smoothing))
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=max(1, config.epochs // 3), gamma=0.5
    )

    history = []
    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, max_grad_norm=config.max_grad_norm
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

        scheduler.step()

    os.makedirs(config.output_dir, exist_ok=True)
    metrics_path = os.path.join(config.output_dir, "cnn_validation_metrics.json")
    report = {
        "config": {
            "top_k_features": config.top_k_features,
            "conv_channels": list(config.conv_channels),
            "kernel_size": config.kernel_size,
            "dropout": config.dropout,
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "label_smoothing": config.label_smoothing,
            "max_grad_norm": config.max_grad_norm,
            "device": device.type,
            "num_features": train_features.shape[-1],
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
