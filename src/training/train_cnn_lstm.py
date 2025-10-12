#!/usr/bin/env python3
"""Train a hybrid CNN + LSTM classifier on the Aalto IoT Device Identification dataset.

This script combines the local pattern extraction of the existing Conv1d baseline with the
sequence modelling capabilities of the LSTM trainer. The ranked feature vector is first processed
by several convolutional blocks and the resulting sequence of feature maps is then fed into an
LSTM head before classification. The preprocessing, metrics, and artifact logging match the other
trainers so the outputs can be compared or ensembled directly.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Optional, Sequence

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
    lr_scheduler: str
    lr_step_size: int
    lr_step_gamma: float
    lr_cosine_t0: int
    lr_cosine_t_mult: int
    conv_channels: Sequence[int]
    kernel_size: int
    conv_dropout: float
    lstm_hidden_size: int
    lstm_num_layers: int
    lstm_dropout: float
    bidirectional: bool
    weight_decay: float
    label_smoothing: float
    max_grad_norm: Optional[float]
    output_dir: str
    seed: int


class TabularCNNLSTM(nn.Module):
    def __init__(
        self,
        input_length: int,
        num_classes: int,
        conv_channels: Sequence[int],
        kernel_size: int,
        conv_dropout: float,
        lstm_hidden_size: int,
        lstm_num_layers: int,
        lstm_dropout: float,
        bidirectional: bool,
    ) -> None:
        super().__init__()
        if input_length <= 0:
            raise ValueError("input_length must be positive")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size should be odd to preserve length with padding")
        if not conv_channels:
            raise ValueError("conv_channels must contain at least one entry")
        if lstm_hidden_size <= 0:
            raise ValueError("lstm_hidden_size must be positive")
        if lstm_num_layers <= 0:
            raise ValueError("lstm_num_layers must be positive")

        conv_layers: list[nn.Module] = []
        in_channels = 1
        current_length = input_length
        for idx, out_channels in enumerate(conv_channels):
            conv_layers.append(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    bias=False,
                )
            )
            conv_layers.append(nn.BatchNorm1d(out_channels))
            conv_layers.append(nn.ReLU(inplace=True))
            if conv_dropout > 0:
                conv_layers.append(nn.Dropout(conv_dropout))

            if idx < len(conv_channels) - 1 and current_length >= 4:
                conv_layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
                current_length = math.floor((current_length + 1) / 2)

            in_channels = out_channels

        if current_length <= 0:
            raise ValueError("Convolutional stack reduced the sequence length to zero")

        self.feature_extractor = nn.Sequential(*conv_layers)
        self.sequence_length = current_length
        self.lstm_input_size = in_channels
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = lstm_hidden_size
        self.num_layers = lstm_num_layers

        lstm_inner_dropout = lstm_dropout if lstm_num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=lstm_inner_dropout,
            bidirectional=bidirectional,
        )
        self.post_lstm_dropout = nn.Dropout(lstm_dropout) if lstm_dropout > 0 else nn.Identity()

        classifier_in = lstm_hidden_size * self.num_directions
        hidden_dim = max(128, classifier_in // 2)
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(lstm_dropout) if lstm_dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() != 3:
            raise ValueError("Expected input tensor of shape (batch, seq_len) or (batch, channels, seq_len)")

        conv_out = self.feature_extractor(x)
        seq = conv_out.permute(0, 2, 1)  # (batch, seq_len, channels)
        _, (hidden, _) = self.lstm(seq)
        hidden = hidden.view(self.num_layers, self.num_directions, x.size(0), self.hidden_size)
        last_layer = hidden[-1]
        last_hidden = last_layer.transpose(0, 1).reshape(x.size(0), -1)
        last_hidden = self.post_lstm_dropout(last_hidden)
        return self.classifier(last_hidden)


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(
        description="Train a hybrid CNN+LSTM classifier on IoT device data"
    )
    parser.add_argument("--train-csv", default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--valid-csv", default=DEFAULT_VALID_PATH)
    parser.add_argument("--feature-ranking-csv", default=DEFAULT_FEATURE_RANKING_PATH)
    parser.add_argument("--top-k", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument(
        "--lr-scheduler", choices=["none", "step", "cosine"], default="step"
    )
    parser.add_argument("--lr-step-size", type=int, default=10)
    parser.add_argument("--lr-step-gamma", type=float, default=0.5)
    parser.add_argument("--lr-cosine-t0", type=int, default=10)
    parser.add_argument("--lr-cosine-t-mult", type=int, default=2)
    parser.add_argument(
        "--conv-channels",
        type=int,
        nargs="+",
        default=[64, 128],
        help="Output channels for each Conv1d block",
    )
    parser.add_argument(
        "--kernel-size",
        type=int,
        default=3,
        help="Kernel size for Conv1d layers (must be odd)",
    )
    parser.add_argument(
        "--conv-dropout",
        type=float,
        default=0.2,
        help="Dropout probability applied after each convolutional block",
    )
    parser.add_argument("--lstm-hidden-size", type=int, default=128)
    parser.add_argument("--lstm-num-layers", type=int, default=2)
    parser.add_argument(
        "--lstm-dropout",
        type=float,
        default=0.3,
        help="Dropout after the LSTM head and within stacked LSTM layers",
    )
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm (set <=0 to disable)",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    return TrainingConfig(
        train_csv=args.train_csv,
        valid_csv=args.valid_csv,
        feature_csv=args.feature_ranking_csv,
        top_k_features=args.top_k,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler=args.lr_scheduler,
        lr_step_size=max(1, args.lr_step_size),
        lr_step_gamma=args.lr_step_gamma,
        lr_cosine_t0=max(1, args.lr_cosine_t0),
        lr_cosine_t_mult=max(1, args.lr_cosine_t_mult),
        conv_channels=args.conv_channels,
        kernel_size=args.kernel_size,
        conv_dropout=max(0.0, args.conv_dropout),
        lstm_hidden_size=args.lstm_hidden_size,
        lstm_num_layers=args.lstm_num_layers,
        lstm_dropout=max(0.0, args.lstm_dropout),
        bidirectional=args.bidirectional,
        weight_decay=args.weight_decay,
        label_smoothing=max(0.0, args.label_smoothing),
        max_grad_norm=args.max_grad_norm if args.max_grad_norm > 0 else None,
        output_dir=args.output_dir,
        seed=args.seed,
    )


def select_features(feature_csv: str, top_k: int) -> list[str]:
    ranking_df = pd.read_csv(feature_csv)
    if {"Votes", "Variable_Name"}.difference(ranking_df.columns):
        raise ValueError("Feature ranking CSV must contain 'Votes' and 'Variable_Name' columns")

    filtered = ranking_df[ranking_df["Votes"] > 0].sort_values("Votes", ascending=False)
    features = filtered["Variable_Name"].tolist()
    if not features:
        raise ValueError("No features with positive vote scores found in ranking CSV")

    if top_k > len(features):
        print(
            f"[warn] Requested top_k={top_k} but only {len(features)} features available. Using {len(features)}."
        )
        top_k = len(features)
    selected = features[:top_k]
    print(f"[info] Selected top {len(selected)} features from ranking")
    return selected


def ensure_label_overlap(train_df: pd.DataFrame, valid_df: pd.DataFrame) -> None:
    train_labels = set(train_df["Label"].astype(str).unique())
    valid_labels = set(valid_df["Label"].astype(str).unique())
    unseen = sorted(valid_labels - train_labels)
    if unseen:
        raise ValueError(
            "Validation set contains labels absent from training data: "
            f"{unseen}. Align splits or filter unseen classes before training."
        )


def prepare_tensors(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    label_encoder: LabelEncoder,
    scaler: StandardScaler,
    fit_transform: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if "Label" not in df.columns:
        raise KeyError("Expected 'Label' column in dataset")

    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing expected feature columns: {missing_cols}")

    features_np = df[feature_cols].astype(np.float32).fillna(0.0).values
    labels_np = df["Label"].astype(str).values

    if fit_transform:
        features_np = scaler.fit_transform(features_np)
        label_encoder.fit(labels_np)
    else:
        features_np = scaler.transform(features_np)

    features_tensor = torch.tensor(features_np, dtype=torch.float32)
    labels_tensor = torch.tensor(label_encoder.transform(labels_np), dtype=torch.long)
    return features_tensor, labels_tensor


def get_device() -> torch.device:
    if torch.cuda.is_available():
        print("[info] Using CUDA GPU backend")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        print("[info] Using Apple Metal Performance Shaders (MPS) backend")
        return torch.device("mps")
    print("[info] Falling back to CPU")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_scheduler(config: TrainingConfig, optimizer: torch.optim.Optimizer):
    if config.lr_scheduler == "none":
        return None
    if config.lr_scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(1, config.lr_step_size),
            gamma=config.lr_step_gamma,
        )
    if config.lr_scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=max(1, config.lr_cosine_t0),
            T_mult=max(1, config.lr_cosine_t_mult),
        )
    raise ValueError(f"Unsupported lr_scheduler option: {config.lr_scheduler}")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_grad_norm: Optional[float],
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)
        total_correct += (logits.argmax(dim=1) == batch_y).sum().item()
        total_samples += batch_x.size(0)

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    preds: list[np.ndarray] = []
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
    ensure_label_overlap(train_df, valid_df)

    feature_cols = [f for f in features if f in train_df.columns]
    missing_train = [f for f in features if f not in train_df.columns]
    if missing_train:
        print(
            f"[warn] Dropping {len(missing_train)} ranked features missing from training data: "
            f"{missing_train}"
        )
    missing_valid = [f for f in feature_cols if f not in valid_df.columns]
    if missing_valid:
        print(
            f"[warn] Dropping {len(missing_valid)} features missing from validation data: "
            f"{missing_valid}"
        )
        feature_cols = [f for f in feature_cols if f in valid_df.columns]
    if not feature_cols:
        raise ValueError("No overlapping features between ranking file and datasets")

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

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    eval_train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False)

    model = TabularCNNLSTM(
        input_length=train_features.shape[1],
        num_classes=len(label_encoder.classes_),
        conv_channels=config.conv_channels,
        kernel_size=config.kernel_size,
        conv_dropout=config.conv_dropout,
        lstm_hidden_size=config.lstm_hidden_size,
        lstm_num_layers=config.lstm_num_layers,
        lstm_dropout=config.lstm_dropout,
        bidirectional=config.bidirectional,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = build_scheduler(config, optimizer)

    history: list[dict] = []
    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            config.max_grad_norm,
        )

        train_preds = predict(model, eval_train_loader, device)
        valid_preds = predict(model, valid_loader, device)

        train_metrics = compute_metrics(train_labels.numpy(), train_preds)
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

        if scheduler is not None:
            scheduler.step()

    os.makedirs(config.output_dir, exist_ok=True)
    metrics_path = os.path.join(config.output_dir, "cnn_lstm_validation_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "config": {
                    "top_k_features": config.top_k_features,
                    "conv_channels": list(config.conv_channels),
                    "kernel_size": config.kernel_size,
                    "conv_dropout": config.conv_dropout,
                    "lstm_hidden_size": config.lstm_hidden_size,
                    "lstm_num_layers": config.lstm_num_layers,
                    "lstm_dropout": config.lstm_dropout,
                    "bidirectional": config.bidirectional,
                    "batch_size": config.batch_size,
                    "epochs": config.epochs,
                    "learning_rate": config.learning_rate,
                    "lr_scheduler": config.lr_scheduler,
                    "lr_step_size": config.lr_step_size,
                    "lr_step_gamma": config.lr_step_gamma,
                    "lr_cosine_t0": config.lr_cosine_t0,
                    "lr_cosine_t_mult": config.lr_cosine_t_mult,
                    "weight_decay": config.weight_decay,
                    "label_smoothing": config.label_smoothing,
                    "max_grad_norm": config.max_grad_norm,
                    "device": device.type,
                    "num_features": train_features.shape[1],
                    "conv_sequence_length": model.sequence_length,
                    "feature_names": list(feature_cols),
                    "classes": label_encoder.classes_.tolist(),
                },
                "history": history,
                "final_validation_metrics": history[-1]["valid_metrics"],
            },
            fh,
            indent=2,
        )
    print(f"[info] Saved validation metrics to {metrics_path}")


if __name__ == "__main__":
    main()
