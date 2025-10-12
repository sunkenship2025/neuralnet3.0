#!/usr/bin/env python3
"""Train an LSTM-based classifier on the Aalto IoT Device Identification dataset.

This mirrors the preprocessing/metrics pipeline from the existing trainers but swaps the classifier
for a small LSTM stack that treats the ranked feature vector as a 1D sequence. Although the features
are not temporal, the sequential inductive bias can capture interactions between neighbouring ranked
features.
"""
from __future__ import annotations

import argparse
import json
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
    hidden_size: int
    num_layers: int
    dropout: float
    bidirectional: bool
    weight_decay: float
    label_smoothing: float
    max_grad_norm: Optional[float]
    output_dir: str
    seed: int


class TabularLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_classes: int,
    ) -> None:
        super().__init__()
        if input_size <= 0:
            raise ValueError("input_size must be positive")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        self.hidden_size = hidden_size
        self.num_directions = 2 if bidirectional else 1

        # For num_layers==1 PyTorch ignores dropout, so we apply dropout manually afterwards.
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=bidirectional,
        )
        self.post_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        classifier_in = hidden_size * self.num_directions
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, max(128, classifier_in // 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(max(128, classifier_in // 2), num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        _, (hidden, _) = self.lstm(x)
        # hidden: (num_layers * directions, batch, hidden_size)
        hidden = hidden.view(self.lstm.num_layers, self.num_directions, x.size(0), self.hidden_size)
        last_layer = hidden[-1]
        last_hidden = last_layer.transpose(0, 1).reshape(x.size(0), -1)
        last_hidden = self.post_dropout(last_hidden)
        return self.classifier(last_hidden)


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Train an LSTM classifier on IoT device data")
    parser.add_argument("--train-csv", default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--valid-csv", default=DEFAULT_VALID_PATH)
    parser.add_argument("--feature-ranking-csv", default=DEFAULT_FEATURE_RANKING_PATH)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--lr-scheduler",
        choices=["none", "step", "cosine"],
        default="step",
        help="Learning rate scheduler to use",
    )
    parser.add_argument(
        "--lr-step-size",
        type=int,
        default=10,
        help="Step size (in epochs) for StepLR scheduler",
    )
    parser.add_argument(
        "--lr-step-gamma",
        type=float,
        default=0.5,
        help="Multiplicative factor of learning rate decay for StepLR",
    )
    parser.add_argument(
        "--lr-cosine-t0",
        type=int,
        default=10,
        help="Initial period for CosineAnnealingWarmRestarts",
    )
    parser.add_argument(
        "--lr-cosine-t-mult",
        type=int,
        default=2,
        help="Period multiplier for CosineAnnealingWarmRestarts",
    )
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument(
        "--max-grad-norm", type=float, default=1.0, help="Gradient clipping norm (<=0 disables)"
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
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        max_grad_norm=args.max_grad_norm if args.max_grad_norm > 0 else None,
        output_dir=args.output_dir,
        seed=args.seed,
    )


def select_features(feature_csv: str, top_k: int) -> Sequence[str]:
    ranking_df = pd.read_csv(feature_csv)
    if "Votes" not in ranking_df.columns or "Variable_Name" not in ranking_df.columns:
        raise ValueError("Feature ranking CSV must contain 'Votes' and 'Variable_Name' columns")
    filtered = ranking_df[ranking_df["Votes"] > 0].sort_values("Votes", ascending=False)
    features = filtered["Variable_Name"].tolist()
    if not features:
        raise ValueError("No features with positive vote scores found")
    if top_k > len(features):
        print(
            f"[warn] Requested top_k={top_k} but only {len(features)} available; using {len(features)}"
        )
        top_k = len(features)
    selected = features[:top_k]
    print(f"[info] Selected top {len(selected)} features from ranking")
    return selected


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


def prepare_tensors(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    label_encoder: LabelEncoder,
    scaler: StandardScaler,
    fit_transform: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if "Label" not in df.columns:
        raise KeyError("Expected 'Label' column")
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing expected features: {missing_cols}")

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


def build_scheduler(config: TrainingConfig, optimizer: torch.optim.Optimizer):
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler]
    if config.lr_scheduler == "none":
        scheduler = None
    elif config.lr_scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(1, config.lr_step_size),
            gamma=config.lr_step_gamma,
        )
    elif config.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=max(1, config.lr_cosine_t0),
            T_mult=max(1, config.lr_cosine_t_mult),
        )
    else:
        raise ValueError(f"Unsupported lr_scheduler option: {config.lr_scheduler}")
    return scheduler


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


def main() -> None:
    config = parse_args()
    set_seed(config.seed)
    device = get_device()

    features = select_features(config.feature_csv, config.top_k_features)

    train_df = pd.read_csv(config.train_csv, low_memory=False)
    valid_df = pd.read_csv(config.valid_csv, low_memory=False)

    train_label_set = set(train_df["Label"].astype(str).unique())
    valid_label_set = set(valid_df["Label"].astype(str).unique())
    unseen_labels = sorted(valid_label_set - train_label_set)
    if unseen_labels:
        raise ValueError(
            "Validation set contains labels not present in training data: "
            f"{unseen_labels}. Ensure label distributions align before training."
        )

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

    seq_len = train_features.shape[1]
    train_seq = train_features.unsqueeze(-1)  # (batch, seq_len, 1)
    valid_seq = valid_features.unsqueeze(-1)

    train_dataset = TensorDataset(train_seq, train_labels)
    valid_dataset = TensorDataset(valid_seq, valid_labels)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    eval_train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False)

    model = TabularLSTM(
        input_size=1,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
        bidirectional=config.bidirectional,
        num_classes=len(label_encoder.classes_),
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=max(0.0, config.label_smoothing))
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = build_scheduler(config, optimizer)

    history: list[dict] = []
    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, config.max_grad_norm
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
    metrics_path = os.path.join(config.output_dir, "lstm_validation_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "config": {
                    "top_k_features": config.top_k_features,
                    "hidden_size": config.hidden_size,
                    "num_layers": config.num_layers,
                    "dropout": config.dropout,
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
                    "num_features": seq_len,
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
