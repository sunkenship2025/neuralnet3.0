#!/usr/bin/env python3
"""Train a feedforward autoencoder on the Aalto IoT Device Identification dataset.

The script mirrors the preprocessing pipeline used by the supervised trainers, but optimises a
reconstruction objective instead of classification accuracy. Learned latent representations and
per-sample reconstruction errors are exported for downstream use (e.g., anomaly detection or
stacked ensembles).
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


DEFAULT_TRAIN_PATH = "Aalto_train_IoTDevID (1).csv"
DEFAULT_VALID_PATH = "Aalto_test_IoTDevID (1).csv"
DEFAULT_FEATURE_RANKING_PATH = "veto_average_results (1).csv"
DEFAULT_OUTPUT_DIR = "outputs"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        print("[info] Using CUDA GPU backend")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        print("[info] Using Apple Metal Performance Shaders (MPS) backend")
        return torch.device("mps")
    print("[info] Falling back to CPU")
    return torch.device("cpu")


@dataclass
class TrainingConfig:
    train_csv: str
    valid_csv: str
    feature_csv: str
    top_k_features: int
    hidden_dims: Sequence[int]
    latent_dim: int
    dropout: float
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    max_grad_norm: Optional[float]
    output_dir: str
    save_embeddings: bool
    seed: int


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Train an autoencoder on IoT device features")
    parser.add_argument("--train-csv", default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--valid-csv", default=DEFAULT_VALID_PATH)
    parser.add_argument("--feature-ranking-csv", default=DEFAULT_FEATURE_RANKING_PATH)
    parser.add_argument("--top-k", type=int, default=60)
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[256, 128],
        help="Hidden layer widths for the encoder (mirrored for decoder)",
    )
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm (<=0 disables clipping)",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--save-embeddings", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    return TrainingConfig(
        train_csv=args.train_csv,
        valid_csv=args.valid_csv,
        feature_csv=args.feature_ranking_csv,
        top_k_features=args.top_k,
        hidden_dims=args.hidden_dims,
        latent_dim=args.latent_dim,
        dropout=max(0.0, args.dropout),
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm if args.max_grad_norm > 0 else None,
        output_dir=args.output_dir,
        save_embeddings=args.save_embeddings,
        seed=args.seed,
    )


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_features(feature_csv: str, top_k: int) -> List[str]:
    ranking_df = pd.read_csv(feature_csv)
    if {"Votes", "Variable_Name"}.difference(ranking_df.columns):
        raise ValueError("Feature ranking CSV must contain 'Votes' and 'Variable_Name' columns")

    filtered = ranking_df[ranking_df["Votes"] > 0].sort_values("Votes", ascending=False)
    features = filtered["Variable_Name"].tolist()
    if not features:
        raise ValueError("No features with positive vote scores found in ranking CSV")

    if top_k > len(features):
        print(
            f"[warn] Requested top_k={top_k} but only {len(features)} ranked features available. "
            f"Using {len(features)}."
        )
        top_k = len(features)

    selected = features[:top_k]
    print(f"[info] Selected top {len(selected)} features from ranking")
    return selected


def ensure_feature_overlap(
    features: Sequence[str],
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
) -> List[str]:
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
    return feature_cols


def prepare_features(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    scaler: StandardScaler,
    fit_transform: bool = False,
) -> Tuple[torch.Tensor, Optional[List[str]]]:
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing expected feature columns: {missing_cols}")

    features_np = df[feature_cols].astype(np.float32).fillna(0.0).values
    if fit_transform:
        features_np = scaler.fit_transform(features_np)
    else:
        features_np = scaler.transform(features_np)

    labels: Optional[List[str]] = None
    if "Label" in df.columns:
        labels = df["Label"].astype(str).tolist()

    features_tensor = torch.tensor(features_np, dtype=torch.float32)
    return features_tensor, labels


class FeedforwardAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        latent_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer width")

        encoder_layers: List[nn.Module] = []
        current_dim = input_dim
        for width in hidden_dims:
            encoder_layers.append(nn.Linear(current_dim, width))
            encoder_layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                encoder_layers.append(nn.Dropout(dropout))
            current_dim = width
        encoder_layers.append(nn.Linear(current_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: List[nn.Module] = []
        current_dim = latent_dim
        for width in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(current_dim, width))
            decoder_layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                decoder_layers.append(nn.Dropout(dropout))
            current_dim = width
        decoder_layers.append(nn.Linear(current_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_grad_norm: Optional[float],
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    for (batch_x,) in loader:
        batch_x = batch_x.to(device)

        optimizer.zero_grad(set_to_none=True)
        recon = model(batch_x)
        loss = criterion(recon, batch_x)
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)
        total_samples += batch_x.size(0)

    return total_loss / total_samples


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    return_details: bool = False,
) -> Tuple[Dict[str, float], Optional[np.ndarray], Optional[np.ndarray]]:
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    total_samples = 0

    per_sample_mse: List[np.ndarray] = []
    per_sample_mae: List[np.ndarray] = []

    for (batch_x,) in loader:
        batch_x = batch_x.to(device)
        recon = model(batch_x)
        mse_batch = torch.mean((recon - batch_x) ** 2, dim=1)
        mae_batch = torch.mean(torch.abs(recon - batch_x), dim=1)

        total_mse += mse_batch.sum().item()
        total_mae += mae_batch.sum().item()
        total_samples += batch_x.size(0)

        if return_details:
            per_sample_mse.append(mse_batch.cpu().numpy())
            per_sample_mae.append(mae_batch.cpu().numpy())

    metrics = {
        "mse": total_mse / total_samples,
        "mae": total_mae / total_samples,
    }

    if not return_details:
        return metrics, None, None

    mse_array = np.concatenate(per_sample_mse, axis=0) if per_sample_mse else None
    mae_array = np.concatenate(per_sample_mae, axis=0) if per_sample_mae else None
    return metrics, mse_array, mae_array


@torch.no_grad()
def collect_embeddings(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    embeddings: List[np.ndarray] = []
    for (batch_x,) in loader:
        batch_x = batch_x.to(device)
        latent = model.encode(batch_x)
        embeddings.append(latent.cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def main() -> None:
    config = parse_args()
    set_seed(config.seed)
    device = get_device()

    ranked_features = select_features(config.feature_csv, config.top_k_features)
    train_df = pd.read_csv(config.train_csv, low_memory=False)
    valid_df = pd.read_csv(config.valid_csv, low_memory=False)
    feature_cols = ensure_feature_overlap(ranked_features, train_df, valid_df)

    scaler = StandardScaler()
    train_features, train_labels = prepare_features(
        train_df, feature_cols, scaler, fit_transform=True
    )
    valid_features, valid_labels = prepare_features(
        valid_df, feature_cols, scaler, fit_transform=False
    )

    train_dataset = TensorDataset(train_features)
    valid_dataset = TensorDataset(valid_features)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False)

    model = FeedforwardAutoencoder(
        input_dim=train_features.shape[1],
        hidden_dims=config.hidden_dims,
        latent_dim=config.latent_dim,
        dropout=config.dropout,
    ).to(device)

    criterion = nn.MSELoss(reduction="mean")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    history: List[Dict[str, float]] = []
    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, config.max_grad_norm
        )
        valid_metrics, _, _ = evaluate(model, valid_loader, device, return_details=False)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_mse": valid_metrics["mse"],
                "valid_mae": valid_metrics["mae"],
            }
        )

        print(
            f"Epoch {epoch:03d}/{config.epochs}: train_mse={train_loss:.6f} "
            f"valid_mse={valid_metrics['mse']:.6f} valid_mae={valid_metrics['mae']:.6f}"
        )

    final_train_metrics, train_mse_arr, train_mae_arr = evaluate(
        model, train_loader, device, return_details=True
    )
    final_valid_metrics, valid_mse_arr, valid_mae_arr = evaluate(
        model, valid_loader, device, return_details=True
    )

    os.makedirs(config.output_dir, exist_ok=True)
    metrics_path = os.path.join(config.output_dir, "autoencoder_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "config": {
                    "top_k_features": config.top_k_features,
                    "hidden_dims": list(config.hidden_dims),
                    "latent_dim": config.latent_dim,
                    "dropout": config.dropout,
                    "batch_size": config.batch_size,
                    "epochs": config.epochs,
                    "learning_rate": config.learning_rate,
                    "weight_decay": config.weight_decay,
                    "max_grad_norm": config.max_grad_norm,
                    "device": device.type,
                    "feature_names": list(feature_cols),
                },
                "history": history,
                "final_train_metrics": final_train_metrics,
                "final_valid_metrics": final_valid_metrics,
            },
            fh,
            indent=2,
        )
    print(f"[info] Saved metrics to {metrics_path}")

    # Save per-sample reconstruction errors.
    if train_mse_arr is not None and train_mae_arr is not None:
        train_error_path = os.path.join(config.output_dir, "autoencoder_train_errors.csv")
        train_error_df = pd.DataFrame(
            {
                "mse": train_mse_arr,
                "mae": train_mae_arr,
                "Label": train_labels if train_labels is not None else None,
            }
        )
        train_error_df.to_csv(train_error_path, index=False)
        print(f"[info] Saved training reconstruction errors to {train_error_path}")

    if valid_mse_arr is not None and valid_mae_arr is not None:
        valid_error_path = os.path.join(config.output_dir, "autoencoder_valid_errors.csv")
        valid_error_df = pd.DataFrame(
            {
                "mse": valid_mse_arr,
                "mae": valid_mae_arr,
                "Label": valid_labels if valid_labels is not None else None,
            }
        )
        valid_error_df.to_csv(valid_error_path, index=False)
        print(f"[info] Saved validation reconstruction errors to {valid_error_path}")

    if config.save_embeddings:
        train_eval_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False)
        valid_eval_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False)
        train_latent = collect_embeddings(model, train_eval_loader, device)
        valid_latent = collect_embeddings(model, valid_eval_loader, device)

        train_latent_path = os.path.join(config.output_dir, "autoencoder_train_latent.npy")
        valid_latent_path = os.path.join(config.output_dir, "autoencoder_valid_latent.npy")
        np.save(train_latent_path, train_latent)
        np.save(valid_latent_path, valid_latent)
        print(f"[info] Saved latent embeddings to {train_latent_path} and {valid_latent_path}")

    model_path = os.path.join(config.output_dir, "autoencoder_state.pt")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_names": feature_cols,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
        },
        model_path,
    )
    print(f"[info] Saved model state to {model_path}")


if __name__ == "__main__":
    main()
