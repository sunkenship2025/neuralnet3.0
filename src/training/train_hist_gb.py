#!/usr/bin/env python3
"""Train a histogram-based gradient boosting model for IoT device identification.

This script mirrors the data-preparation flow used by the neural trainers, but swaps in
scikit-learn's HistGradientBoostingClassifier. Besides aggregate metrics, it also stores
per-class statistics and the raw validation probabilities so they can be ensembled later on.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import LabelEncoder


DEFAULT_TRAIN_PATH = "data/Aalto_train_IoTDevID (1).csv"
DEFAULT_VALID_PATH = "data/Aalto_test_IoTDevID (1).csv"
DEFAULT_FEATURE_RANKING_PATH = "data/veto_average_results (1).csv"
DEFAULT_OUTPUT_DIR = "outputs"


@dataclass
class TrainingConfig:
    train_csv: str
    valid_csv: str
    feature_csv: str
    top_k_features: int
    learning_rate: float
    max_iter: int
    max_depth: int | None
    max_leaf_nodes: int | None
    min_samples_leaf: int
    l2_regularization: float
    max_bins: int
    class_weight: str | None
    output_dir: str
    seed: int


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(
        description="Train a histogram gradient boosting classifier on IoT device data"
    )
    parser.add_argument("--train-csv", default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--valid-csv", default=DEFAULT_VALID_PATH)
    parser.add_argument("--feature-ranking-csv", default=DEFAULT_FEATURE_RANKING_PATH)
    parser.add_argument("--top-k", type=int, default=80, help="Number of top-ranked features")
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-iter", type=int, default=400)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--max-leaf-nodes", type=int, default=63)
    parser.add_argument("--min-samples-leaf", type=int, default=20)
    parser.add_argument("--l2-regularization", type=float, default=1.0)
    parser.add_argument("--max-bins", type=int, default=255)
    parser.add_argument(
        "--class-weight",
        choices=[None, "balanced"],
        default="balanced",
        help="Class weighting strategy passed to the classifier",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    return TrainingConfig(
        train_csv=args.train_csv,
        valid_csv=args.valid_csv,
        feature_csv=args.feature_ranking_csv,
        top_k_features=args.top_k,
        learning_rate=args.learning_rate,
        max_iter=args.max_iter,
        max_depth=args.max_depth,
        max_leaf_nodes=args.max_leaf_nodes,
        min_samples_leaf=args.min_samples_leaf,
        l2_regularization=args.l2_regularization,
        max_bins=args.max_bins,
        class_weight=args.class_weight,
        output_dir=args.output_dir,
        seed=args.seed,
    )


def select_features(feature_csv: str, top_k: int) -> List[str]:
    ranking_df = pd.read_csv(feature_csv)
    if {"Votes", "Variable_Name"}.difference(ranking_df.columns):
        raise ValueError("Feature ranking CSV must contain 'Votes' and 'Variable_Name' columns")

    filtered = ranking_df[ranking_df["Votes"] > 0].sort_values("Votes", ascending=False)
    features = filtered["Variable_Name"].tolist()
    if not features:
        raise ValueError("No positively ranked features found in the ranking CSV")

    if top_k > len(features):
        print(
            f"[warn] Requested top_k={top_k} but only {len(features)} features have positive votes. "
            f"Using {len(features)}."
        )
        top_k = len(features)
    print(f"[info] Using top {top_k} ranked features")
    return features[:top_k]


def ensure_label_overlap(train_df: pd.DataFrame, valid_df: pd.DataFrame) -> None:
    train_labels = set(train_df["Label"].astype(str).unique())
    valid_labels = set(valid_df["Label"].astype(str).unique())
    unseen = sorted(valid_labels - train_labels)
    if unseen:
        raise ValueError(
            "Validation set contains labels absent from training data: "
            f"{unseen}. Align splits or remove unseen classes before training."
        )


def load_features(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> np.ndarray:
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing expected feature columns in dataframe: {missing}")
    return df[feature_cols].astype(np.float32).fillna(0.0).values


def compute_macro_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
    }


def compute_per_class_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_encoder: LabelEncoder,
) -> Dict[str, Dict[str, float]]:
    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(label_encoder.classes_)),
        target_names=label_encoder.classes_,
        zero_division=0,
        output_dict=True,
    )
    # Remove global averages to keep file compact.
    return {label: metrics for label, metrics in report.items() if label in label_encoder.classes_}


def maybe_feature_importances(model: HistGradientBoostingClassifier, feature_cols: Sequence[str]):
    importance = getattr(model, "feature_importances_", None)
    if importance is None:
        return None
    return {feature: float(score) for feature, score in zip(feature_cols, importance)}


def main() -> None:
    config = parse_args()

    train_df = pd.read_csv(config.train_csv, low_memory=False)
    valid_df = pd.read_csv(config.valid_csv, low_memory=False)
    ensure_label_overlap(train_df, valid_df)

    ranked_features = select_features(config.feature_csv, config.top_k_features)
    feature_cols = [f for f in ranked_features if f in train_df.columns]
    missing_from_train = sorted(set(ranked_features) - set(feature_cols))
    if missing_from_train:
        print(
            f"[warn] Dropping {len(missing_from_train)} ranked features not found in train CSV: "
            f"{missing_from_train}"
        )
    missing_from_valid = [f for f in feature_cols if f not in valid_df.columns]
    if missing_from_valid:
        print(
            f"[warn] Dropping {len(missing_from_valid)} features missing from validation CSV: "
            f"{missing_from_valid}"
        )
        feature_cols = [f for f in feature_cols if f in valid_df.columns]
    if not feature_cols:
        raise ValueError("No overlapping features between ranking CSV and datasets")

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df["Label"].astype(str).values)
    y_valid = label_encoder.transform(valid_df["Label"].astype(str).values)

    X_train = load_features(train_df, feature_cols)
    X_valid = load_features(valid_df, feature_cols)
    
    # Apply StandardScaler BEFORE training
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)

    model = HistGradientBoostingClassifier(
        learning_rate=config.learning_rate,
        max_iter=config.max_iter,
        max_depth=config.max_depth,
        max_leaf_nodes=config.max_leaf_nodes,
        min_samples_leaf=config.min_samples_leaf,
        l2_regularization=config.l2_regularization,
        max_bins=config.max_bins,
        class_weight=config.class_weight,
        random_state=config.seed,
        early_stopping=False,
    )

    print(
        "[info] Training HistGradientBoostingClassifier with "
        f"{X_train.shape[1]} features and {len(label_encoder.classes_)} classes"
    )
    model.fit(X_train_scaled, y_train)

    train_pred = model.predict(X_train_scaled)
    valid_pred = model.predict(X_valid_scaled)
    valid_proba = model.predict_proba(X_valid_scaled)

    train_metrics = compute_macro_metrics(y_train, train_pred)
    valid_metrics = compute_macro_metrics(y_valid, valid_pred)
    per_class_metrics = compute_per_class_report(y_valid, valid_pred, label_encoder)
    conf_mat = confusion_matrix(y_valid, valid_pred).tolist()

    os.makedirs(config.output_dir, exist_ok=True)

    metrics = {
        "config": {
            "top_k_features": config.top_k_features,
            "learning_rate": config.learning_rate,
            "max_iter": config.max_iter,
            "max_depth": config.max_depth,
            "max_leaf_nodes": config.max_leaf_nodes,
            "min_samples_leaf": config.min_samples_leaf,
            "l2_regularization": config.l2_regularization,
            "max_bins": config.max_bins,
            "class_weight": config.class_weight,
            "num_features": X_train.shape[1],
            "feature_names": feature_cols,
            "classes": label_encoder.classes_.tolist(),
        },
        "train_metrics": train_metrics,
        "valid_metrics": valid_metrics,
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": conf_mat,
        "feature_importances": maybe_feature_importances(model, feature_cols),
    }

    metrics_path = os.path.join(config.output_dir, "hist_gb_validation_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"[info] Saved metrics to {metrics_path}")

    # Persist validation probabilities for ensembling.
    proba_path = os.path.join(config.output_dir, "hist_gb_valid_proba.npy")
    np.save(proba_path, valid_proba)
    print(f"[info] Saved validation probabilities to {proba_path}")

    pred_path = os.path.join(config.output_dir, "hist_gb_valid_predictions.csv")
    pd.DataFrame(
        {
            "Label": label_encoder.inverse_transform(y_valid),
            "Predicted": label_encoder.inverse_transform(valid_pred),
        }
    ).to_csv(pred_path, index=False)
    print(f"[info] Saved validation predictions to {pred_path}")

    # Save the trained model and preprocessing artifacts for API deployment
    import pickle
    
    model_path = os.path.join(config.output_dir, "hist_gb_model.pkl")
    with open(model_path, "wb") as fh:
        pickle.dump(model, fh)
    print(f"[info] Saved trained model to {model_path}")
    
    # Save the scaler that was actually used during training
    scaler_path = os.path.join(config.output_dir, "scaler.pkl")
    with open(scaler_path, "wb") as fh:
        pickle.dump(scaler, fh)
    print(f"[info] Saved scaler to {scaler_path}")
    
    # Save the label encoder
    encoder_path = os.path.join(config.output_dir, "label_encoder.pkl")
    with open(encoder_path, "wb") as fh:
        pickle.dump(label_encoder, fh)
    print(f"[info] Saved label encoder to {encoder_path}")


if __name__ == "__main__":
    main()
