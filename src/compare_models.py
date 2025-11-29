#!/usr/bin/env python3
"""Compare ML model performance metrics for IoT Device Identification."""

import json
from pathlib import Path


def load_model_metrics():
    """Load metrics from all trained models."""
    models = [
        ("HistGradientBoosting", "outputs/hist_gb_validation_metrics.json", "valid_metrics"),
        ("CNN+LSTM Hybrid", "outputs/cnn_lstm_full/cnn_lstm_validation_metrics.json", "final_validation_metrics"),
        ("LSTM (Cosine)", "outputs/lstm_cosine_long/lstm_validation_metrics.json", "final_validation_metrics"),
        ("CNN", "outputs/cnn_validation_metrics.json", "final_validation_metrics"),
        ("LSTM", "outputs/lstm_validation_metrics.json", "final_validation_metrics"),
        ("MLP", "outputs/mlp_validation_metrics.json", "final_validation_metrics"),
    ]

    results = []
    for name, path, key in models:
        try:
            with open(path) as f:
                data = json.load(f)
            vm = data.get(key, {})
            results.append({
                "Model": name,
                "Accuracy": vm.get("accuracy", 0),
                "Balanced Acc": vm.get("balanced_accuracy", 0),
                "F1 (Macro)": vm.get("f1_macro", 0),
                "Precision": vm.get("precision_macro", 0),
                "Recall": vm.get("recall_macro", 0),
                "Cohen Kappa": vm.get("cohen_kappa", 0),
            })
        except Exception as e:
            print(f"Could not load {name}: {e}")

    return sorted(results, key=lambda x: x["Accuracy"], reverse=True)


def load_autoencoder_metrics():
    """Load autoencoder metrics (unsupervised)."""
    try:
        with open("outputs/autoencoder_run1/autoencoder_metrics.json") as f:
            ae = json.load(f)
        return ae.get("final_metrics", {}), ae.get("config", {})
    except Exception as e:
        print(f"Could not load Autoencoder: {e}")
        return {}, {}


def print_comparison():
    """Print formatted comparison table."""
    results = load_model_metrics()

    print("=" * 100)
    print("ML MODEL PERFORMANCE COMPARISON - IoT Device Identification")
    print("=" * 100)

    # Header
    print(f"\n{'Model':<25} {'Accuracy':>10} {'Bal.Acc':>10} {'F1':>10} {'Precision':>10} {'Recall':>10} {'Kappa':>10}")
    print("-" * 100)

    # Each model
    for r in results:
        print(
            f"{r['Model']:<25} "
            f"{r['Accuracy']*100:>9.2f}% "
            f"{r['Balanced Acc']*100:>9.2f}% "
            f"{r['F1 (Macro)']:>10.4f} "
            f"{r['Precision']:>10.4f} "
            f"{r['Recall']:>10.4f} "
            f"{r['Cohen Kappa']:>10.4f}"
        )

    print("-" * 100)

    # Autoencoder
    print("\n" + "=" * 100)
    print("AUTOENCODER (Unsupervised - Reconstruction Loss)")
    print("=" * 100)
    ae_metrics, ae_config = load_autoencoder_metrics()
    if ae_metrics:
        print(f"Final Train MSE:  {ae_metrics.get('train_mse', 0):.6f}")
        print(f"Final Train MAE:  {ae_metrics.get('train_mae', 0):.6f}")
        print(f"Final Valid MSE:  {ae_metrics.get('valid_mse', 0):.6f}")
        print(f"Final Valid MAE:  {ae_metrics.get('valid_mae', 0):.6f}")
        print(f"Epochs trained:   {ae_config.get('epochs', 'N/A')}")

    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    if results:
        best = results[0]
        worst = results[-1]
        print(f"Best Model:  {best['Model']} with {best['Accuracy']*100:.2f}% accuracy")
        print(f"Worst Model: {worst['Model']} with {worst['Accuracy']*100:.2f}% accuracy")
        print(f"Gap:         {(best['Accuracy'] - worst['Accuracy'])*100:.2f} percentage points")
    print("=" * 100)


if __name__ == "__main__":
    print_comparison()
