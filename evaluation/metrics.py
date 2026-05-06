"""
metrics.py
Evaluation metrics and graph generation for all pipeline stages.
Pure CPU/numpy/matplotlib — no GPU usage.
"""

import os
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, roc_auc_score, roc_curve,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ---------------------------------------------------------------------------
# Prediction metrics
# ---------------------------------------------------------------------------

def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.abs(y_true - y_pred))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-8
    return np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + eps)) * 100


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray,
                         feature_names: list = None) -> dict:
    """
    Compute MAE, RMSE, MAPE per feature and overall.

    Args:
        y_true, y_pred: (N, features) or (N, horizon, features)
        feature_names: list of feature names

    Returns:
        dict of metrics
    """
    if feature_names is None:
        feature_names = config.FEATURES

    # Flatten horizon if present
    if y_true.ndim == 3:
        y_true = y_true.reshape(-1, y_true.shape[-1])
        y_pred = y_pred.reshape(-1, y_pred.shape[-1])

    results = {}
    for i, name in enumerate(feature_names):
        results[name] = {
            "MAE": compute_mae(y_true[:, i], y_pred[:, i]),
            "RMSE": compute_rmse(y_true[:, i], y_pred[:, i]),
            "MAPE": compute_mape(y_true[:, i], y_pred[:, i]),
        }
    results["overall"] = {
        "MAE": compute_mae(y_true, y_pred),
        "RMSE": compute_rmse(y_true, y_pred),
        "MAPE": compute_mape(y_true, y_pred),
    }
    return results


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray,
                            y_prob: np.ndarray = None) -> dict:
    """
    Compute accuracy, precision, and AUC for VM overload classification.

    Args:
        y_true: (N,) binary ground truth
        y_pred: (N,) binary predictions
        y_prob: (N,) probability scores for AUC (optional)

    Returns:
        dict of classification metrics
    """
    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        results["auc"] = roc_auc_score(y_true, y_prob)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        results["roc_fpr"] = fpr
        results["roc_tpr"] = tpr
    return results


# ---------------------------------------------------------------------------
# Graph generation
# ---------------------------------------------------------------------------

def _setup_style():
    """Apply consistent plot styling."""
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams.update({"figure.figsize": (10, 6), "figure.dpi": 120})


def plot_training_loss(histories: dict, save_dir: str = None):
    """Plot training loss curves for all models (all 150 epochs)."""
    _setup_style()
    if save_dir is None:
        save_dir = config.GRAPHS_DIR
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for name, hist in histories.items():
        epochs = range(1, len(hist["train_loss"]) + 1)
        axes[0].plot(epochs, hist["train_loss"], label=name.upper())
        axes[1].plot(epochs, hist["val_loss"], label=name.upper())

    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].set_title("Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "1_training_loss_curves.png"))
    plt.close()


def plot_predictions_vs_actual(predictions: np.ndarray, actuals: np.ndarray,
                               model_name: str, save_dir: str = None):
    """Plot predicted vs actual for each resource feature."""
    _setup_style()
    if save_dir is None:
        save_dir = config.GRAPHS_DIR
    os.makedirs(save_dir, exist_ok=True)

    # Flatten horizon
    if predictions.ndim == 3:
        predictions = predictions.reshape(-1, predictions.shape[-1])
        actuals = actuals.reshape(-1, actuals.shape[-1])

    n_samples = min(500, len(predictions))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, feat in enumerate(config.FEATURES):
        axes[i].plot(actuals[:n_samples, i], label="Actual", alpha=0.7)
        axes[i].plot(predictions[:n_samples, i], label="Predicted", alpha=0.7)
        axes[i].set_title(f"{model_name.upper()} — {feat}")
        axes[i].set_xlabel("Sample")
        axes[i].set_ylabel("Value (normalized)")
        axes[i].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"2_pred_vs_actual_{model_name}.png"))
    plt.close()


def plot_confidence_distribution(confidence: np.ndarray, save_dir: str = None):
    """Plot histogram of confidence scores."""
    _setup_style()
    if save_dir is None:
        save_dir = config.GRAPHS_DIR
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    flat = confidence.reshape(-1, confidence.shape[-1]) if confidence.ndim == 3 else confidence

    for i, feat in enumerate(config.FEATURES):
        axes[i].hist(flat[:, i], bins=50, edgecolor="black", alpha=0.7)
        axes[i].set_title(f"Confidence — {feat}")
        axes[i].set_xlabel("Confidence")
        axes[i].set_ylabel("Count")
        axes[i].axvline(flat[:, i].mean(), color="red", linestyle="--",
                        label=f"Mean={flat[:, i].mean():.3f}")
        axes[i].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "3_confidence_distribution.png"))
    plt.close()


def plot_roc_curves(classification_results: dict, save_dir: str = None):
    """Plot ROC curves comparing all models."""
    _setup_style()
    if save_dir is None:
        save_dir = config.GRAPHS_DIR
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(8, 6))
    for name, res in classification_results.items():
        if "roc_fpr" in res:
            auc_val = res.get("auc", 0)
            plt.plot(res["roc_fpr"], res["roc_tpr"],
                     label=f"{name.upper()} (AUC={auc_val:.3f})")

    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — VM Overload Detection")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "4_roc_curves.png"))
    plt.close()


def plot_accuracy_precision_bars(classification_results: dict,
                                 save_dir: str = None):
    """Bar chart comparing accuracy and precision across models."""
    _setup_style()
    if save_dir is None:
        save_dir = config.GRAPHS_DIR
    os.makedirs(save_dir, exist_ok=True)

    models = list(classification_results.keys())
    accs = [classification_results[m]["accuracy"] for m in models]
    precs = [classification_results[m]["precision"] for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, accs, width, label="Accuracy")
    ax.bar(x + width/2, precs, width, label="Precision")
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in models])
    ax.set_ylabel("Score")
    ax.set_title("Classification Performance Comparison")
    ax.legend()
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "5_accuracy_precision_bars.png"))
    plt.close()


def plot_energy_over_time(energy_values: list, save_dir: str = None):
    """Plot energy consumption across simulation timesteps."""
    _setup_style()
    if save_dir is None:
        save_dir = config.GRAPHS_DIR
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(energy_values, marker="o", markersize=3)
    plt.xlabel("Timestep")
    plt.ylabel("Energy (Watts)")
    plt.title("Datacenter Energy Consumption Over Time")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "6_energy_over_time.png"))
    plt.close()


def plot_active_servers(active_counts: list, total: int,
                        save_dir: str = None):
    """Plot active server count over time."""
    _setup_style()
    if save_dir is None:
        save_dir = config.GRAPHS_DIR
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(active_counts, marker="o", markersize=3, color="green")
    plt.axhline(y=total, color="red", linestyle="--", label=f"Total={total}")
    plt.xlabel("Timestep")
    plt.ylabel("Active Servers")
    plt.title("Active Servers Over Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "7_active_servers.png"))
    plt.close()


def plot_migrations(migration_counts: list, save_dir: str = None):
    """Plot migration count per timestep."""
    _setup_style()
    if save_dir is None:
        save_dir = config.GRAPHS_DIR
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(migration_counts)), migration_counts, alpha=0.7)
    plt.xlabel("Timestep")
    plt.ylabel("Migrations")
    plt.title("VM Migrations Per Timestep")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "8_migrations.png"))
    plt.close()


def plot_milestone_metrics(histories: dict, save_dir: str = None):
    """Plot MAE and RMSE at milestone epochs for all models."""
    _setup_style()
    if save_dir is None:
        save_dir = config.GRAPHS_DIR
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for name, hist in histories.items():
        milestone_mae = [hist["val_mae"][e - 1] for e in config.MILESTONE_EPOCHS
                         if e - 1 < len(hist["val_mae"])]
        milestone_rmse = [hist["val_rmse"][e - 1] for e in config.MILESTONE_EPOCHS
                          if e - 1 < len(hist["val_rmse"])]
        ms = config.MILESTONE_EPOCHS[:len(milestone_mae)]

        axes[0].plot(ms, milestone_mae, marker="s", label=name.upper())
        axes[1].plot(ms, milestone_rmse, marker="s", label=name.upper())

    axes[0].set_title("MAE at Milestone Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MAE")
    axes[0].legend()

    axes[1].set_title("RMSE at Milestone Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("RMSE")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "9_milestone_metrics.png"))
    plt.close()


def plot_final_comparison(pred_metrics: dict, save_dir: str = None):
    """Combined comparison chart of all models across all prediction metrics."""
    _setup_style()
    if save_dir is None:
        save_dir = config.GRAPHS_DIR
    os.makedirs(save_dir, exist_ok=True)

    models = list(pred_metrics.keys())
    metrics_list = ["MAE", "RMSE", "MAPE"]
    n_metrics = len(metrics_list)

    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))

    for j, metric in enumerate(metrics_list):
        values = [pred_metrics[m]["overall"][metric] for m in models]
        axes[j].bar([m.upper() for m in models], values, alpha=0.8)
        axes[j].set_title(metric)
        axes[j].set_ylabel(metric)

    plt.suptitle("Final Model Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "10_final_comparison.png"))
    plt.close()


def save_summary_csv(pred_metrics: dict, classification_results: dict,
                     system_metrics: dict, save_dir: str = None):
    """Save a summary CSV of all evaluation metrics."""
    if save_dir is None:
        save_dir = config.METRICS_DIR
    os.makedirs(save_dir, exist_ok=True)

    path = os.path.join(save_dir, "final_summary.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "metric", "value"])

        for model, metrics in pred_metrics.items():
            for feat, vals in metrics.items():
                for metric_name, val in vals.items():
                    w.writerow([model, f"{feat}_{metric_name}", f"{val:.6f}"])

        for model, cls in classification_results.items():
            for key in ["accuracy", "precision", "auc"]:
                if key in cls:
                    w.writerow([model, f"cls_{key}", f"{cls[key]:.6f}"])

        for key, val in system_metrics.items():
            w.writerow(["system", key, f"{val}"])

    print(f"[Evaluation] Summary saved to {path}")
