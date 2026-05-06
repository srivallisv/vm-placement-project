"""
comparison_graphs.py
Generates milestone-based comparison visualizations for GRU vs Informer vs PatchTST.
All graphs use only milestone epochs [30, 50, 70, 90, 110, 130, 150].

Outputs saved to:  outputs/graphs/comparisons/
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from evaluation.milestone_evaluator import METRIC_NAMES


COMP_DIR = os.path.join(config.GRAPHS_DIR, "comparisons")

# Display-friendly names and groupings
METRIC_LABELS = {
    "MAE": "Mean Absolute Error",
    "RMSE": "Root Mean Squared Error",
    "MAPE": "Mean Absolute Percentage Error (%)",
    "Accuracy": "Classification Accuracy",
    "Precision": "Classification Precision",
    "AUC": "Area Under ROC Curve",
    "Energy": "Energy Consumption (W)",
    "Migrations": "Migration Count",
    "Failure_Rate": "Failure Rate",
    "Active_Servers": "Active Server Count",
}

METRIC_GROUPS = {
    "Prediction Accuracy": ["MAE", "RMSE", "MAPE"],
    "Resource Allocation Quality": ["Accuracy", "Precision", "AUC"],
    "Energy Efficiency": ["Energy", "Active_Servers"],
    "Consolidation & Failures": ["Migrations", "Failure_Rate"],
}

MODEL_COLORS = {
    "gru": "#2196F3",
    "informer": "#FF5722",
    "patchtst": "#4CAF50",
}

MODEL_MARKERS = {
    "gru": "o",
    "informer": "s",
    "patchtst": "D",
}


def _setup():
    sns.set_theme(style="whitegrid", palette="deep")
    os.makedirs(COMP_DIR, exist_ok=True)


def plot_metric_comparison_lines(results: dict, metric: str):
    """
    Line plot comparing one metric across milestones for all 3 models.
    x-axis = milestone epoch, y-axis = metric value.
    """
    _setup()
    plt.figure(figsize=(10, 6))

    for model_name in sorted(results.keys()):
        epochs = sorted(results[model_name].keys())
        values = [results[model_name][e].get(metric, 0) for e in epochs]
        plt.plot(
            epochs, values,
            marker=MODEL_MARKERS.get(model_name, "o"),
            color=MODEL_COLORS.get(model_name, None),
            linewidth=2, markersize=8,
            label=model_name.upper(),
        )

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel(METRIC_LABELS.get(metric, metric), fontsize=12)
    plt.title(f"{METRIC_LABELS.get(metric, metric)} — Milestone Comparison", fontsize=14)
    plt.legend(fontsize=11)
    plt.xticks(config.MILESTONE_EPOCHS)
    plt.tight_layout()
    plt.savefig(os.path.join(COMP_DIR, f"line_{metric.lower()}.png"), dpi=150)
    plt.close()


def plot_grouped_bar_chart(results: dict, metric: str):
    """
    Grouped bar chart comparing one metric at each milestone for all 3 models.
    """
    _setup()
    models = sorted(results.keys())
    milestones = config.MILESTONE_EPOCHS
    n_models = len(models)
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(milestones))

    for i, model_name in enumerate(models):
        values = [results[model_name].get(e, {}).get(metric, 0) for e in milestones]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, values, width,
            label=model_name.upper(),
            color=MODEL_COLORS.get(model_name, None),
            alpha=0.85, edgecolor="white",
        )
        # Value labels on top of bars
        for bar, val in zip(bars, values):
            if val != 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{val:.3f}" if val < 100 else f"{val:.0f}",
                        ha="center", va="bottom", fontsize=7)

    ax.set_xlabel("Milestone Epoch", fontsize=12)
    ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=12)
    ax.set_title(f"{METRIC_LABELS.get(metric, metric)} — Grouped Comparison", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([str(m) for m in milestones])
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(COMP_DIR, f"bar_{metric.lower()}.png"), dpi=150)
    plt.close()


def plot_group_summary(results: dict, group_name: str, metrics: list):
    """
    Multi-panel plot for a metric group (e.g., Prediction Accuracy: MAE, RMSE, MAPE).
    """
    _setup()
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    models = sorted(results.keys())
    milestones = config.MILESTONE_EPOCHS

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        for model_name in models:
            epochs = sorted(results[model_name].keys())
            values = [results[model_name][e].get(metric, 0) for e in epochs]
            ax.plot(
                epochs, values,
                marker=MODEL_MARKERS.get(model_name, "o"),
                color=MODEL_COLORS.get(model_name, None),
                linewidth=2, markersize=7,
                label=model_name.upper(),
            )
        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_xticks(milestones)
        ax.legend(fontsize=9)

    fig.suptitle(group_name, fontsize=14, fontweight="bold")
    plt.tight_layout()
    safe_name = group_name.lower().replace(" ", "_").replace("&", "and")
    plt.savefig(os.path.join(COMP_DIR, f"group_{safe_name}.png"), dpi=150)
    plt.close()


def plot_final_combined_chart(results: dict):
    """
    Final combined comparison: bar chart of the LAST milestone (epoch 150)
    for all metrics, comparing all 3 models.
    """
    _setup()
    last_epoch = config.MILESTONE_EPOCHS[-1]
    models = sorted(results.keys())

    # Separate metrics by scale for readability
    small_metrics = ["MAE", "RMSE", "Accuracy", "Precision", "AUC", "Failure_Rate"]
    large_metrics = ["MAPE", "Energy", "Migrations", "Active_Servers"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, metric_list, title in [
        (axes[0], small_metrics, "Normalized Metrics (Epoch {})".format(last_epoch)),
        (axes[1], large_metrics, "Scale Metrics (Epoch {})".format(last_epoch)),
    ]:
        x = np.arange(len(metric_list))
        width = 0.25
        for i, model_name in enumerate(models):
            values = [
                results[model_name].get(last_epoch, {}).get(m, 0)
                for m in metric_list
            ]
            offset = (i - len(models) / 2 + 0.5) * width
            ax.bar(x + offset, values, width,
                   label=model_name.upper(),
                   color=MODEL_COLORS.get(model_name, None),
                   alpha=0.85, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(metric_list, rotation=30, ha="right")
        ax.set_title(title, fontsize=12)
        ax.legend()

    fig.suptitle("Final Model Comparison (Last Milestone)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(COMP_DIR, "final_combined_comparison.png"), dpi=150)
    plt.close()


def generate_milestone_comparison_table(results: dict):
    """
    Generate a formatted text table per milestone and save to comparison_tables/.
    """
    table_dir = os.path.join(config.OUTPUT_DIR, "comparison_tables")
    os.makedirs(table_dir, exist_ok=True)

    models = sorted(results.keys())
    path = os.path.join(table_dir, "milestone_comparison_table.txt")

    with open(path, "w") as f:
        for epoch in config.MILESTONE_EPOCHS:
            f.write(f"\n{'='*70}\n")
            f.write(f"  MILESTONE EPOCH {epoch}\n")
            f.write(f"{'='*70}\n")
            header = f"{'Metric':<20}" + "".join(f"{m.upper():>15}" for m in models)
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")
            for metric in METRIC_NAMES:
                row = f"{metric:<20}"
                for m in models:
                    val = results[m].get(epoch, {}).get(metric, 0)
                    if isinstance(val, float):
                        row += f"{val:>15.5f}"
                    else:
                        row += f"{val:>15}"
                f.write(row + "\n")

    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Master graph generation
# ---------------------------------------------------------------------------

def generate_all_comparison_graphs(results: dict):
    """
    Generate all milestone comparison graphs and tables.

    Produces:
      - Line plots per metric (10 plots)
      - Grouped bar charts per metric (10 plots)
      - Group summary panels (4 multi-panel plots)
      - Final combined comparison chart (1 plot)
      - Text comparison table

    All saved to outputs/graphs/comparisons/
    """
    print("\n  Generating milestone comparison graphs ...")

    # Individual metric plots
    for metric in METRIC_NAMES:
        plot_metric_comparison_lines(results, metric)
        plot_grouped_bar_chart(results, metric)

    # Grouped summary panels
    for group_name, metrics in METRIC_GROUPS.items():
        plot_group_summary(results, group_name, metrics)

    # Final combined chart
    plot_final_combined_chart(results)

    # Text table
    generate_milestone_comparison_table(results)

    total = len(METRIC_NAMES) * 2 + len(METRIC_GROUPS) + 1
    print(f"  Generated {total} comparison graphs in {COMP_DIR}")
