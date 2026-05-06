"""
milestone_evaluator.py
Evaluates all 3 models at each milestone epoch, computing the full set of
metrics (prediction, classification, placement/energy) and generating
milestone-based comparison graphs and tables.

Metrics are computed ONLY at milestone epochs: [30, 50, 70, 90, 110, 130, 150]
"""

import os
import sys
import csv
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from confidence.confidence_score import compute_combined_confidence, compute_mc_variance
from allocation.allocation_engine import classify_vms, allocate_resources
from placement.placement_engine import create_server_fleet, place_vm
from placement.energy_model import compute_datacenter_power
from failure.failure_handler import handle_failures
from consolidation.consolidation_engine import consolidate_servers
from evaluation.metrics import (
    compute_mae, compute_rmse, compute_mape, evaluate_classification,
)


# ---------------------------------------------------------------------------
# Per-checkpoint evaluation
# ---------------------------------------------------------------------------

def _get_predictions(model, test_loader, device):
    """Run inference and return (predictions, actuals) as numpy arrays."""
    model.eval()
    preds_list, targets_list = [], []
    use_amp = device.type == "cuda"

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(x)
            preds_list.append(pred.cpu().numpy())
            targets_list.append(y.numpy())

    return np.concatenate(preds_list), np.concatenate(targets_list)


def _run_mini_simulation(allocated, labels, predicted, current, confidence):
    """
    Run a single-pass placement simulation and return system metrics.
    Returns dict with energy, migrations, failure_rate, active_servers.
    """
    servers = create_server_fleet()

    # Place overloaded VMs
    placements = {}
    failures = []
    for vm_id in range(len(labels)):
        if labels[vm_id] == 0:
            continue
        cpu, mem, storage = allocated[vm_id]
        sid = place_vm(servers, vm_id, cpu, mem, storage)
        if sid >= 0:
            placements[vm_id] = sid
        else:
            failures.append(vm_id)

    total_attempted = int(labels.sum())

    # Handle failures
    recovered = 0
    if failures:
        fail_result = handle_failures(
            servers, failures, predicted, current, confidence
        )
        recovered = len(fail_result["recovered"])

    # Consolidate
    consol = consolidate_servers(servers, allocated)

    # Compute energy
    utils = np.array([s.cpu_util for s in servers if s.is_active])
    energy = compute_datacenter_power(utils) if len(utils) > 0 else 0.0

    net_failures = len(failures) - recovered
    failure_rate = net_failures / max(total_attempted, 1)

    return {
        "energy": energy,
        "migrations": consol["migrations"],
        "failure_rate": failure_rate,
        "active_servers": consol["active_servers"],
    }


def evaluate_checkpoint(model, test_loader, device):
    """
    Full evaluation of a single model checkpoint.
    Returns dict with all 10 metrics.
    """
    # --- Prediction metrics ---
    predictions, actuals = _get_predictions(model, test_loader, device)
    n = len(predictions)

    # Flatten to (N*horizon, features) for aggregate metrics
    pred_flat = predictions.reshape(-1, predictions.shape[-1])
    act_flat = actuals.reshape(-1, actuals.shape[-1])

    mae = compute_mae(act_flat, pred_flat)
    rmse = compute_rmse(act_flat, pred_flat)
    mape = compute_mape(act_flat, pred_flat)

    # --- Confidence + Classification ---
    variance = compute_mc_variance(model, test_loader, device, n_passes=10)
    n = min(n, len(variance))
    confidence = compute_combined_confidence(
        predictions[:n], actuals[:n], variance[:n]
    )

    pred_step = predictions[:n, 0, :]
    curr_step = actuals[:n, 0, :]
    conf_step = confidence[:n, 0, :]

    labels_pred = classify_vms(pred_step, curr_step)
    labels_true = classify_vms(actuals[:n, 0, :], curr_step)

    cls = evaluate_classification(
        labels_true, labels_pred,
        y_prob=conf_step.mean(axis=1),
    )

    # --- Placement simulation ---
    allocated = allocate_resources(curr_step, pred_step, conf_step)
    sim = _run_mini_simulation(allocated, labels_pred, pred_step,
                               curr_step, conf_step)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "Accuracy": cls["accuracy"],
        "Precision": cls["precision"],
        "AUC": cls.get("auc", 0.0),
        "Energy": sim["energy"],
        "Migrations": sim["migrations"],
        "Failure_Rate": sim["failure_rate"],
        "Active_Servers": sim["active_servers"],
    }


# ---------------------------------------------------------------------------
# Main milestone evaluation loop
# ---------------------------------------------------------------------------

METRIC_NAMES = [
    "MAE", "RMSE", "MAPE", "Accuracy", "Precision", "AUC",
    "Energy", "Migrations", "Failure_Rate", "Active_Servers",
]


def run_milestone_evaluation(model_configs: dict, test_loader,
                             device: torch.device) -> dict:
    """
    Evaluate all models at every milestone epoch.

    Args:
        model_configs: {name: (ModelClass, kwargs_dict)}
        test_loader: DataLoader for test set
        device: torch device

    Returns:
        results: {model_name: {epoch: {metric: value}}}
    """
    results = {}

    for model_name, (model_class, model_kwargs) in model_configs.items():
        print(f"\n{'='*50}")
        print(f"  Milestone evaluation: {model_name.upper()}")
        print(f"{'='*50}")

        results[model_name] = {}

        for epoch in config.MILESTONE_EPOCHS:
            ckpt_path = os.path.join(
                config.CHECKPOINT_DIR, f"{model_name}_epoch{epoch}.pt"
            )

            if not os.path.exists(ckpt_path):
                print(f"  [SKIP] Epoch {epoch}: checkpoint not found")
                continue

            print(f"  Epoch {epoch} ...", end=" ", flush=True)

            # Load checkpoint
            model = model_class(**model_kwargs)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            model.to(device)

            # Evaluate
            metrics = evaluate_checkpoint(model, test_loader, device)
            results[model_name][epoch] = metrics

            print(f"MAE={metrics['MAE']:.5f}  RMSE={metrics['RMSE']:.5f}  "
                  f"Acc={metrics['Accuracy']:.4f}  Energy={metrics['Energy']:.1f}W")

            # Free memory
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Save milestone CSVs
# ---------------------------------------------------------------------------

def save_milestone_csvs(results: dict):
    """Save per-model milestone metric CSVs to outputs/comparison_tables/."""
    table_dir = os.path.join(config.OUTPUT_DIR, "comparison_tables")
    os.makedirs(table_dir, exist_ok=True)

    for model_name, epoch_data in results.items():
        path = os.path.join(table_dir, f"{model_name}_milestone_metrics.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch"] + METRIC_NAMES)
            for epoch in sorted(epoch_data.keys()):
                row = [epoch] + [f"{epoch_data[epoch][m]:.6f}" for m in METRIC_NAMES]
                w.writerow(row)
        print(f"  Saved: {path}")


def save_final_comparison_csv(results: dict):
    """
    Save outputs/final_milestone_comparison.csv with all models × milestones.
    Format: epoch, metric, gru, informer, patchtst
    """
    path = os.path.join(config.OUTPUT_DIR, "final_milestone_comparison.csv")
    models = sorted(results.keys())

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "metric"] + [m.upper() for m in models])

        for epoch in config.MILESTONE_EPOCHS:
            for metric in METRIC_NAMES:
                row = [epoch, metric]
                for m in models:
                    val = results[m].get(epoch, {}).get(metric, "")
                    row.append(f"{val:.6f}" if isinstance(val, (int, float)) else "")
                w.writerow(row)

    print(f"  Saved: {path}")
