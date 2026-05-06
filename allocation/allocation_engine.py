"""
allocation_engine.py
Confidence-based resource allocation and VM overload classification.
Pure CPU/numpy — no GPU usage.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def classify_vms(predicted: np.ndarray,
                 current: np.ndarray,
                 threshold: float = None) -> np.ndarray:
    """
    Classify VMs as needing migration (overloaded) or not.

    Rule:
        MR = 1  if  Predicted > Current + threshold  else  0

    Evaluated per-sample across all features; a VM is flagged if ANY
    feature exceeds the threshold.

    Args:
        predicted: (N, n_features) — predicted resource usage
        current:   (N, n_features) — current resource usage
        threshold: overload threshold (default from config)

    Returns:
        labels: (N,) — binary array (1 = migration required)
    """
    if threshold is None:
        threshold = config.OVERLOAD_THRESHOLD

    # A VM is overloaded if ANY resource exceeds threshold
    excess = predicted - current  # (N, n_features)
    overloaded = (excess > threshold).any(axis=1).astype(np.int32)
    return overloaded


def allocate_resources(current: np.ndarray,
                       predicted: np.ndarray,
                       confidence: np.ndarray) -> np.ndarray:
    """
    Confidence-based resource allocation.

    Formula:
        Allocated = Current + Confidence × (Predicted - Current)

    High confidence → allocation closer to prediction.
    Low confidence  → allocation stays near current usage.

    Args:
        current:    (N, n_features)
        predicted:  (N, n_features)
        confidence: (N, n_features) values in [0, 1]

    Returns:
        allocated: (N, n_features) — allocated resources
    """
    allocated = current + confidence * (predicted - current)
    # Ensure non-negative allocations
    return np.clip(allocated, 0.0, None)


def run_allocation_pipeline(predictions: np.ndarray,
                            actuals: np.ndarray,
                            confidence: np.ndarray) -> dict:
    """
    Run the full allocation pipeline.

    Uses first-step predictions (horizon index 0) for allocation decisions.

    Args:
        predictions: (N, horizon, features)
        actuals:     (N, horizon, features)
        confidence:  (N, horizon, features)

    Returns:
        dict with 'allocated', 'labels', 'predicted', 'current' arrays
    """
    # Use first prediction step
    pred_step = predictions[:, 0, :]    # (N, features)
    curr_step = actuals[:, 0, :]        # (N, features)
    conf_step = confidence[:, 0, :]     # (N, features)

    labels = classify_vms(pred_step, curr_step)
    allocated = allocate_resources(curr_step, pred_step, conf_step)

    n_overloaded = labels.sum()
    print(f"[Allocation] VMs classified: {len(labels)} total, "
          f"{n_overloaded} overloaded ({100*n_overloaded/max(len(labels),1):.1f}%)")
    print(f"[Allocation] Mean allocated — "
          f"CPU: {allocated[:,0].mean():.3f}, "
          f"Mem: {allocated[:,1].mean():.3f}, "
          f"Storage: {allocated[:,2].mean():.3f}")

    return {
        "allocated": allocated,
        "labels": labels,
        "predicted": pred_step,
        "current": curr_step,
        "confidence": conf_step,
    }
