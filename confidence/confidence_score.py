"""
confidence_score.py
Computes confidence scores for model predictions using error-based scoring
and Monte Carlo Dropout variance estimation.

Note: MC Dropout forward passes use the model on whatever device it's loaded on.
The final confidence arrays are returned as numpy on CPU.
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def compute_error_confidence(predictions: np.ndarray,
                             actuals: np.ndarray) -> np.ndarray:
    """
    Error-based confidence per sample per feature.

    Formula:
        Base_Confidence = 1 - |Predicted - Actual| / (|Actual| + ε)
        Clipped to [0, 1]

    Args:
        predictions: (N, horizon, n_features) or (N, n_features)
        actuals: same shape

    Returns:
        confidence: same shape, values in [0, 1]
    """
    eps = 1e-8
    error = np.abs(predictions - actuals)
    confidence = 1.0 - error / (np.abs(actuals) + eps)
    return np.clip(confidence, 0.0, 1.0)


def compute_mc_variance(model: torch.nn.Module,
                        data_loader,
                        device: torch.device,
                        n_passes: int = None) -> np.ndarray:
    """
    Monte Carlo Dropout: run N stochastic forward passes with dropout enabled
    to estimate prediction variance.

    Args:
        model: trained model (must have Dropout layers)
        data_loader: test data loader
        device: torch device the model is on
        n_passes: number of MC forward passes (default from config)

    Returns:
        variance: (N, horizon, n_features) — variance across MC passes
    """
    if n_passes is None:
        n_passes = config.MC_DROPOUT_PASSES

    # Enable dropout during inference
    model.train()

    all_variances = []
    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device, non_blocking=True)
            mc_preds = []
            for _ in range(n_passes):
                pred = model(x)
                mc_preds.append(pred.cpu().numpy())
            # Stack: (n_passes, batch, horizon, features)
            stacked = np.stack(mc_preds, axis=0)
            # Variance across passes: (batch, horizon, features)
            var = stacked.var(axis=0)
            all_variances.append(var)

    model.eval()
    return np.concatenate(all_variances, axis=0)


def compute_combined_confidence(predictions: np.ndarray,
                                actuals: np.ndarray,
                                variance: np.ndarray,
                                alpha: float = None) -> np.ndarray:
    """
    Combine error-based confidence with variance penalty.

    Formula:
        Variance_penalty = 1 - min(Variance / max_variance, 1.0)
        Final = α × Base_Confidence + (1-α) × Variance_penalty
        Clipped to [0, 1]

    Args:
        predictions, actuals: (N, horizon, features)
        variance: (N, horizon, features) — from MC Dropout
        alpha: blending weight (default from config)

    Returns:
        confidence: (N, horizon, features) in [0, 1]
    """
    if alpha is None:
        alpha = config.CONFIDENCE_ALPHA

    base = compute_error_confidence(predictions, actuals)

    max_var = variance.max() if variance.max() > 0 else 1.0
    var_penalty = 1.0 - np.clip(variance / max_var, 0.0, 1.0)

    combined = alpha * base + (1.0 - alpha) * var_penalty
    return np.clip(combined, 0.0, 1.0)


def run_confidence_pipeline(model, test_loader, device,
                            predictions, actuals) -> np.ndarray:
    """
    Full confidence computation pipeline.

    Args:
        model: trained model on device
        test_loader: DataLoader for test split
        device: torch device
        predictions: (N, horizon, features) numpy array
        actuals: (N, horizon, features) numpy array

    Returns:
        confidence: (N, horizon, features) in [0, 1]
    """
    print("[Confidence] Computing MC Dropout variance ...")
    variance = compute_mc_variance(model, test_loader, device)

    # Align lengths (MC may cover subset)
    n = min(len(predictions), len(variance))
    predictions = predictions[:n]
    actuals = actuals[:n]
    variance = variance[:n]

    print("[Confidence] Computing combined confidence scores ...")
    confidence = compute_combined_confidence(predictions, actuals, variance)
    print(f"[Confidence] Mean confidence: {confidence.mean():.4f}, "
          f"Std: {confidence.std():.4f}")
    return confidence
