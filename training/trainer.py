"""
trainer.py
Shared training loop for all three models.
GPU handling (device transfer, autocast, GradScaler, cuDNN) lives here
and in the per-model train scripts. No early stopping — trains to MAX_EPOCHS.
"""

import os
import sys
import csv
import time
import math
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def setup_device():
    """Detect GPU/CPU and print device info. Returns torch.device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device        : {device}")
    if torch.cuda.is_available():
        print(f"  GPU           : {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"  VRAM          : {props.total_mem / 1e9:.2f} GB")
        print(f"  CUDA version  : {torch.version.cuda}")
        torch.backends.cudnn.benchmark = config.CUDNN_BENCHMARK
    else:
        print("  WARNING: No CUDA GPU detected — training will be slow.")
    return device


def train_model(
    model: nn.Module,
    model_name: str,
    train_loader,
    val_loader,
    test_loader,
    criterion: nn.Module,
    learning_rate: float,
    device: torch.device,
) -> dict:
    """
    Full training loop with:
      - Mixed precision (autocast + GradScaler) — GPU only
      - Cosine annealing LR scheduler
      - Gradient clipping
      - Milestone checkpointing and metric logging
      - NO early stopping

    Returns dict of final metrics.
    """
    # Move model to device
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.MAX_EPOCHS
    )

    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Prepare output directories
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.METRICS_DIR, exist_ok=True)
    os.makedirs(config.PREDICTIONS_DIR, exist_ok=True)
    os.makedirs(config.LOGS_DIR, exist_ok=True)

    # CSV log for all epochs
    log_path = os.path.join(config.LOGS_DIR, f"{model_name}_training_log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "train_loss", "val_loss", "val_mae", "val_rmse", "lr", "time_sec"
        ])

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "val_mae": [], "val_rmse": []}

    print(f"\n{'='*60}")
    print(f"  Training {model_name.upper()} — {config.MAX_EPOCHS} epochs, "
          f"batch_size={config.BATCH_SIZE}")
    print(f"  Milestones: {config.MILESTONE_EPOCHS}")
    print(f"{'='*60}")

    for epoch in range(1, config.MAX_EPOCHS + 1):
        t0 = time.time()

        # ---- Training phase ----
        model.train()
        train_loss_sum = 0.0
        train_samples = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(x)
                loss = criterion(pred, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=config.GRADIENT_CLIP_MAX_NORM
            )
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item() * x.size(0)
            train_samples += x.size(0)

        scheduler.step()
        train_loss_avg = train_loss_sum / max(train_samples, 1)

        # ---- Validation phase ----
        model.eval()
        val_loss_sum = 0.0
        val_mae_sum = 0.0
        val_se_sum = 0.0
        val_samples = 0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    pred = model(x)
                    loss = criterion(pred, y)
                val_loss_sum += loss.item() * x.size(0)
                val_mae_sum += (pred - y).abs().sum().item()
                val_se_sum += ((pred - y) ** 2).sum().item()
                val_samples += x.size(0)

        val_loss_avg = val_loss_sum / max(val_samples, 1)
        n_elements = max(val_samples * config.OUTPUT_HORIZON * config.NUM_FEATURES, 1)
        val_mae = val_mae_sum / n_elements
        val_rmse = math.sqrt(val_se_sum / n_elements)

        # GPU memory cleanup after validation
        if device.type == "cuda":
            torch.cuda.empty_cache()

        elapsed = time.time() - t0

        # Track history
        history["train_loss"].append(train_loss_avg)
        history["val_loss"].append(val_loss_avg)
        history["val_mae"].append(val_mae)
        history["val_rmse"].append(val_rmse)

        # Track best
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg

        # Log every epoch
        current_lr = scheduler.get_last_lr()[0]
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, f"{train_loss_avg:.6f}", f"{val_loss_avg:.6f}",
                f"{val_mae:.6f}", f"{val_rmse:.6f}",
                f"{current_lr:.8f}", f"{elapsed:.1f}"
            ])

        # Print progress
        if epoch % 10 == 0 or epoch in config.MILESTONE_EPOCHS or epoch == 1:
            print(f"  Epoch {epoch:>3d}/{config.MAX_EPOCHS} | "
                  f"train_loss={train_loss_avg:.5f}  val_loss={val_loss_avg:.5f}  "
                  f"MAE={val_mae:.5f}  RMSE={val_rmse:.5f}  "
                  f"lr={current_lr:.2e}  [{elapsed:.1f}s]")

        # ---- Milestone: save checkpoint + metrics + prediction summary ----
        if epoch in config.MILESTONE_EPOCHS:
            # Checkpoint
            ckpt_path = os.path.join(
                config.CHECKPOINT_DIR, f"{model_name}_epoch{epoch}.pt"
            )
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": train_loss_avg,
                "val_loss": val_loss_avg,
            }, ckpt_path)

            # Milestone metrics CSV
            metrics_path = os.path.join(
                config.METRICS_DIR, f"{model_name}_epoch{epoch}_metrics.csv"
            )
            with open(metrics_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["metric", "value"])
                w.writerow(["train_loss", f"{train_loss_avg:.6f}"])
                w.writerow(["val_loss", f"{val_loss_avg:.6f}"])
                w.writerow(["val_mae", f"{val_mae:.6f}"])
                w.writerow(["val_rmse", f"{val_rmse:.6f}"])

            # Save prediction statistics (NOT full tensors) for a subset
            _save_prediction_summary(
                model, test_loader, device, use_amp, model_name, epoch
            )

            print(f"  >>> Milestone {epoch}: checkpoint + metrics + predictions saved")

    print(f"\n  {model_name.upper()} training complete. "
          f"Best val_loss: {best_val_loss:.6f}")
    return history


def _save_prediction_summary(
    model, test_loader, device, use_amp, model_name, epoch
):
    """
    Save prediction summary statistics and a small subset of actual predictions
    at a milestone epoch. Avoids saving the full prediction tensor.
    """
    model.eval()
    all_preds = []
    all_targets = []
    n_collected = 0
    subset = config.PRED_SAVE_SUBSET

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(x)
            # Move back to CPU for saving
            pred_np = pred.cpu().numpy()
            y_np = y.numpy()

            remaining = subset - n_collected
            if remaining <= 0:
                break
            take = min(pred_np.shape[0], remaining)
            all_preds.append(pred_np[:take])
            all_targets.append(y_np[:take])
            n_collected += take

    if not all_preds:
        return

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # Save the subset
    save_path = os.path.join(
        config.PREDICTIONS_DIR, f"{model_name}_epoch{epoch}_subset.npz"
    )
    np.savez_compressed(
        save_path,
        predictions=preds,
        targets=targets,
        # Summary statistics
        pred_mean=preds.mean(axis=0),
        pred_std=preds.std(axis=0),
        target_mean=targets.mean(axis=0),
        target_std=targets.std(axis=0),
    )
