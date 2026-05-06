"""
preprocess.py
Normalizes cleaned data and builds window-position maps for lazy loading.
No large in-memory array materialization — windows are generated on-the-fly
by the Dataset class using these precomputed position maps.
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def normalize_and_save(df: pd.DataFrame) -> None:
    """
    Mean-std normalize features, save the normalized data array and scaler
    params, then compute per-split window position maps for lazy loading.
    """
    proc_dir = config.PROCESSED_DATA_DIR
    os.makedirs(proc_dir, exist_ok=True)

    # --- Compute normalization parameters ---
    features = df[config.FEATURES].values.astype(np.float32)
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std == 0] = 1.0  # avoid division by zero

    np.savez(
        os.path.join(proc_dir, "scaler_params.npz"),
        mean=mean, std=std,
    )
    print(f"[Preprocess] Scaler — mean: {mean}, std: {std}")

    # --- Normalize ---
    normalized = ((features - mean) / std).astype(np.float32)

    # --- Group by machine and record boundaries ---
    machine_ids = df["machine_id"].values
    unique_machines = pd.unique(machine_ids)  # preserves order
    print(f"[Preprocess] Processing {len(unique_machines)} machines ...")

    # Save the full normalized array (will be memory-mapped at training time)
    data_path = os.path.join(proc_dir, "normalized_data.npy")
    np.save(data_path, normalized)

    # --- Build window position maps per split ---
    win_len = config.INPUT_WINDOW + config.OUTPUT_HORIZON
    train_windows = []
    val_windows = []
    test_windows = []

    for mid in unique_machines:
        mask = machine_ids == mid
        indices = np.where(mask)[0]
        n = len(indices)
        if n < win_len:
            continue  # skip machines with insufficient data

        start_idx = indices[0]  # first row of this machine in the array

        # Number of valid windows for this machine
        n_windows = n - win_len + 1
        if n_windows <= 0:
            continue

        # Chronological split of windows
        n_train = max(1, int(n_windows * config.TRAIN_RATIO))
        n_val = max(1, int(n_windows * config.VAL_RATIO))
        # n_test gets the remainder

        for w in range(n_windows):
            # Each entry: [absolute_offset_in_array, <unused>]
            # The window starts at  start_idx + w  in the normalized array
            entry = np.array([start_idx + w], dtype=np.int64)
            if w < n_train:
                train_windows.append(entry)
            elif w < n_train + n_val:
                val_windows.append(entry)
            else:
                test_windows.append(entry)

    for name, windows in [
        ("train", train_windows),
        ("val", val_windows),
        ("test", test_windows),
    ]:
        arr = np.array(windows, dtype=np.int64) if windows else np.empty((0, 1), dtype=np.int64)
        path = os.path.join(proc_dir, f"{name}_windows.npy")
        np.save(path, arr)
        print(f"[Preprocess] {name}: {len(arr):,} windows")

    print("[Preprocess] Done — files saved to", proc_dir)


def run_preprocess_pipeline(df: pd.DataFrame) -> None:
    """Entry point: normalize and create window maps."""
    normalize_and_save(df)


if __name__ == "__main__":
    df = pd.read_csv(config.CLEAN_DATA_PATH, low_memory=False)
    run_preprocess_pipeline(df)
