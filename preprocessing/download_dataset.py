"""
download_dataset.py
Downloads and cleans the Alibaba Cluster Trace v2018 machine_usage dataset.
No synthetic data generation — raises RuntimeError if dataset is unavailable.
"""

import os
import sys
import tarfile
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# Column names for the Alibaba machine_usage CSV (no header in raw file)
RAW_COLUMNS = [
    "machine_id", "time_stamp", "cpu_util_percent",
    "mem_util_percent", "mem_gps", "mkpi",
    "net_in", "net_out", "disk_io_percent",
]


def download_tarball(url: str, dest_path: str) -> None:
    """Download a file with a progress bar. Raises RuntimeError on failure."""
    print(f"[Download] Downloading from {url} ...")
    try:
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
    except Exception as exc:
        raise RuntimeError(
            "Alibaba Cluster Trace dataset could not be downloaded. "
            "Please manually download machine_usage.tar.gz and place it in "
            f"{config.RAW_DATA_DIR}/\n"
            f"Original error: {exc}"
        )

    total = int(resp.headers.get("content-length", 0))
    with open(dest_path, "wb") as fh, tqdm(
        total=total, unit="B", unit_scale=True, desc="machine_usage.tar.gz"
    ) as bar:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            fh.write(chunk)
            bar.update(len(chunk))
    print(f"[Download] Saved to {dest_path}")


def extract_tarball(tar_path: str, extract_dir: str) -> list:
    """Extract all CSV files from a tarball and return their paths."""
    print(f"[Extract] Extracting {tar_path} ...")
    csv_paths = []
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            if member.name.endswith(".csv"):
                tar.extract(member, path=extract_dir)
                csv_paths.append(os.path.join(extract_dir, member.name))
    if not csv_paths:
        raise RuntimeError(
            f"No CSV files found inside {tar_path}. "
            "The archive may be corrupted — please re-download."
        )
    print(f"[Extract] Found {len(csv_paths)} CSV file(s)")
    return sorted(csv_paths)


def load_and_clean(csv_paths: list) -> pd.DataFrame:
    """Load CSV files, select required columns, and clean invalid values."""
    frames = []
    for path in csv_paths:
        print(f"[Load] Reading {os.path.basename(path)} ...")
        df = pd.read_csv(path, header=None, names=RAW_COLUMNS, low_memory=False)
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    print(f"[Load] Total rows loaded: {len(df):,}")

    # Keep only required columns
    keep_cols = ["machine_id", "time_stamp"] + config.FEATURES
    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Required columns not found in dataset: {missing}. "
            f"Available columns: {list(df.columns)}"
        )
    df = df[keep_cols].copy()

    # --- DEBUG MODE: subset the data ---
    if config.DEBUG_MODE:
        machines = df["machine_id"].unique()
        if len(machines) > config.MAX_MACHINES:
            machines = machines[: config.MAX_MACHINES]
            df = df[df["machine_id"].isin(machines)].copy()
            print(f"[Debug] Limited to {config.MAX_MACHINES} machines")
        if len(df) > config.MAX_ROWS:
            df = df.head(config.MAX_ROWS)
            print(f"[Debug] Limited to {config.MAX_ROWS:,} rows")

    # --- Clean invalid values ---
    for col in config.FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col] < 0, col] = np.nan
        df.loc[df[col] > 100, col] = np.nan

    # Sort chronologically within each machine
    df.sort_values(["machine_id", "time_stamp"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Forward-fill within each machine, then fill remaining with column median
    df[config.FEATURES] = df.groupby("machine_id")[config.FEATURES].transform(
        lambda s: s.ffill()
    )
    for col in config.FEATURES:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)

    print(f"[Clean] Final dataset: {len(df):,} rows, "
          f"{df['machine_id'].nunique()} machines")
    return df


def run_download_pipeline() -> pd.DataFrame:
    """
    Full download → extract → clean pipeline.
    Returns cleaned DataFrame. Raises RuntimeError if dataset unavailable.
    """
    os.makedirs(config.RAW_DATA_DIR, exist_ok=True)

    # 1. Check if cleaned file already exists
    if os.path.exists(config.CLEAN_DATA_PATH):
        print(f"[Download] Cleaned dataset found at {config.CLEAN_DATA_PATH}")
        df = pd.read_csv(config.CLEAN_DATA_PATH, low_memory=False)
        if config.DEBUG_MODE:
            machines = df["machine_id"].unique()
            if len(machines) > config.MAX_MACHINES:
                machines = machines[: config.MAX_MACHINES]
                df = df[df["machine_id"].isin(machines)].copy()
            if len(df) > config.MAX_ROWS:
                df = df.head(config.MAX_ROWS)
        return df

    # 2. Check for existing CSV(s) in raw dir
    raw_csvs = sorted(
        [os.path.join(config.RAW_DATA_DIR, f)
         for f in os.listdir(config.RAW_DATA_DIR)
         if f.endswith(".csv")]
    )
    if raw_csvs:
        print(f"[Download] Found {len(raw_csvs)} raw CSV(s), skipping download")
        df = load_and_clean(raw_csvs)
        df.to_csv(config.CLEAN_DATA_PATH, index=False)
        return df

    # 3. Check for existing tarball
    tar_path = os.path.join(config.RAW_DATA_DIR, "machine_usage.tar.gz")
    if not os.path.exists(tar_path):
        # 4. Attempt download
        download_tarball(config.DATASET_URL, tar_path)

    # 5. Extract and clean
    csv_paths = extract_tarball(tar_path, config.RAW_DATA_DIR)
    df = load_and_clean(csv_paths)

    # 6. Save cleaned CSV
    os.makedirs(os.path.dirname(config.CLEAN_DATA_PATH), exist_ok=True)
    df.to_csv(config.CLEAN_DATA_PATH, index=False)
    print(f"[Download] Saved cleaned CSV to {config.CLEAN_DATA_PATH}")
    return df


if __name__ == "__main__":
    run_download_pipeline()
