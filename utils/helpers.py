"""
helpers.py
Utility functions: seeding, logging, timing, directory setup.
"""

import os
import sys
import time
import random
import logging
import functools
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def set_seed(seed: int = None):
    """Set random seed for reproducibility across all libraries."""
    if seed is None:
        seed = config.RANDOM_SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[Seed] Set random seed to {seed}")


def setup_logging(name: str = "vm-placement") -> logging.Logger:
    """Configure and return a logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def create_output_dirs():
    """Create all output directories."""
    dirs = [
        config.CHECKPOINT_DIR,
        config.METRICS_DIR,
        config.PREDICTIONS_DIR,
        config.LOGS_DIR,
        config.GRAPHS_DIR,
        config.RAW_DATA_DIR,
        config.PROCESSED_DATA_DIR,
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def timer(func):
    """Decorator to time a function and print elapsed time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - t0
        mins, secs = divmod(elapsed, 60)
        print(f"  ⏱  {func.__name__} completed in {int(mins)}m {secs:.1f}s")
        return result
    return wrapper


def print_banner(title: str):
    """Print a formatted section banner."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
