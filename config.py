"""
Central configuration for the VM Placement project.
All constants are defined here and imported by other modules.
Adjust DEBUG_MODE and related settings for quick testing vs full-scale runs.
"""

import os

# === PROJECT ROOT ===
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# === DEBUG MODE ===
# Set True for quick testing with a real data subset; False for full-scale runs.
DEBUG_MODE = True
MAX_ROWS = 500_000          # Max rows loaded from Alibaba traces in debug mode
MAX_MACHINES = 50           # Max unique machines to process in debug mode

# === DATASET ===
DATASET_URL = (
    "http://aliopentrace.oss-cn-beijing.aliyuncs.com/"
    "v2018Traces/machine_usage.tar.gz"
)
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
CLEAN_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "machine_usage_clean.csv")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# === PREPROCESSING ===
INPUT_WINDOW = 48           # Number of past timesteps as model input
OUTPUT_HORIZON = 6          # Number of future timesteps to predict
FEATURES = ["cpu_util_percent", "mem_util_percent", "disk_io_percent"]
NUM_FEATURES = len(FEATURES)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# === TRAINING ===
MAX_EPOCHS = 150
MILESTONE_EPOCHS = [30, 50, 70, 90, 110, 130, 150]
BATCH_SIZE = 64
GRADIENT_CLIP_MAX_NORM = 1.0
LEARNING_RATES = {
    "gru": 1e-3,
    "informer": 1e-4,
    "patchtst": 3e-4,
}

# === GPU / DATALOADER ===
NUM_WORKERS = 0             # Set >0 on Linux for faster loading; 0 is safest on Windows
PIN_MEMORY = True
PERSISTENT_WORKERS = False  # Set True when NUM_WORKERS > 0
CUDNN_BENCHMARK = True

# === PREDICTION SAVING ===
PRED_SAVE_SUBSET = 1000     # Save predictions for first N test samples at milestones

# === CONFIDENCE ===
MC_DROPOUT_PASSES = 20      # Number of Monte Carlo forward passes
CONFIDENCE_ALPHA = 0.7      # Weight for error-based confidence vs variance penalty

# === PLACEMENT / ENERGY ===
P_IDLE = 150                # Server idle power (Watts)
P_MAX = 400                 # Server max power (Watts)
NUM_SERVERS = 20            # Number of servers in simulation
SERVER_CPU_CAPACITY = 100.0
SERVER_MEM_CAPACITY = 100.0
SERVER_STORAGE_CAPACITY = 100.0
OVERLOAD_THRESHOLD = 0.1    # Normalized threshold for VM overload classification

# === FAILURE HANDLING ===
CONFIDENCE_DECAY = 0.8      # Multiplicative decay on placement failure
MAX_RETRIES = 5

# === CONSOLIDATION ===
CONSOLIDATION_THRESHOLD = 0.20  # Servers below this avg utilization get consolidated

# === OUTPUTS ===
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
METRICS_DIR = os.path.join(OUTPUT_DIR, "metrics")
PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, "predictions")
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")
GRAPHS_DIR = os.path.join(OUTPUT_DIR, "graphs")

# === REPRODUCIBILITY ===
RANDOM_SEED = 42
