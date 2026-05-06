# Confidence-Aware Multi-Model VM Placement with Energy Optimization (FP-PC-CA-E)

A production-grade system that predicts VM resource usage using deep learning (GRU, Informer, PatchTST), computes confidence scores, and performs energy-aware VM placement with adaptive failure handling and server consolidation.

## Architecture

```
Alibaba Trace → Preprocess → [GRU / Informer / PatchTST] → Confidence
    → Allocation → Energy-Aware Placement → Failure Handling → Consolidation
```

**Pipeline stages (executed by `python main.py`):**

| Step | Description |
|------|-------------|
| 1 | GPU detection and device setup |
| 2 | Alibaba dataset download and validation |
| 3 | Preprocessing (normalize, sliding window maps) |
| 4 | Train GRU — 150 epochs, GPU, milestone checkpoints |
| 5 | Train Informer — 150 epochs, GPU, milestone checkpoints |
| 6 | Train PatchTST — 150 epochs, GPU, milestone checkpoints |
| 7 | **Milestone evaluation** — all 10 metrics at [30,50,70,90,110,130,150] |
| 8 | **Comparison graphs** — grouped bars, line plots, summary panels |
| 9 | Placement simulation — energy-aware placement with best model |
| 10 | CloudSim Plus export — CSV/JSON + Java simulation template |

---

## Dataset Policy

> **Only real Alibaba Cluster Trace v2018 data is used.**
> No synthetic data generation, no artificial workload simulation, no fake traces.

- **Source:** [Alibaba Cluster Trace v2018](https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2018)
- **File:** `machine_usage.tar.gz` (~1.7 GB)
- **Fields used:**
  - `cpu_util_percent` — CPU utilization [0, 100]
  - `mem_util_percent` — Memory utilization [0, 100]
  - `disk_io_percent` — Storage/disk I/O utilization [0, 100]

If the dataset cannot be downloaded automatically, the pipeline will raise an error with instructions for manual download.

---

## GPU Requirements

- **CUDA GPU required** for efficient training of Informer and PatchTST
- GPU handling is limited to the **training scripts only** (GRU, Informer, PatchTST)
- Mixed precision training (FP16) via `torch.cuda.amp`
- cuDNN benchmarking enabled for optimized convolutions

### Recommended Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA GTX 1060 | NVIDIA RTX 3060+ |
| VRAM | 4 GB | 8 GB+ |
| RAM | 8 GB | 16 GB+ |
| Storage | 5 GB free | 10 GB free |

### Verify GPU availability

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

---

## Setup

```bash
# Clone the repository
git clone https://github.com/srivallisv/vm-placement-project.git
cd vm-placement-project

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python main.py
```

### Debug Mode (default)

The project ships with `DEBUG_MODE = True` in `config.py` for safe first execution:
- Loads only 50 machines / 500K rows from the real Alibaba traces
- Trains all 3 models for 150 epochs with batch_size=64
- Full pipeline completes in ~30-60 minutes on GPU

To run at full scale, edit `config.py`:
```python
DEBUG_MODE = False
```

---

## Training Strategy

- **Epochs:** 150 (no early stopping)
- **Milestones:** [30, 50, 70, 90, 110, 130, 150]
- **Scheduler:** Cosine annealing over full 150 epochs
- **Gradient clipping:** max_norm=1.0
- **Mixed precision:** FP16 autocast + GradScaler (GPU only)
- **Checkpoints:** Saved at each milestone epoch

All models train to completion — no patience logic or best-loss termination.

---

## Milestone-Based Evaluation

After training, each model is evaluated at every milestone epoch by loading its checkpoint and computing **10 metrics**:

| Category | Metrics |
|----------|---------|
| Prediction Accuracy | MAE, RMSE, MAPE |
| Resource Allocation Quality | Accuracy, Precision, AUC |
| Energy Efficiency | Energy Consumption, Active Servers |
| Consolidation & Failures | Migration Count, Failure Rate |

**How it works:**
1. For each model (GRU, Informer, PatchTST):
2. For each milestone (30, 50, 70, 90, 110, 130, 150):
   - Load the saved checkpoint
   - Run inference on the test set
   - Compute prediction metrics (MAE, RMSE, MAPE)
   - Compute confidence via MC Dropout → classify VMs → Accuracy, Precision, AUC
   - Run placement simulation → Energy, Migrations, Failure Rate, Active Servers
3. Generate comparison graphs and tables

**No per-epoch comparisons are generated** — only milestone-based comparisons.

### Generated Comparison Graphs

| Graph Type | Count | Description |
|------------|-------|-------------|
| `line_*.png` | 10 | Per-metric line plots (x=milestone, y=value, 3 model lines) |
| `bar_*.png` | 10 | Per-metric grouped bar charts at each milestone |
| `group_*.png` | 4 | Multi-panel summaries by category (Prediction, Allocation, Energy, Consolidation) |
| `final_combined_comparison.png` | 1 | Combined bar chart at final milestone (epoch 150) |

---

## Models

| Model | Architecture | Loss | LR |
|-------|-------------|------|-----|
| GRU | Bidirectional, 2 layers, 128 hidden | MSE | 1e-3 |
| Informer | ProbSparse attention, 3 enc + 2 dec layers | Smooth L1 | 1e-4 |
| PatchTST | Channel-independent, patch_len=8, stride=4 | MSE | 3e-4 |

---

## Key Formulas

**Confidence Score:**
```
Base_Confidence = 1 - |Predicted - Actual| / (|Actual| + ε)
Variance_penalty = 1 - min(Variance / max_variance, 1.0)
Final = 0.7 × Base + 0.3 × Variance_penalty
```

**Resource Allocation:**
```
Allocated = Current + Confidence × (Predicted - Current)
```

**Power Model:**
```
P = P_idle + U × (P_max - P_idle)    (P_idle=150W, P_max=400W)
```

---

## CloudSim Plus Integration

The pipeline exports data for use with [CloudSim Plus](https://cloudsimplus.org/) (Java-based cloud simulation):

| Export File | Description |
|-------------|-------------|
| `outputs/cloudsim/vm_workloads.csv` | VM resource demands (denormalized to %) |
| `outputs/cloudsim/host_config.json` | Server specifications (CPU, RAM, Storage, Power) |
| `outputs/cloudsim/placement_map.csv` | VM-to-Host placement mapping |
| `outputs/cloudsim/consolidation_events.json` | Migration and consolidation events |
| `outputs/cloudsim/VmPlacementSimulation.java` | Starter Java simulation template |

To use: copy `outputs/cloudsim/` to your CloudSim Plus project, add the Java template to your source tree, and compile with CloudSim Plus 7.x+ on the classpath.

---

## Project Structure

```
vm-placement-project/
├── config.py                           # Central configuration
├── main.py                             # Master pipeline (10 steps)
├── requirements.txt                    # Dependencies
├── preprocessing/                      # Data download, cleaning, normalization
│   ├── download_dataset.py             # Alibaba trace download (no synthetic fallback)
│   ├── preprocess.py                   # Normalize + window position maps
│   └── dataset.py                      # Lazy sliding-window Dataset (memory-mapped)
├── models/                             # Deep learning architectures
│   ├── gru.py                          # Bidirectional GRU
│   ├── informer.py                     # ProbSparse self-attention encoder-decoder
│   └── patchtst.py                     # Channel-independent patch transformer
├── training/                           # GPU training scripts
│   ├── trainer.py                      # Shared training loop (autocast, GradScaler)
│   ├── train_gru.py
│   ├── train_informer.py
│   └── train_patchtst.py
├── confidence/                         # MC Dropout confidence scoring
│   └── confidence_score.py
├── allocation/                         # Confidence-based resource allocation
│   └── allocation_engine.py
├── placement/                          # Energy-aware VM placement
│   ├── energy_model.py                 # Linear power model
│   └── placement_engine.py             # Min-ΔP server selection
├── failure/                            # Adaptive failure handling
│   └── failure_handler.py              # Confidence decay + retry + Best Fit
├── consolidation/                      # Server consolidation
│   └── consolidation_engine.py
├── evaluation/                         # Metrics and visualization
│   ├── metrics.py                      # MAE, RMSE, MAPE, classification, base graphs
│   ├── milestone_evaluator.py          # Full evaluation at each milestone checkpoint
│   └── comparison_graphs.py            # Milestone comparison visualizations
├── cloudsim/                           # CloudSim Plus integration
│   └── cloudsim_exporter.py            # CSV/JSON export + Java template
├── utils/                              # Helpers (seeding, timing, logging)
│   └── helpers.py
├── checkpoints/                        # Model checkpoints (7 milestones × 3 models)
└── outputs/
    ├── logs/                           # Full 150-epoch training logs per model
    ├── metrics/                        # Per-milestone metric CSVs
    ├── predictions/                    # Prediction subset summaries
    ├── graphs/                         # Base evaluation graphs
    │   └── comparisons/                # Milestone comparison graphs (25 plots)
    ├── comparison_tables/              # Per-model milestone CSVs + text table
    ├── final_milestone_comparison.csv  # All models × milestones × metrics
    └── cloudsim/                       # CloudSim Plus export files
```

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| No synthetic data | Ensures all experiments use only real cloud traces |
| No early stopping | Required for unbiased convergence analysis |
| GPU in training only | Other modules use CPU/numpy — avoids unnecessary GPU overhead |
| Mixed precision training | Reduces VRAM usage and accelerates training |
| Lazy window generation | Memory-safe — windows computed on-the-fly in Dataset.\_\_getitem\_\_() |
| Milestone-only comparisons | Avoids 150×3 evaluations; 7 milestones give clear convergence picture |
| Monte Carlo Dropout | Lightweight uncertainty estimation without ensemble overhead |
| Channel-independent PatchTST | Official architecture; reduces overfitting |
| Linear power model | Standard in literature (Beloglazov, 2012) |
| Cosine annealing over 150 epochs | Smooth LR decay for stable convergence |
| Confidence decay 0.8 | Balanced between aggressive reduction and stability |

---

## Outputs Summary

After a full run, find:

| Output | Location |
|--------|----------|
| Model checkpoints | `checkpoints/` — 21 files (7 milestones × 3 models) |
| Training logs | `outputs/logs/` — full 150-epoch CSV per model |
| Milestone metrics | `outputs/metrics/` — per-milestone CSVs |
| Comparison graphs | `outputs/graphs/comparisons/` — 25 visualizations |
| Comparison tables | `outputs/comparison_tables/` — per-model CSVs + text table |
| Final comparison | `outputs/final_milestone_comparison.csv` |
| Prediction summaries | `outputs/predictions/` — subset at milestones |
| CloudSim exports | `outputs/cloudsim/` — CSV, JSON, Java template |

## License

MIT
