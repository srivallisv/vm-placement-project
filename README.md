# Confidence-Aware Multi-Model VM Placement with Energy Optimization (FP-PC-CA-E)

A production-grade system that predicts VM resource usage using deep learning (GRU, Informer, PatchTST), computes confidence scores, and performs energy-aware VM placement with adaptive failure handling and server consolidation.

## Architecture

```
Alibaba Trace → Preprocess → [GRU / Informer / PatchTST] → Confidence
    → Allocation → Energy-Aware Placement → Failure Handling → Consolidation
```

**Pipeline stages:**
1. **Prediction** — 3 deep learning models forecast CPU, Memory, Storage usage
2. **Confidence** — Error + MC Dropout variance → confidence scores
3. **Allocation** — Confidence-weighted resource interpolation
4. **Placement** — Minimum incremental power server selection
5. **Failure Handling** — Adaptive confidence decay with retry
6. **Consolidation** — Migrate VMs off underutilized servers

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

## Project Structure

```
vm-placement-project/
├── config.py                  # Central configuration
├── main.py                    # Master pipeline script
├── requirements.txt           # Dependencies
├── preprocessing/             # Data download, cleaning, normalization
├── models/                    # GRU, Informer, PatchTST architectures
├── training/                  # GPU training scripts (150 epochs, no early stopping)
├── confidence/                # MC Dropout confidence scoring
├── allocation/                # Confidence-based resource allocation
├── placement/                 # Energy-aware VM placement
├── failure/                   # Adaptive failure handling
├── consolidation/             # Server consolidation
├── evaluation/                # Metrics and graph generation
├── utils/                     # Helpers (seeding, timing, logging)
├── checkpoints/               # Model checkpoints at milestones
└── outputs/                   # Metrics, predictions, logs, graphs
```

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| No synthetic data | Ensures all experiments use only real cloud traces |
| No early stopping | Required for unbiased convergence analysis |
| GPU acceleration | Necessary for efficient transformer training on large-scale traces |
| Mixed precision training | Reduces VRAM usage and accelerates training |
| Lazy window generation | Memory-safe — windows computed on-the-fly in Dataset.__getitem__() |
| Monte Carlo Dropout | Lightweight uncertainty estimation without ensemble overhead |
| Channel-independent PatchTST | Official architecture; reduces overfitting |
| Linear power model | Standard in literature (Beloglazov, 2012) |
| Cosine annealing over 150 epochs | Smooth LR decay for stable convergence |
| Confidence decay 0.8 | Balanced between aggressive reduction and stability |

---

## Outputs

After a full run, find:
- `checkpoints/` — 21 model checkpoints (7 milestones × 3 models)
- `outputs/logs/` — Full 150-epoch training logs per model
- `outputs/metrics/` — Per-milestone and final summary CSVs
- `outputs/graphs/` — 10+ evaluation visualizations
- `outputs/predictions/` — Prediction subset summaries at milestones

## License

MIT
