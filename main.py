"""
main.py
Master orchestration script for the Confidence-Aware Multi-Model
VM Placement with Energy Optimization (FP-PC-CA-E) pipeline.

Execution order:
    1. GPU detection
    2. Dataset download / validation
    3. Preprocessing
    4. GRU training (150 epochs, GPU)
    5. Informer training (150 epochs, GPU)
    6. PatchTST training (150 epochs, GPU)
    7. Confidence computation
    8. Resource allocation
    9. VM placement
   10. Failure handling
   11. Server consolidation
   12. Evaluation & graph generation
   13. CloudSim Plus export
"""

import os
import sys
import numpy as np
import torch

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from utils.helpers import set_seed, create_output_dirs, timer, print_banner
from preprocessing.download_dataset import run_download_pipeline
from preprocessing.preprocess import run_preprocess_pipeline
from preprocessing.dataset import create_dataloaders, LazyWindowDataset
from training.train_gru import run_gru_training
from training.train_informer import run_informer_training
from training.train_patchtst import run_patchtst_training
from allocation.allocation_engine import run_allocation_pipeline, classify_vms
from placement.placement_engine import create_server_fleet, run_placement
from placement.energy_model import compute_datacenter_power
from failure.failure_handler import handle_failures
from consolidation.consolidation_engine import consolidate_servers
from confidence.confidence_score import run_confidence_pipeline
from evaluation.metrics import (
    evaluate_predictions, plot_training_loss, plot_predictions_vs_actual,
    plot_confidence_distribution, plot_energy_over_time,
    plot_active_servers, plot_migrations, save_summary_csv,
)
from evaluation.milestone_evaluator import (
    run_milestone_evaluation, save_milestone_csvs, save_final_comparison_csv,
)
from evaluation.comparison_graphs import generate_all_comparison_graphs
from cloudsim.cloudsim_exporter import run_cloudsim_export


def get_test_predictions(model, test_loader, device):
    """Run inference on the test set and return (predictions, actuals) as numpy."""
    model.eval()
    all_preds, all_targets = [], []
    use_amp = device.type == "cuda"

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(x)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.numpy())

    return np.concatenate(all_preds), np.concatenate(all_targets)


def load_best_model(model_class, model_name, model_kwargs, device):
    """Load the final (epoch 150) checkpoint for a model."""
    ckpt_path = os.path.join(
        config.CHECKPOINT_DIR, f"{model_name}_epoch{config.MAX_EPOCHS}.pt"
    )
    model = model_class(**model_kwargs)

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Loaded checkpoint: {ckpt_path}")
    else:
        print(f"  WARNING: Checkpoint not found at {ckpt_path}, using last trained state")

    model.to(device)
    model.eval()
    return model


@timer
def step_download():
    """Step 1-2: Download and clean the Alibaba dataset."""
    print_banner("STEP 1: Dataset Download & Cleaning")
    return run_download_pipeline()


@timer
def step_preprocess(df):
    """Step 3: Preprocess — normalize, create window maps."""
    print_banner("STEP 2: Preprocessing")
    run_preprocess_pipeline(df)


@timer
def step_train_gru():
    """Step 4: Train GRU model."""
    print_banner("STEP 3: Training GRU")
    return run_gru_training()


@timer
def step_train_informer():
    """Step 5: Train Informer model."""
    print_banner("STEP 4: Training Informer")
    return run_informer_training()


@timer
def step_train_patchtst():
    """Step 6: Train PatchTST model."""
    print_banner("STEP 5: Training PatchTST")
    return run_patchtst_training()


def main():
    """Run the full FP-PC-CA-E pipeline."""
    print_banner("FP-PC-CA-E Pipeline")
    print(f"  DEBUG_MODE     : {config.DEBUG_MODE}")
    print(f"  MAX_EPOCHS     : {config.MAX_EPOCHS}")
    print(f"  BATCH_SIZE     : {config.BATCH_SIZE}")
    print(f"  MILESTONES     : {config.MILESTONE_EPOCHS}")

    if config.DEBUG_MODE:
        print(f"  MAX_MACHINES   : {config.MAX_MACHINES}")
        print(f"  MAX_ROWS       : {config.MAX_ROWS}")

    # Setup
    set_seed()
    create_output_dirs()

    # ================================================================
    # STEP 1-2: Dataset
    # ================================================================
    df = step_download()

    # ================================================================
    # STEP 3: Preprocessing
    # ================================================================
    step_preprocess(df)
    del df  # Free memory

    # ================================================================
    # STEPS 4-6: Training all models
    # ================================================================
    gru_history = step_train_gru()
    informer_history = step_train_informer()
    patchtst_history = step_train_patchtst()

    histories = {
        "gru": gru_history,
        "informer": informer_history,
        "patchtst": patchtst_history,
    }

    # ================================================================
    # STEP 7: Milestone Evaluation (all metrics at milestone epochs)
    # ================================================================
    print_banner("STEP 6: Milestone Evaluation")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_loader = torch.utils.data.DataLoader(
        LazyWindowDataset("test"),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    from models.gru import GRUModel
    from models.informer import InformerModel
    from models.patchtst import PatchTSTModel

    model_configs = {
        "gru": (GRUModel, {
            "n_features": config.NUM_FEATURES,
            "hidden_size": 128, "num_layers": 2,
            "dropout": 0.2, "output_horizon": config.OUTPUT_HORIZON,
        }),
        "informer": (InformerModel, {
            "n_features": config.NUM_FEATURES,
            "seq_len": config.INPUT_WINDOW, "pred_len": config.OUTPUT_HORIZON,
            "label_len": config.INPUT_WINDOW // 2,
            "d_model": 256, "n_heads": 8,
            "e_layers": 3, "d_layers": 2, "d_ff": 512, "dropout": 0.1,
        }),
        "patchtst": (PatchTSTModel, {
            "n_features": config.NUM_FEATURES,
            "seq_len": config.INPUT_WINDOW, "pred_len": config.OUTPUT_HORIZON,
            "patch_len": 8, "stride": 4,
            "d_model": 128, "n_heads": 4,
            "e_layers": 3, "d_ff": 256, "dropout": 0.2,
        }),
    }

    # Run full evaluation at every milestone for all 3 models
    milestone_results = run_milestone_evaluation(
        model_configs, test_loader, device
    )

    # Save milestone metric CSVs
    save_milestone_csvs(milestone_results)
    save_final_comparison_csv(milestone_results)

    # ================================================================
    # STEP 8: Comparison Graphs & Tables (milestone-based only)
    # ================================================================
    print_banner("STEP 7: Milestone Comparison Graphs")

    generate_all_comparison_graphs(milestone_results)

    # Also generate training loss curves from full epoch logs
    plot_training_loss(histories)

    # ================================================================
    # STEP 9: Placement Simulation (best model at epoch 150)
    # ================================================================
    print_banner("STEP 8: Placement Simulation")

    # Determine best model from final milestone
    last_epoch = config.MILESTONE_EPOCHS[-1]
    best_model_name = None
    best_rmse = float("inf")
    for m_name in milestone_results:
        if last_epoch in milestone_results[m_name]:
            rmse = milestone_results[m_name][last_epoch].get("RMSE", float("inf"))
            if rmse < best_rmse:
                best_rmse = rmse
                best_model_name = m_name

    if best_model_name is None:
        best_model_name = list(model_configs.keys())[0]
    print(f"  Best model (lowest RMSE at epoch {last_epoch}): {best_model_name.upper()}")

    # Load best model for placement simulation
    best_class, best_kwargs = model_configs[best_model_name]
    best_model = load_best_model(best_class, best_model_name, best_kwargs, device)
    best_predictions, best_actuals = get_test_predictions(best_model, test_loader, device)

    # Compute confidence for best model
    best_confidence = run_confidence_pipeline(
        best_model, test_loader, device, best_predictions, best_actuals
    )
    n = min(len(best_predictions), len(best_confidence))
    best_predictions = best_predictions[:n]
    best_actuals = best_actuals[:n]

    # Pred vs actual graphs for best model
    plot_predictions_vs_actual(best_predictions, best_actuals, best_model_name)
    plot_confidence_distribution(best_confidence)

    # Allocation
    alloc_result = run_allocation_pipeline(
        best_predictions, best_actuals, best_confidence
    )

    # Multi-timestep placement simulation
    n_vms = len(alloc_result["allocated"])
    vms_per_step = max(config.MAX_MACHINES if config.DEBUG_MODE else 100, 1)
    n_steps = max(1, n_vms // vms_per_step)

    energy_timeline = []
    active_timeline = []
    migration_timeline = []
    total_placements = 0
    total_failures = 0
    total_migrations = 0

    for step in range(n_steps):
        start = step * vms_per_step
        end = min(start + vms_per_step, n_vms)
        if start >= n_vms:
            break

        step_allocated = alloc_result["allocated"][start:end]
        step_labels = alloc_result["labels"][start:end]
        step_pred = alloc_result["predicted"][start:end]
        step_curr = alloc_result["current"][start:end]
        step_conf = alloc_result["confidence"][start:end]

        servers = create_server_fleet()
        placement_result = run_placement(servers, step_allocated, step_labels)
        total_placements += step_labels.sum()
        failures = placement_result["failures"]
        total_failures += len(failures)

        if failures:
            fail_result = handle_failures(
                servers, failures, step_pred, step_curr, step_conf
            )
            total_failures -= len(fail_result["recovered"])

        consol_result = consolidate_servers(servers, step_allocated)
        total_migrations += consol_result["migrations"]

        utils = np.array([s.cpu_util for s in servers if s.is_active])
        energy = compute_datacenter_power(utils) if len(utils) > 0 else 0.0
        energy_timeline.append(energy)
        active_timeline.append(consol_result["active_servers"])
        migration_timeline.append(consol_result["migrations"])

    # Simulation timeline graphs
    plot_energy_over_time(energy_timeline)
    plot_active_servers(active_timeline, config.NUM_SERVERS)
    plot_migrations(migration_timeline)

    system_metrics = {
        "total_placements": int(total_placements),
        "total_failures": total_failures,
        "failure_rate": total_failures / max(total_placements, 1),
        "total_migrations": total_migrations,
        "avg_energy": np.mean(energy_timeline) if energy_timeline else 0.0,
        "avg_active_servers": np.mean(active_timeline) if active_timeline else 0.0,
    }
    save_summary_csv({}, {}, system_metrics)

    # ================================================================
    # STEP 10: CloudSim Plus Export
    # ================================================================
    print_banner("STEP 9: CloudSim Plus Export")

    last_placements = placement_result["placements"] if 'placement_result' in dir() else {}
    last_consol = consol_result if 'consol_result' in dir() else {}
    run_cloudsim_export(
        predictions=best_predictions,
        actuals=best_actuals,
        confidence=best_confidence,
        placements=last_placements,
        allocated=alloc_result["allocated"],
        consolidation_result=last_consol,
    )

    # ================================================================
    # Final summary
    # ================================================================
    print_banner("PIPELINE COMPLETE")
    print(f"  Models trained       : GRU, Informer, PatchTST (all {config.MAX_EPOCHS} epochs)")
    print(f"  Milestones evaluated : {config.MILESTONE_EPOCHS}")
    print(f"  Best model           : {best_model_name.upper()} (RMSE={best_rmse:.5f})")
    print(f"  Total placements     : {int(total_placements)}")
    print(f"  Total failures       : {total_failures}")
    print(f"  Total migrations     : {total_migrations}")
    print(f"  Avg energy           : {system_metrics['avg_energy']:.1f} W")
    print(f"  Outputs              : {config.OUTPUT_DIR}")
    print(f"  Comparisons          : {config.GRAPHS_DIR}/comparisons/")
    print(f"  Comparison tables    : {config.OUTPUT_DIR}/comparison_tables/")
    print(f"  Final CSV            : {config.OUTPUT_DIR}/final_milestone_comparison.csv")
    print(f"  CloudSim exports     : {config.OUTPUT_DIR}/cloudsim/")


if __name__ == "__main__":
    main()
