"""
train_patchtst.py
Train the PatchTST model on GPU with full 150-epoch schedule.
No early stopping.
"""

import os
import sys
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.patchtst import PatchTSTModel
from preprocessing.dataset import create_dataloaders
from training.trainer import setup_device, train_model


def run_patchtst_training() -> dict:
    """Build PatchTST model, create data loaders, and train to completion."""
    device = setup_device()
    train_loader, val_loader, test_loader = create_dataloaders()

    model = PatchTSTModel(
        n_features=config.NUM_FEATURES,
        seq_len=config.INPUT_WINDOW,
        pred_len=config.OUTPUT_HORIZON,
        patch_len=8,
        stride=4,
        d_model=128,
        n_heads=4,
        e_layers=3,
        d_ff=256,
        dropout=0.2,
    )

    criterion = nn.MSELoss()

    history = train_model(
        model=model,
        model_name="patchtst",
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        learning_rate=config.LEARNING_RATES["patchtst"],
        device=device,
    )
    return history


if __name__ == "__main__":
    run_patchtst_training()
