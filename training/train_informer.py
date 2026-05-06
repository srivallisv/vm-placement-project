"""
train_informer.py
Train the Informer model on GPU with full 150-epoch schedule.
No early stopping.
"""

import os
import sys
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.informer import InformerModel
from preprocessing.dataset import create_dataloaders
from training.trainer import setup_device, train_model


def run_informer_training() -> dict:
    """Build Informer model, create data loaders, and train to completion."""
    device = setup_device()
    train_loader, val_loader, test_loader = create_dataloaders()

    model = InformerModel(
        n_features=config.NUM_FEATURES,
        seq_len=config.INPUT_WINDOW,
        pred_len=config.OUTPUT_HORIZON,
        label_len=config.INPUT_WINDOW // 2,
        d_model=256,
        n_heads=8,
        e_layers=3,
        d_layers=2,
        d_ff=512,
        dropout=0.1,
    )

    criterion = nn.SmoothL1Loss()

    history = train_model(
        model=model,
        model_name="informer",
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        learning_rate=config.LEARNING_RATES["informer"],
        device=device,
    )
    return history


if __name__ == "__main__":
    run_informer_training()
