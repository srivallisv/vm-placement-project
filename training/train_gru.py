"""
train_gru.py
Train the Bidirectional GRU model on GPU with full 150-epoch schedule.
No early stopping.
"""

import os
import sys
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.gru import GRUModel
from preprocessing.dataset import create_dataloaders
from training.trainer import setup_device, train_model


def run_gru_training() -> dict:
    """Build GRU model, create data loaders, and train to completion."""
    device = setup_device()
    train_loader, val_loader, test_loader = create_dataloaders()

    model = GRUModel(
        n_features=config.NUM_FEATURES,
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
        output_horizon=config.OUTPUT_HORIZON,
    )

    criterion = nn.MSELoss()

    history = train_model(
        model=model,
        model_name="gru",
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        learning_rate=config.LEARNING_RATES["gru"],
        device=device,
    )
    return history


if __name__ == "__main__":
    run_gru_training()
