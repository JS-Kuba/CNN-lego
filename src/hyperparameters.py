import torch
from dataclasses import dataclass

@dataclass
class Hyperparameters:
    num_epochs: int = 10
    learning_rate: float = 0.001
    batch_size: int = 64

    wandb_config = {
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size
    }