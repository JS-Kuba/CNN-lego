import torch
from dataclasses import dataclass

@dataclass
class Hyperparameters:
    num_epochs: int = 10
    learning_rate: float = 0.001

    wandb_config = {
            "num_epochs": num_epochs,
    }
