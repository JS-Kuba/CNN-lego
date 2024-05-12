import torch
from dataclasses import dataclass

@dataclass
class Hyperparameters:
    batch_size: int = 128
    max_iters: int = 5000
    eval_interval: int = 100
    learning_rate: float = 2e-4
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_iters: int = 200
    n_layer: int = 12
    dropout: float = 0.2

    wandb_config = {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "max_iters": max_iters,
            "eval_interval": eval_interval,
            "eval_iters": eval_iters,
            "n_layer": n_layer,
            "device": device,
            "dropout": dropout
    }
