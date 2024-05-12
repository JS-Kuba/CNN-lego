import torch
from wandb_logging import WandbLogger
from dotenv import load_dotenv
import os
import argparse
import time
# from pynvml import *

from hyperparameters import Hyperparameters as hp


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description='Argument parser (use --nolog to disable logging).')
    parser.add_argument('--nolog', action='store_true', help='Disable logging')
    args = parser.parse_args()
    disable_logging = args.nolog

    dataset = None

    wandb_logger = WandbLogger(disable_logging)
    wandb_logger.initialize(config=hp.wandb_config, dataset=dataset, gpu = os.environ.get("GPU"))

    tic = time.time()
    ## training
    ## logging metrics

    toc = time.time()
    wandb_logger.log_time(toc-tic)