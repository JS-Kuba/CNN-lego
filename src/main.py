# torch imports
import torch
import torch.nn as nn
import torch.optim as optim

# external modules
from wandb_logging import WandbLogger
from dotenv import load_dotenv
import argparse
import time
import os
import datetime

# internal modules
from model import CNN
from hyperparameters import Hyperparameters as hp
from data import DataHandler

import torch.nn.functional as F


if __name__ == "__main__":
    ### WANDB
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Argument parser (use --nolog to disable logging)."
    )
    parser.add_argument("--nolog", action="store_true", help="Disable logging")
    parser.add_argument("--save", action="store_true", help="Save model")
    args = parser.parse_args()
    disable_logging = args.nolog
    dataset = None
    wandb_logger = WandbLogger(disable_logging)
    wandb_logger.initialize(
        config=hp.wandb_config, dataset=dataset, gpu=os.environ.get("GPU")
    )
    tic = time.time()
    toc = time.time()
    wandb_logger.log_time(toc - tic)

    ### Get the data
    train_loader, valid_loader, test_loader = DataHandler.get_dataset()

    # Initialize the model, loss function, and optimizer
    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=hp.learning_rate)

    # Training loop
    for epoch in range(hp.num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)

            # logits = outputs.clone().detach()  # Clone and detach to prevent gradient computation
            # probs = F.softmax(logits, dim=1)
            # print(probs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(
            f"Epoch {epoch+1}/{hp.num_epochs}, Loss: {running_loss/len(train_loader):.4f}, Validation Accuracy: {(correct/total)*100:.2f}%"
        )

    # Test the model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {(correct/total)*100:.2f}%")

    if args.save:
        timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
        torch.save(model.state_dict(), f"final_models/model_{timestamp}.pth")