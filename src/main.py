# torch imports
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn as nn
import torch.optim as optim

# external modules
from wandb_logging import WandbLogger
from dotenv import load_dotenv
import os
import argparse
import time

# internal modules
from model import CNN
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

    # Define data transforms
    transform = transforms.Compose([
        # transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_path = os.path.join('..' ,'dataset_prep', 'dataset_v3', 'train')
    valid_path = os.path.join('..', 'dataset_prep', 'dataset_v3', 'test')
    test_path = os.path.join('..', 'dataset_prep', 'dataset_v3', 'valid')

    # Load datasets
    train_data = datasets.ImageFolder(train_path, transform=transform)
    valid_data = datasets.ImageFolder(valid_path, transform=transform)
    test_data = datasets.ImageFolder(test_path, transform=transform)

    # Data loaders
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    # Initialize the model, loss function, and optimizer
    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
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
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, Validation Accuracy: {(correct/total)*100:.2f}%")

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