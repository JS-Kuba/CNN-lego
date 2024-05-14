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
from utils import Report


if __name__ == "__main__":
    ### WANDB
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Argument parser (use --nolog to disable logging)."
    )
    parser.add_argument("--nolog", action="store_true", help="Disable logging")
    parser.add_argument("--save", action="store_true", help="Save model")
    parser.add_argument('--load', type=str, help='Path to the model')

    args = parser.parse_args()
    disable_logging = args.nolog
    model_path = args.load
    dataset = None
    wandb_logger = WandbLogger(disable_logging)
    wandb_logger.initialize(
        config=hp.wandb_config, dataset=dataset, gpu=os.environ.get("GPU")
    )

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Found device: {device}")

    train_loader, valid_loader, test_loader = DataHandler.get_dataset()
    print(f"Train size: {len(train_loader)}")
    print(f"Valid size: {len(valid_loader)}")
    print(f"Test size: {len(test_loader)}")

    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=hp.learning_rate)

    if device == 'cuda:0':
        model = model.to(device)

    if not model_path:
        tic = time.time()
        for epoch in range(hp.num_epochs):
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                if device == "cuda:0":
                    inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            # Validation
            model.eval()

            running_val_loss = 0.0
            val_true = []
            val_pred = []
            val_class_probs = []

            with torch.no_grad():
                for inputs, labels in valid_loader:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    running_val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)

                    val_pred.extend(predicted.cpu().numpy())
                    val_true.extend(labels.cpu().numpy())
                    val_class_probs.extend(torch.nn.functional.softmax(outputs, dim=1).cpu().numpy())

            train_loss = running_loss/len(train_loader)
            val_loss = running_val_loss/len(valid_loader)

            print(
                f"Epoch {epoch+1}/{hp.num_epochs}, \
                Train Loss: {train_loss:.4f}, \
                Val Loss: {val_loss:.4f}"
            )
            wandb_logger.log_loss(train_loss, val_loss)
            Report.report_results(val_true, val_pred, "VAL", wandb_logger)
            Report.report_roc(val_true, val_class_probs, False, "VAL", wandb_logger)


        toc = time.time()
        wandb_logger.log_time(toc - tic)
    else:
        model = CNN()
        model.load_state_dict(torch.load(model_path))
        
    # Test the model
    model.eval()
    test_predictions = []
    test_true_labels = []
    class_probs = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            test_predictions.extend(predicted.cpu().numpy())
            test_true_labels.extend(labels.cpu().numpy())
            class_probs.extend(torch.nn.functional.softmax(outputs, dim=1).cpu().numpy())

    Report.report_results(test_true_labels, test_predictions, "TEST", wandb_logger)
    Report.report_roc(test_true_labels, class_probs, True, "TEST", wandb_logger)

    wandb_logger.finish()

    if args.save:
        timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
        torch.save(model.state_dict(), f"final_models/model_{timestamp}.pth")
