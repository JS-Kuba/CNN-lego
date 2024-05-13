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
import numpy as np

# internal modules
from model import CNN
from hyperparameters import Hyperparameters as hp
from data import DataHandler
from class_mapping import ClassMapping

from sklearn.metrics import accuracy_score, f1_score, roc_curve, confusion_matrix, recall_score, precision_score
import matplotlib.pyplot as plt

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
        model.to(device)

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
            running_val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in valid_loader:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    running_val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            train_loss = running_loss/len(train_loader)
            val_loss = running_val_loss/len(valid_loader)
            val_acc = (correct/total)

            print(
                f"Epoch {epoch+1}/{hp.num_epochs}, \
                Train Loss: {train_loss:.4f}, \
                Val Loss: {val_loss:.4f}, \
                Validation Accuracy: {val_acc*100:.2f}%"
            )
            wandb_logger.log_loss(train_loss, val_loss)
            wandb_logger.log_val_accuracy(val_acc)

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

    # Calculate test accuracy
    test_acc = accuracy_score(test_true_labels, test_predictions)
    wandb_logger.log_test_accuracy(test_acc)
    print(f"TEST ACCURACY: {test_acc}")

    # Calculate F1 score for test set
    test_f1 = f1_score(test_true_labels, test_predictions, average='weighted')
    wandb_logger.log_f1_score(test_f1)
    print(f"TEST F1 SCORE: {test_f1}")

    # Calculate recall for test set
    test_recall = recall_score(test_true_labels, test_predictions, average='weighted')
    wandb_logger.log_recall(test_recall)
    print(f"TEST RECALL: {test_recall}")

    # Calculate precision for test set
    precision = precision_score(test_true_labels, test_predictions, average='weighted')
    wandb_logger.log_precision(precision)
    print(f"TEST PRECISION: {precision}")

    # Calculate confusion matrix for test set
    cm = confusion_matrix(test_true_labels, test_predictions)
    wandb_logger.log_confusion_matrix(test_true_labels, test_predictions)
    print(f"CONFUSION MATRIX:\n{cm}")

    # Calculate ROC curves using One vs. Rest (OvR) strategy
    num_classes = len(np.unique(test_true_labels))
    test_true_labels = np.array(test_true_labels)
    class_probs = np.array(class_probs)


    plt.figure(figsize=(8, 6))
    for class_idx in range(num_classes):
        class_true_labels = (test_true_labels == class_idx).astype(int)
        class_scores = class_probs[:, class_idx]

        fpr, tpr, _ = roc_curve(class_true_labels, class_scores)
        label = ClassMapping.get_label(class_idx)
        plt.plot(fpr, tpr, label=f"Class {label}")
        
        y_probas = np.column_stack((1 - class_scores, class_scores))
        wandb_logger.log_roc_curve(class_true_labels, y_probas, label)

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for each class')
    plt.legend()
    plt.grid(True)
    plt.show()

    wandb_logger.finish()

    if args.save:
        timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
        torch.save(model.state_dict(), f"final_models/model_{timestamp}.pth")
