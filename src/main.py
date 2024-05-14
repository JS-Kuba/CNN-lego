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

def report_results(y_true, y_pred, tag):
    # Calculate test accuracy
    test_acc = accuracy_score(y_true, y_pred)
    wandb_logger.log_accuracy(test_acc, tag)
    # print(f"{tag} ACCURACY: {test_acc}")

    # Calculate F1 score for test set
    test_f1 = f1_score(y_true, y_pred, average='weighted')
    wandb_logger.log_f1_score(test_f1, tag)
    # print(f"{tag} F1 SCORE: {test_f1}")

    # Calculate recall for test set
    test_recall = recall_score(y_true, y_pred, average='weighted')
    wandb_logger.log_recall(test_recall, tag)
    # print(f"{tag} RECALL: {test_recall}")

    # Calculate precision for test set
    precision = precision_score(y_true, y_pred, average='weighted')
    wandb_logger.log_precision(precision, tag)
    # print(f"{tag} PRECISION: {precision}")

    # Calculate confusion matrix for test set
    cm = confusion_matrix(y_true, y_pred)
    wandb_logger.log_confusion_matrix(y_true, y_pred, tag)
    # print(f"{tag} CONFUSION MATRIX:\n{cm}")


    print(
        f"{tag} Acc: {test_acc:.4f}, \
        {tag} F1: {test_f1:.4f}, \
        {tag} Recall: {test_recall:.4f}, \
        {tag} Precision: {precision:.4f}, \
        {tag} CM:\n{cm}"
    )

def report_roc(true_labels, class_probs, matplot: bool, tag):
    num_classes = len(np.unique(true_labels))
    true_labels = np.array(true_labels)
    class_probs = np.array(class_probs)

    if matplot: plt.figure(figsize=(8, 6))
    for class_idx in range(num_classes):
        class_true_labels = (true_labels == class_idx).astype(int)
        class_scores = class_probs[:, class_idx]
        label = ClassMapping.get_label(class_idx)
        y_probas = np.column_stack((1 - class_scores, class_scores))

        wandb_logger.log_roc_curve(class_true_labels, y_probas, label, tag)
        
        fpr, tpr, _ = roc_curve(class_true_labels, class_scores)
        if matplot: plt.plot(fpr, tpr, label=f"Class {label}")
    if matplot: 
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for each class')
        plt.legend()
        plt.grid(True)
        plt.show()

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
    print(f"Found device: {device} / {torch.cuda.get_device_name(torch.cuda.current_device())}")

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

                # logits = outputs.clone().detach()  # Clone and detach to prevent gradient computation
                # probs = F.softmax(logits, dim=1)
                # print(probs)
                # print(len(inputs))
                # print(len(labels))
                # print(len(outputs))

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
                    # inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    running_val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)

                    val_pred.extend(predicted.cpu().numpy())
                    val_true.extend(labels.cpu().numpy())
                    val_class_probs.extend(torch.nn.functional.softmax(outputs, dim=1).cpu().numpy())

            train_loss = running_loss/len(train_loader)
            val_loss = running_val_loss/len(valid_loader)
            # val_acc = (correct/total)

            print(
                f"Epoch {epoch+1}/{hp.num_epochs}, \
                Train Loss: {train_loss:.4f}, \
                Val Loss: {val_loss:.4f}"
            )
            wandb_logger.log_loss(train_loss, val_loss)
            report_results(val_true, val_pred, "VAL")
            report_roc(val_true, val_class_probs, False, "VAL")


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
            # inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            test_predictions.extend(predicted.cpu().numpy())
            test_true_labels.extend(labels.cpu().numpy())
            class_probs.extend(torch.nn.functional.softmax(outputs, dim=1).cpu().numpy())

    report_results(test_true_labels, test_predictions, "TEST")
    report_roc(test_true_labels, class_probs, True, "TEST")

    wandb_logger.finish()

    if args.save:
        timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
        torch.save(model.state_dict(), f"final_models/model_{timestamp}.pth")
