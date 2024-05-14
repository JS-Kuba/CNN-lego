
from sklearn.metrics import accuracy_score, f1_score, roc_curve, confusion_matrix, recall_score, precision_score
from wandb_logging import WandbLogger
from class_mapping import ClassMapping
import matplotlib.pyplot as plt
import numpy as np

class Report:
    @staticmethod
    def report_results(y_true, y_pred, tag, wandb_logger):
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

    @staticmethod
    def report_roc(true_labels, class_probs, matplot: bool, tag, wandb_logger):
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