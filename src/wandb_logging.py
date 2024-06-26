import wandb
import numpy as np
from class_mapping import ClassMapping

class WandbLogger:
    def __init__(self, disable_logging) -> None:
        self.disable_logging = disable_logging

    def initialize(self, config, dataset, gpu):
        if not self.disable_logging:
            config["dataset"] = dataset
            config["gpu"] = gpu

            wandb.init(
                project="cnn-lego",
                config=config
            )

    def log_loss(self, train_loss, val_loss):
        if not self.disable_logging:
            wandb.log({"train_loss": train_loss, "val_loss": val_loss})

    def log_time(self, time):
        if not self.disable_logging:
            wandb.log({"train_time": round(time, 3)})

    def log_accuracy(self, accuracy, tag):
        if not self.disable_logging:
            wandb.log({f"{tag} accuracy": accuracy})

    def log_f1_score(self, f1_score, tag):
        if not self.disable_logging:
            wandb.log({f"{tag} f1_score": f1_score})

    def log_recall(self, recall, tag):
        if not self.disable_logging:
            wandb.log({f"{tag} recall": recall})

    def log_precision(self, precision, tag):
        if not self.disable_logging:
            wandb.log({f"{tag} precision": precision})

    def log_roc_curve(self, class_true_labels, class_scores, class_label, tag):
        if not self.disable_logging:
            wandb.log({f"ROC_curve_{class_label}": wandb.plot.roc_curve(
                y_true=class_true_labels, 
                y_probas=class_scores, 
                labels=["Other", class_label],
                title=f"{tag} ROC - {class_label} vs others"
                )
            })

    def log_confusion_matrix(self, test_true_labels, test_predictions, tag):
        if not self.disable_logging:
            wandb.log({f"{tag} Confusion Matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=test_true_labels,
                preds=test_predictions,
                class_names=["Flower", "Leaf", "Stone", "Wood"],
                title=f"{tag} Confusion Matrix"
            )})

    def finish(self):
        if not self.disable_logging:
            wandb.finish()