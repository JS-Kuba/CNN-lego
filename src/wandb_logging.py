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

    def log_val_accuracy(self, accuracy):
        if not self.disable_logging:
            wandb.log({"val_accuracy": accuracy})
    
    def log_test_accuracy(self, accuracy):
        if not self.disable_logging:
            wandb.log({"test_accuracy": accuracy})


    def log_f1_score(self, f1_score):
        if not self.disable_logging:
            wandb.log({"f1_score": f1_score})

    def log_recall(self, recall):
        if not self.disable_logging:
            wandb.log({"recall": recall})

    def log_precision(self, precision):
        if not self.disable_logging:
            wandb.log({"precision": precision})

    def log_roc_curve(self, class_true_labels, class_scores, class_label):
        if not self.disable_logging:
            wandb.log({f"ROC_curve_{class_label}": wandb.plot.roc_curve(
                class_true_labels, class_scores, ["Other", class_label]
                )
            })

    def log_confusion_matrix(self, test_true_labels, test_predictions):
        if not self.disable_logging:
            wandb.log({"Test Confusion Matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=test_true_labels,
                preds=test_predictions,
                class_names=["Flower", "Leaf", "Stone", "Wood"]
            )})

    def finish(self):
        if not self.disable_logging:
            wandb.finish()