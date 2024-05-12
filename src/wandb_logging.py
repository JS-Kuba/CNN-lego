import wandb

class WandbLogger:
    def __init__(self, disable_logging) -> None:
        self.disable_logging = disable_logging

    def initialize(self, config, dataset, gpu):
        if not self.disable_logging:
            config["dataset"] = dataset
            config["gpu"] = gpu

            wandb.init(
                project="zpd-project",
                config=config
            )

    def log_loss(self, losses):
        if not self.disable_logging:
            wandb.log({"train_loss": losses['train'], "val_loss": losses['val']})

    def log_time(self, time):
        if not self.disable_logging:
            wandb.log({"train_time": round(time, 3)})

    def finish(self):
        if not self.disable_logging:
            wandb.finish()