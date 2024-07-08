import wandb
from typing import Optional, Callable
from tianshou.utils import BaseLogger

class WandbLogger(BaseLogger):
    """A logger that logs data to Weights & Biases."""

    def __init__(
            self,
            project_name: str,
            run_name: Optional[str] = None,
            train_interval: int = 1000,
            test_interval: int = 1,
            update_interval: int = 1000,
            info_interval: int = 1,
    ) -> None:
        super().__init__(train_interval, test_interval, update_interval, info_interval)
        self.project_name = project_name
        self.run_name = run_name
        wandb.init(project=self.project_name, name=self.run_name)

    def write(self, step_type: str, step: int, data) -> None:
        """Logs data to Weights & Biases."""
        log_data = {f"{step_type}/{k}": v for k, v in data.items()}
        log_data['step'] = step
        wandb.log(log_data)

    def save_data(
            self,
            epoch: int,
            env_step: int,
            gradient_step: int,
            save_checkpoint_fn: Callable[[int, int, int], str] | None = None,
    ) -> None:
        """Logs metadata to Weights & Biases and saves checkpoint."""
        metadata = {
            "epoch": epoch,
            "env_step": env_step,
            "gradient_step": gradient_step,
        }
        wandb.log(metadata)

        if save_checkpoint_fn:
            checkpoint_path = save_checkpoint_fn(epoch, env_step, gradient_step)
            wandb.save(checkpoint_path)

    def restore_data(self) -> tuple[int, int, int]:
        """Restores metadata from Weights & Biases logs."""
        try:
            run = wandb.Api().run(f"{wandb.run.entity}/{self.project_name}/{wandb.run.id}")
            summary = run.summary
            epoch = summary.get('epoch', 0)
            env_step = summary.get('env_step', 0)
            gradient_step = summary.get('gradient_step', 0)
            return epoch, env_step, gradient_step
        except Exception as e:
            print(f"Error restoring data: {e}")
            return 0, 0, 0

# Usage example:
# logger = WandbLogger(project_name="my_project", run_name="experiment_1")
# logger.log_train_data({"loss": 0.1, "accuracy": 0.9}, step=1000)
