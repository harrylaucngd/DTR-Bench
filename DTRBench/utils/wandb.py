import wandb
from collections.abc import Callable
from typing import Any, Optional

from tianshou.utils.logger.base import VALID_LOG_VALS_TYPE, BaseLogger


class WandbLogger(BaseLogger):
    """A logger that logs data to Weights & Biases.

    :param project_name: the name of the wandb project.
    :param run_name: the name of the wandb run. Default is None.
    :param train_interval: the log interval in log_train_data(). Default to 1000.
    :param test_interval: the log interval in log_test_data(). Default to 1.
    :param update_interval: the log interval in log_update_data(). Default to 1000.
    :param info_interval: the log interval in log_info_data(). Default to 1.
    :param save_interval: the save interval in save_data(). Default to 1 (save at
        the end of each epoch).
    """

    def __init__(
        self,
        project_name: Optional[str] = None,
        run_name: Optional[str] = None,
        train_interval: int = 1000,
        test_interval: int = 1,
        update_interval: int = 1000,
        info_interval: int = 1,
        save_interval: int = 1,
    ) -> None:
        super().__init__(train_interval, test_interval, update_interval, info_interval)
        self.save_interval = save_interval
        self.last_save_step = -1
        wandb.init(project=project_name, name=run_name)

    def write(self, step_type: str, step: int, data: dict[str, VALID_LOG_VALS_TYPE]) -> None:
        """Logs data to Weights & Biases."""
        log_data = data.copy()
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
        if save_checkpoint_fn and epoch - self.last_save_step >= self.save_interval:
            self.last_save_step = epoch
            checkpoint_path = save_checkpoint_fn(epoch, env_step, gradient_step)
            wandb.save(checkpoint_path)
            self.write("save/epoch", epoch, {"save/epoch": epoch})
            self.write("save/env_step", env_step, {"save/env_step": env_step})
            self.write("save/gradient_step", gradient_step, {"save/gradient_step": gradient_step})

    def restore_data(self) -> tuple[int, int, int]:
        """Restores metadata from Weights & Biases logs."""
        try:
            run = wandb.Api().run(f"{wandb.run.entity}/{wandb.run.project}/{wandb.run.id}")
            summary = run.summary
            epoch = summary.get('save/epoch', 0)
            env_step = summary.get('save/env_step', 0)
            gradient_step = summary.get('save/gradient_step', 0)
            self.last_save_step = epoch
            self.last_log_train_step = env_step
            self.last_log_update_step = gradient_step
            return epoch, env_step, gradient_step
        except Exception as e:
            print(f"Error restoring data: {e}")
            return 0, 0, 0
