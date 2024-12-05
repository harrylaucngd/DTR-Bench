import wandb
from collections.abc import Callable
from typing import Any, Optional
import subprocess
from tianshou.utils.logger.base import VALID_LOG_VALS_TYPE, BaseLogger
import pandas as pd
from tqdm import tqdm


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
        log_data["step"] = step
        wandb.log(log_data)

    def save_data(
        self,
        epoch: int,
        env_step: int,
        gradient_step: int,
        save_checkpoint_fn: Callable[[int, int, int], str] | None = None,
    ) -> None:
        # """Logs metadata to Weights & Biases and saves checkpoint."""
        # if save_checkpoint_fn and epoch - self.last_save_step >= self.save_interval:
        #     self.last_save_step = epoch
        #     checkpoint_path = save_checkpoint_fn(epoch, env_step, gradient_step)
        #     wandb.save(checkpoint_path)
        #     self.write("save/epoch", epoch, {"save/epoch": epoch})
        #     self.write("save/env_step", env_step, {"save/env_step": env_step})
        #     self.write("save/gradient_step", gradient_step, {"save/gradient_step": gradient_step})
        pass

    def restore_data(self) -> tuple[int, int, int]:
        """Restores metadata from Weights & Biases logs."""
        try:
            run = wandb.Api().run(f"{wandb.run.entity}/{wandb.run.project}/{wandb.run.id}")
            summary = run.summary
            epoch = summary.get("save/epoch", 0)
            env_step = summary.get("save/env_step", 0)
            gradient_step = summary.get("save/gradient_step", 0)
            self.last_save_step = epoch
            self.last_log_train_step = env_step
            self.last_log_update_step = gradient_step
            return epoch, env_step, gradient_step
        except Exception as e:
            print(f"Error restoring data: {e}")
            return 0, 0, 0

    def save_conda_env(self, file_name: str = "environment.yaml") -> None:
        """Save the current conda environment to a file and log it to Weights & Biases."""
        try:
            # Export the current conda environment to a YAML file
            subprocess.run(f"conda env export > {file_name}", shell=True, check=True)

            # Log the environment file as an artifact in Weights & Biases
            artifact = wandb.Artifact(name="conda-environment", type="environment")
            artifact.add_file(file_name)
            wandb.log_artifact(artifact)

            print(f"Conda environment saved and logged as artifact: {file_name}")

        except subprocess.CalledProcessError as e:
            print(f"Failed to export conda environment: {e}")
        except Exception as e:
            print(f"Error logging conda environment to wandb: {e}")


import pandas as pd
import numpy as np
from scipy.integrate import simps


def calculate_metric(df):
    # todo: chunk until the best test step
    best_step = df.loc[df["history_test/returns_stat/mean"].idxmax(), "history_step"]

    # Remove rows where 'history_step' is None
    df = df.dropna(subset=["history_step"])

    # Forward fill and backward fill missing values for 'history_train/returns_stat/mean'
    df["history_train/returns_stat/mean"] = df["history_train/returns_stat/mean"].fillna(method="ffill").fillna(method="bfill")

    # Group by 'history_step' and calculate the mean for duplicate 'history_step' values
    df = df.groupby("history_step", as_index=False)["history_train/returns_stat/mean"].mean()

    chunk_df = df[df["history_step"] <= best_step]
    chunk_df = chunk_df.sort_values(by="history_step")

    # Calculate the area under the curve (AUC) using Simpson's rule
    x = chunk_df["history_step"].values
    y = chunk_df["history_train/returns_stat/mean"].values
    auc = simps(y, x)

    return {
        "train_auc": auc,
        "best_test_step": best_step,
        "history_step": df["history_step"].values,
        "history_train/returns_stat/mean": df["history_train/returns_stat/mean"].values,
    }


def get_sweep(project_path, sweep_name):
    wandb.login()
    api = wandb.Api()
    sweeps = api.project(project_path, entity="gilesluo").sweeps()
    if len(sweeps) == 0:
        raise ValueError(f"No sweep found for project {project_path}")
    sweep_ids = []
    all_sweep_names = [sweep.config["name"] for sweep in sweeps]
    for sweep in sweeps:
        sweep_id = sweep.id
        if sweep.config["name"] != sweep_name:
            continue
        sweep_ids.append(sweep_id)

    if len(sweep_ids) == 0:
        raise ValueError(f"No sweep found for project {project_path} {sweep_name}, \n" f"all sweeps are {all_sweep_names}")
    elif len(sweep_ids) > 1:
        raise ValueError(f"Multiple sweeps found for {sweep_name}. Pls check")

    sweep_data = []
    runs = api.runs(project_path, {"sweep": sweep_ids[0]})
    for run in tqdm(runs, desc="Pulling runs from wandb"):
        # Extract the desired data from each experiment
        col_to_extract = [
            "step",
            "train/returns_stat/mean",
            "train/returns_stat/std",
            "test/returns_stat/mean",
            "test/returns_stat/std",
            "train/bg",
            "train/drug_mean",
        ]
        history = run.scan_history()

        data = {}
        for col in col_to_extract:
            data[f"history_{col}"] = [h[col] for h in history]
            if not data[f"history_{col}"]:
                raise ValueError(f"Empty data for {col}")
        # process the data
        data_df = pd.DataFrame(data)

        history_metrics = calculate_metric(data_df)

        sweep_data.append(
            dict(**{"id": run.id}, **history_metrics, **run.summary, **data, **run.config, **{"config_columns": list(run.config.keys())})
        )  # add experiment logdir to base obj
    # Create a DataFrame for the current sweep
    df = pd.DataFrame(sweep_data)
    return df


def summary_one_algo(task_name, algo_name, metrics, project="LLM4RL", ignore=None):
    if ignore is None:
        ignore = ["min", "max", "lens_stat", "collect_time", "n_collected_episodes", "collect_speed", "timestamp"]
    raw_data_df = get_sweep(project, f"{task_name}-{algo_name}")

    hyperparam_cols = [col for col in raw_data_df["config_columns"][0] if col not in ["logdir", "seed"]]
    raw_data_df["hyperparam"] = raw_data_df.apply(lambda x: "-".join([str(x[col]) for col in hyperparam_cols]), axis=1)

    final_test_cols = [col for col in raw_data_df.columns if "final_test/" in col]
    final_test_cols = [col for col in final_test_cols if not any([ig in col for ig in ignore])]
    data_df = raw_data_df[final_test_cols]
    data_df = pd.concat([data_df, raw_data_df[["hyperparam"]]], axis=1)

    data_mean, data_std = data_df.groupby("hyperparam").mean(), data_df.groupby("hyperparam").std()

    # process returns_stat differently
    return_prefix = [col for col in data_mean.columns if "returns_stat/mean" in col and "returns_stat/std" in col]
    for col in return_prefix:
        data_mean[col.replace("mean", "mean"), col.replace("mean", "std")] = data_mean[col], data_std[col]
        data_mean.drop(columns=col, inplace=True)
        data_std.drop(columns=col, inplace=True)

    assert data_mean.shape == data_std.shape
    assert len(data_mean) == len(raw_data_df) / raw_data_df["seed"].nunique()
    data_mean = data_mean.round(2)
    data_mean["policy_name"] = algo_name
    data_std = data_std.round(2)
    data_std["policy_name"] = algo_name

    # select best hyperparam for each metric, based on the best mean
    best_mean, best_std = [], []

    for metric in metrics:
        best_mean.append(data_mean.loc[data_mean[metric].idxmax(), metric])
        best_std.append(data_std.loc[data_mean[metric].idxmax(), metric])
    best_mean = pd.DataFrame([best_mean], columns=metrics, index=[algo_name])
    best_std = pd.DataFrame([best_std], columns=metrics, index=[algo_name])

    return best_mean, best_std


def summarize_one_task(task_name, algos, metrics, mode="all"):
    best_mean, best_std = [], []
    for algo in algos:
        mean, std = summary_one_algo(task_name, algo, metrics, mode)
        best_mean.append(mean)
        best_std.append(std)
    best_mean = pd.concat(best_mean)
    best_std = pd.concat(best_std)
    return best_mean, best_std


if __name__ == "__main__":
    best_mean, best_std = summary_one_algo(
        "SimGlucoseEnv-adult1",
        algo_name="DQN",
        project="LLM4RL-1127",
        metrics=["test/returns_stat/mean", "test/returns_stat/std"],
    )
