import os

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import wandb
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tianshou.data import Collector
from tianshou.env import DummyVectorEnv
from tianshou.policy.base import BasePolicy
from tianshou.utils import TensorboardLogger, WandbLogger
from torch.utils.tensorboard import SummaryWriter

from DTRGym import buffer_registry
from DTRGym.base import make_env
from DTRBench.utils.misc import set_global_seed
from DTRBench.src.offpolicyRLHparams import OffPolicyRLHyperParameterSpace
from DTRBench.utils.data import load_buffer


class RLObjective:
    def __init__(self, env_name, seed, training_num, test_num, num_actions, logdir, device, logger="wandb", **kwargs
                 ):

        # define high level parameters
        self.env_name, self.logger_type = env_name, logger
        self.logger = None
        self.device = device

        # define job name for logging
        self.job_name = self.env_name

        # early stopping counter
        self.rew_history = []

        # prepare env
        self.env, self.train_envs, self.test_envs = make_env(env_name, seed, training_num, 1,
                                                             num_actions=num_actions)

        state_shape = self.env.observation_space.shape or self.env.observation_space.n
        self.state_space = self.env.observation_space
        action_shape = self.env.action_space.shape or self.env.action_space.n
        self.action_space = self.env.action_space
        if isinstance(state_shape, (tuple, list)):
            if len(state_shape) > 1:
                raise NotImplementedError("state shape > 1 not supported yet")
            self.state_shape = state_shape[0]
        else:
            self.state_shape = int(state_shape)
        if isinstance(action_shape, (tuple, list)):
            if len(action_shape) > 1:
                raise NotImplementedError("action shape > 1 not supported yet")
            self.action_shape = action_shape
        else:
            self.action_shape = int(action_shape)

    def search_once(self, meta_params: dict, hparams: dict = None, metric="best_reward", log_name=None):
        # todo: online search should be problematic because offline wandb logger cannot be used with online trainer
        if self.logger == "wandb":
            self.logger = WandbLogger(project="SepsisRL",
                                      name=f"{self.env_name}-{meta_params['algo_name']}",
                                      save_interval=1,
                                      )
        else:
            raise NotImplementedError("Only wandb is supported for search_once")
        hparams = hparams if hparams is not None else wandb.config
        set_global_seed(hparams["seed"])

        # use all hparam combinations as the job name
        if log_name is None:
            trial_name = "-".join([f"{k}{v}" for k, v in hparams.items()])
            log_name = os.path.join(self.job_name, hparams["algo_name"],
                                    f"{trial_name}-seed{hparams['seed']}")
        else:
            print(f"log name {log_name} is provided, will not use hparams to generate log name")
        log_path = os.path.join(meta_params["logdir"], log_name)

        self.log_path = str(log_path)
        print(f"logging to {self.log_path}")
        wandb.log({"model_dir": log_path})
        os.makedirs(self.log_path, exist_ok=True)
        self.policy = self.define_policy(**hparams)

        result = self.run(self.policy, **hparams)
        score = result[metric.replace("test/", "")]

        self.logger.log_test_data(result, step=0)
        return score

    def early_stop_fn(self, mean_rewards):
        # todo: early stopping is not working for now, because stop_fn is called at each training step
        return False

    # no checkpoint saving needed
    def save_checkpoint_fn(self, epoch, env_step, gradient_step):
        return

    def define_policy(self, *args, **kwargs) -> BasePolicy:
        raise NotImplementedError

    def run(self, *args, **kwargs):
        raise NotImplementedError