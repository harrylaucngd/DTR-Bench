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
    def __init__(self, env_name, hyperparam: OffPolicyRLHyperParameterSpace, device, **kwargs):
        # define high level parameters
        self.env_name = env_name
        self.hyperparam = hyperparam
        self.meta_param = self.hyperparam.get_meta_params()
        self.device = device

    def get_search_space(self):
        return self.hyperparam.get_search_space()

    def prepare_env(self, seed):
        # prepare env
        self.env, self.train_envs, self.test_envs = make_env(self.env_name, int(seed),
                                                             self.meta_param["training_num"], 1,
                                                             num_actions=self.meta_param["num_actions"])
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

    def search_once(self, hparams: dict, metric="best_reward"):
        self.prepare_env(int(hparams["seed"]))
        set_global_seed(int(hparams["seed"]))

        # get names
        hp_name = "-".join([f"{v}" if not isinstance(v, dict) else f"{list(v.keys())[0]}"
                            for k, v in hparams.items() if k not in self.meta_param.keys() or k != "wandb_project_name"])
        self.log_path = os.path.join(self.meta_param["log_dir"], f"{hp_name}")

        print(f"logging to {self.log_path}")
        os.makedirs(self.log_path, exist_ok=True)
        writer = SummaryWriter(log_dir=self.log_path)
        self.logger = WandbLogger(project=hparams["wandb_project_name"], config=hparams, train_interval=24*15)
        self.logger.load(writer)

        # start training
        self.policy = self.define_policy(**{**hparams, **self.meta_param})
        result = self.run(self.policy, **{**hparams, **self.meta_param})
        score = result[metric.replace("test/", "")]

        self.logger.log_test_data(result, step=0)

        # todo:load best policy to test on other envs
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
