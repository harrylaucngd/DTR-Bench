import argparse
import warnings
import wandb
from DTRBench.src.helper_fn import baselineLOOKUP
from DTRBench.src.base_obj import RLObjective
from DTRBench.utils.wandb_fn import WandbLogger
from DTRBench.utils.misc import set_global_seed
from DTRBench.naive_baselines.baselineHparams import BaselineHyperParams


class BaselineObj(RLObjective):
    def __init__(self, env_args: dict, hyperparam: BaselineHyperParams, **kwargs):
        super().__init__(None, env_args, hyperparam, device="cpu", **kwargs)

    def define_policy(self, policy_name, **kwargs):
        return baselineLOOKUP[policy_name]["policy"](action_space=self.env.action_space, **baselineLOOKUP[policy_name]["policy_args"])

    def wandb_search(self):
        self.logger = WandbLogger(train_interval=10, update_interval=100)
        self.env_name = wandb.config["env_name"]
        self.meta_param["training_num"] = 1
        self.meta_param["num_actions"] = None
        hparams = wandb.config

        self.prepare_env(int(hparams["seed"]), self.env_name, **self.env_args)
        set_global_seed(int(hparams["seed"]))

        # start training
        print("prepare policy")
        self.policy = self.define_policy(**{**hparams, **self.meta_param})

        # test on all envs
        self.test_all_patients(self.policy, None, int(hparams["seed"]), self.logger, n_episode=20)
