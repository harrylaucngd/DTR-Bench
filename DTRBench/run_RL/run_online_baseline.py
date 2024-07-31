#!/usr/bin/env python
import argparse
import wandb
from pathlib import Path
from DTRBench.src.helper_fn import get_policy_class, get_hparam_class, get_obj_class, get_policy_type
import warnings
from DTRBench.src.naive_baselines import BaselineHyperParams
import os
from tqdm import tqdm
import wandb
from DTRBench.utils.wandb import WandbLogger
from torch.utils.tensorboard import SummaryWriter
from tianshou.trainer.utils import test_episode
from DTRGym.base import make_env
from DTRBench.utils.misc import set_global_seed
from DTRBench.src.offpolicyRLHparams import OffPolicyRLHyperParameterSpace
from DTRBench.src.helper_fn import baselineLOOKUP
from DTRBench.src.naive_baselines import BaselineHyperParams
from DTRBench.src.base_obj import RLObjective

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
wandb.require("core")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_project_name", type=str, default="LLM4RL2")
    parser.add_argument("--sweep_id", type=str, default="3mis2z1p", help="sweep id for wandb,"
                                                                         " only used in agent mode")
    parser.add_argument("--role", type=str, default="sweep", choices=["sweep", "agent", "run_single"])
    args = parser.parse_known_args()[0]
    return args


def call_agent():
    try:
        obj = BaselineObj({"discrete": False}, hparam)
        obj.wandb_search()
    except TimeoutError as e:
        # Update the status to 'crashed' due to timeout
        wandb.run.summary["status"] = "crashed"
        wandb.run.summary["failure_reason"] = str(e)
        wandb.run.finish()
        raise e
    else:
        # Finish the wandb experiment normally if no issues
        wandb.finish()
    return


class BaselineObj(RLObjective):
    def __init__(self, env_args: dict, hyperparam: BaselineHyperParams, **kwargs):
        super().__init__(None, env_args, hyperparam, device="cpu", **kwargs)

    def define_policy(self, policy_name, **kwargs):
        return baselineLOOKUP[policy_name]["policy"](
            action_space=self.env.action_space, **baselineLOOKUP[policy_name]["policy_args"]
        )

    def wandb_search(self):
        self.logger = WandbLogger(train_interval=10)
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



if __name__ == "__main__":
    args = parse_args()
    hparam = BaselineHyperParams()
    if args.role == "sweep":
        sweep_configuration = {
            "method": "grid",
            "project": args.wandb_project_name,
            "name": "all-naive-baselines",
            "metric": {"goal": "maximize", "name": "test/returns_stat/mean"},
            "parameters": hparam.get_search_space(),
        }
        sweep_id = wandb.sweep(sweep_configuration, project=args.wandb_project_name)
        wandb.agent(sweep_id=sweep_id, function=call_agent, project=args.wandb_project_name)
    elif args.role == "agent":
        wandb.agent(sweep_id=args.sweep_id, function=call_agent, project=args.wandb_project_name)
    else:
        print("role must be one of [sweep, agent, run_single], get {}".format(args.role))
        raise NotImplementedError
