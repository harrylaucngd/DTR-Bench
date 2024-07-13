import torch
from tianshou.policy import BasePolicy
from DTRBench.src.baseline_policy import RandomPolicy, MaxPolicy, MinPolicy

from tianshou.policy import C51Policy, DQNPolicy, DDPGPolicy, \
    TD3Policy, SACPolicy, REDQPolicy, DiscreteSACPolicy, DiscreteBCQPolicy, DiscreteCQLPolicy, BCQPolicy, CQLPolicy, \
    ImitationPolicy
from DTRBench.src.base_obj import RLObjective
from pathlib import Path
from DTRBench.src.RLObj import LLM_DQN_Objective, DQNObjective, TD3Objective
from DTRBench.src.base_obj import OffPolicyRLHyperParameterSpace
from DTRBench.src.offpolicyRLHparams import DQNHyperParams,  SACHyperParams, TD3HyperParams
import os
import shutil


def policy_load(policy, ckpt_path: str, device: str, is_train: bool = False):
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location=torch.device(device))
        ckpt = ckpt if ckpt_path.endswith("policy.pth") else ckpt["model"]  # policy.pth and ckpt.pth has different keys
        policy.load_state_dict(ckpt)
    if is_train:
        policy.train()
    else:
        policy.eval()
    return policy


offpolicyLOOKUP = {
    "dqn": {"hparam": DQNHyperParams, "policy": DQNPolicy, "obj": DQNObjective, "type": "discrete"},
    "td3": {"hparam": TD3HyperParams, "policy": TD3Policy, "obj": TD3Objective, "type": "continuous"},
}

BASELINE_LOOKUP = {"random": {"policy": RandomPolicy},
                   "max": {"policy": MaxPolicy},
                   "min": {"policy": MinPolicy}
                   }

# todo: add onpolicy
# onpolicyLOOKUP = {
#     "PPO": {"hparam": PPOHyperParams, "policy": PPOPolicy, "obj": PPOObjective, "type": "discrete"},
# }

def get_policy_class(algo_name) -> BasePolicy:
    algo_name = algo_name.lower()
    if "llm" not in algo_name:
        if "dqn" in algo_name:
            algo_name = "dqn"
        elif "ddqn" in algo_name:
            algo_name = "ddqn"
        elif "discrete-imitation" in algo_name:
            algo_name = "discrete-imitation"
    return offpolicyLOOKUP[algo_name]["policy"]


def get_hparam_class(algo_name: str, offline) -> OffPolicyRLHyperParameterSpace.__class__:
    algo_name = algo_name.lower()
    if "llm" not in algo_name:
        if "dqn" in algo_name:
            algo_name = "dqn"
        elif "ddqn" in algo_name:
            algo_name = "ddqn"
        elif "discrete-imitation" in algo_name:
            algo_name = "discrete-imitation"
    if offline:
        raise NotImplementedError("Offline RL is not supported yet")
    else:
        return offpolicyLOOKUP[algo_name]["hparam"]


def get_obj_class(algo_name: str, offline) -> RLObjective.__class__:
    algo_name = algo_name.lower()
    if "llm" not in algo_name:
        if "dqn" in algo_name:
            algo_name = "dqn"
        elif "ddqn" in algo_name:
            algo_name = "ddqn"
        elif "discrete-imitation" in algo_name:
            algo_name = "discrete-imitation"
    if offline:
        raise NotImplementedError("Offline RL is not supported yet")
    else:
        return offpolicyLOOKUP[algo_name]["obj"]


def get_policy_type(algo_name: str, offline: bool) -> str:
    algo_name = algo_name.lower()
    if "llm" not in algo_name:
        if "dqn" in algo_name:
            algo_name = "dqn"
        elif "ddqn" in algo_name:
            algo_name = "ddqn"
        elif "discrete-imitation" in algo_name:
            algo_name = "discrete-imitation"
    if offline:
        raise NotImplementedError("Offline RL is not supported yet")
    else:
        return offpolicyLOOKUP[algo_name]["type"]


def get_baseline_policy_class(algo_name: str) -> BasePolicy:
    return BASELINE_LOOKUP[algo_name]["policy"]
