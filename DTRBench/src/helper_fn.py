import torch
from tianshou.policy import BasePolicy
from GlucoseLLM.LLM_policy import LLM_DQN_Policy, LLM_PPO_Policy, LLM_Policy
from GlucoseLLM.LLM_hparams import LLM_DQN_HyperParams, LLM_PPO_HyperParams, LLMInference_HyperParams
from GlucoseLLM.LLMObj import LLM_DQN_Objective, LLM_PPO_Objective, LLMInferenceObjective
from DTRBench.src.onpolicyRLHparams import OnPolicyRLHyperParameterSpace, PPOHyperParams
from DTRBench.naive_baselines.naive_baselines import RandomPolicy, ConstantPolicy, PulsePolicy
from typing import Union

from tianshou.policy import (
    C51Policy,
    DQNPolicy,
    DDPGPolicy,
    TD3Policy,
    SACPolicy,
    REDQPolicy,
    DiscreteSACPolicy,
    DiscreteBCQPolicy,
    DiscreteCQLPolicy,
    BCQPolicy,
    CQLPolicy,
    ImitationPolicy,
    PPOPolicy,
)
from DTRBench.src.base_obj import RLObjective
from pathlib import Path
from DTRBench.src.RLObj import DQNObjective, TD3Objective, PPOObjective
from DTRBench.src.base_obj import OffPolicyRLHyperParameterSpace
from DTRBench.src.offpolicyRLHparams import DQNHyperParams, TD3HyperParams
from DTRBench.src.onpolicyRLHparams import PPOHyperParams


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


policyLOOKUP = {
    "ppo": {"hparam": PPOHyperParams, "policy": PPOPolicy, "obj": PPOObjective, "type": "continuous"},
    "dqn": {"hparam": DQNHyperParams, "policy": DQNPolicy, "obj": DQNObjective, "type": "discrete"},
    "td3": {"hparam": TD3HyperParams, "policy": TD3Policy, "obj": TD3Objective, "type": "continuous"},
    "llm-dqn": {"hparam": LLM_DQN_HyperParams, "policy": LLM_DQN_Policy, "obj": LLM_DQN_Objective, "type": "discrete"},
    "llm": {"hparam": LLMInference_HyperParams, "policy": LLM_Policy, "obj": LLMInferenceObjective, "type": "continuous"},
    "llm-ppo": {"hparam": LLM_PPO_HyperParams, "policy": LLM_PPO_Policy, "obj": LLM_PPO_Objective, "type": "continuous"},
}

baselineLOOKUP = {
    "zero_drug": {"policy": ConstantPolicy, "policy_args": {"dose": 0}},
    "constant0.02": {"policy": ConstantPolicy, "policy_args": {"dose": 0.02}},
    "random0.1": {"policy": RandomPolicy, "policy_args": {"min_act": 0, "max_act": 0.1}},
    "random0.5": {"policy": RandomPolicy, "policy_args": {"min_act": 0, "max_act": 0.5}},
    "pulse60-0.05": {"policy": PulsePolicy, "policy_args": {"dose": 0.05, "interval": 12}},
    "pulse60-0.1": {"policy": PulsePolicy, "policy_args": {"dose": 0.1, "interval": 12}},
    "pulse60-0.2": {"policy": PulsePolicy, "policy_args": {"dose": 0.2, "interval": 12}},
}


def get_policy_class(algo_name) -> BasePolicy:
    algo_name = algo_name.lower()
    if "llm" not in algo_name:
        if "dqn" in algo_name:
            algo_name = "dqn"
        elif "ddqn" in algo_name:
            algo_name = "ddqn"
        elif "discrete-imitation" in algo_name:
            algo_name = "discrete-imitation"
    return policyLOOKUP[algo_name]["policy"]


def get_hparam_class(algo_name: str, offline) -> Union[OffPolicyRLHyperParameterSpace.__class__, OnPolicyRLHyperParameterSpace.__class__]:
    algo_name = algo_name.lower()
    if offline:
        raise NotImplementedError("Offline RL is not supported yet")
    else:
        return policyLOOKUP[algo_name]["hparam"]


def get_obj_class(algo_name: str, offline) -> RLObjective.__class__:
    algo_name = algo_name.lower()
    if offline:
        raise NotImplementedError("Offline RL is not supported yet")
    else:
        return policyLOOKUP[algo_name]["obj"]


def get_policy_type(algo_name: str, offline: bool) -> str:
    algo_name = algo_name.lower()
    if offline:
        raise NotImplementedError("Offline RL is not supported yet")
    else:
        return policyLOOKUP[algo_name]["type"]
