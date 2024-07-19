import torch
from tianshou.policy import BasePolicy
from tianshou.policy import DQNPolicy, TD3Policy
from GlucoseLLM.LLM_policy import LLM_DQN_Policy, LLM_Policy
from DTRBench.src.base_obj import RLObjective

from GlucoseLLM.LLM_hparams import LLM_DQN_HyperParams, LLM_HyperParams
from GlucoseLLM.LLMObj import LLM_DQN_Objective, LLM_Objective
from DTRBench.src.RLObj import DQNObjective, TD3Objective
from DTRBench.src.base_obj import OffPolicyRLHyperParameterSpace
from DTRBench.src.offpolicyRLHparams import DQNHyperParams, TD3HyperParams
from DTRBench.naive_baselines.naive_baselines import RandomPolicy, ConstantPolicy, PulsePolicy



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
    # "ddqn": {"hparam": DQNHyperParams, "policy": DQNPolicy, "obj": DQNObjective, "type": "discrete"},
    # "sac": {"hparam": SACHyperParams, "policy": SACPolicy, "obj": SACObjective, "type": "continuous"},
    "td3": {"hparam": TD3HyperParams, "policy": TD3Policy, "obj": TD3Objective, "type": "continuous"},
    "llm-dqn": {"hparam": LLM_DQN_HyperParams, "policy": LLM_DQN_Policy, "obj": LLM_DQN_Objective, "type": "discrete"},
    # "llm-ddqn": {"hparam": LLM_DQN_HyperParams, "policy": LLM_DQN_Policy, "obj": LLM_DQN_Objective, "type": "discrete"},
    "llm": {"hparam": LLM_HyperParams, "policy": LLM_Policy, "obj": LLM_Objective, "type": "continuous"},
}

baselineLOOKUP = {"zero_drug": {"policy": ConstantPolicy, "policy_args": {"dose": 0}},
                  "constant0.02": {"policy": ConstantPolicy, "policy_args": {"dose": 0.02}},
                  "random0.1": {"policy": RandomPolicy, "policy_args": {"min_act": 0, "max_act": 0.1}},
                  "random0.5": {"policy": RandomPolicy, "policy_args": {"min_act": 0, "max_act": 0.5}},
                  "pulse30-0.1": {"policy": PulsePolicy, "policy_args": {"dose": 0.1, "interval": 6}},
                  "pulse60-0.2": {"policy": PulsePolicy, "policy_args": {"dose": 0.2, "interval": 12}}}


# todo: add onpolicy
# onpolicyLOOKUP = {
#     "PPO": {"hparam": PPOHyperParams, "policy": PPOPolicy, "obj": PPOObjective, "type": "discrete"},
# }

def get_policy_class(algo_name) -> BasePolicy:
    algo_name = algo_name.lower()
    return offpolicyLOOKUP[algo_name]["policy"]


def get_hparam_class(algo_name: str, offline) -> OffPolicyRLHyperParameterSpace.__class__:
    algo_name = algo_name.lower()
    if offline:
        raise NotImplementedError("Offline RL is not supported yet")
    else:
        return offpolicyLOOKUP[algo_name]["hparam"]


def get_obj_class(algo_name: str, offline) -> RLObjective.__class__:
    algo_name = algo_name.lower()
    if offline:
        raise NotImplementedError("Offline RL is not supported yet")
    else:
        return offpolicyLOOKUP[algo_name]["obj"]


def get_policy_type(algo_name: str, offline: bool) -> str:
    algo_name = algo_name.lower()
    if offline:
        raise NotImplementedError("Offline RL is not supported yet")
    else:
        return offpolicyLOOKUP[algo_name]["type"]


def get_baseline_policy_class(algo_name: str) -> BasePolicy:
    return baselineLOOKUP[algo_name]["policy"]
