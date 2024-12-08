import DTRBench.src.offpolicyRLHparams as offpolicyRLHparams
from DTRBench.src.offpolicyRLHparams import common_hparams
from DTRBench.src.offpolicyRLHparams import OffPolicyRLHyperParameterSpace
import DTRBench.src.onpolicyRLHparams as onpolicyRLHparams

"""
Open LLM Leaderboard Top 3 Average Performance Model under 10B (2024.7.7):
1. internlm/internlm2_5-7b-chat
2. microsoft/Phi-3-small-128k-instruct
3. 01-ai/Yi-1.5-9B-Chat
Open LLM Leaderboard Top 1 Average Performance Model under 1B (2024.7.7):
1. Qwen/Qwen2-1.5B-Instruct
"""


class LLM_DQN_HyperParams(offpolicyRLHparams.DQNHyperParams):
    # "internlm2_5-7b-chat", "Phi-3-small-128k-instruct",
    # "Yi-1.5-9B-Chat", "Qwen2-1.5B-Instruct"
    # "llama-2-13b", "llama-13b",
    # "llama-3-8b", "llama-2-7b", "llama-7b",
    # "gpt2"
    _supported_algos = ("llm-dqn", "llm-ddqn")
    _general_hparams = {
        # general parameters
        "seed": common_hparams["llm_seed"],
        "batch_size": 2,
        "obs_mode": common_hparams["obs_mode"],
        "step_per_collect": common_hparams["step_per_collect"],  # number of steps per collect. refer to tianshou's doc
        "update_per_step": common_hparams["update_per_step"],
        # number of frames to concatenate, cannot be used with stack_num or rnn, must be specified in the child class
        "gamma": common_hparams["gamma"],
        "start_timesteps": common_hparams["start_timesteps"],
    }
    _policy_hparams = {
        "lr": common_hparams["lr"],  # learning rate
        "eps_test": common_hparams["eps_test"],
        "eps_train": common_hparams["eps_train"],
        "eps_train_final": common_hparams["eps_train_final"],
        "n_step": common_hparams["n_step"],
        "target_update_freq": common_hparams["target_update_freq"],
        "is_double": False,
        "use_dueling": False,
        # llm hparam
        "llm_mode": [
            {"llm": "Qwen2.5-0.5B-Instruct", "token_dim": 896},
            {"llm": "Qwen2.5-1.5B-Instruct", "token_dim": 1536},
        ],
        # prompt hparam
        "sum_prob": 1.0,
    }


class LLM_PPO_HyperParams(onpolicyRLHparams.PPOHyperParams):
    _supported_algos = ("llm-ppo",)
    _general_hparams = {
        # general parameters
        "seed": common_hparams["llm_seed"],
        "batch_size": 64,
        "step_per_collect": common_hparams["onpolicy_step_per_collect"],
        # number of steps per collect. refer to tianshou's doc
        "repeat_per_collect": common_hparams["repeat_per_collect"],
        # number of steps per collect. refer to tianshou's doc
        # number of frames to concatenate, cannot be used with stack_num or rnn, must be specified in the child class
        "gamma": common_hparams["gamma"],
    }
    _policy_hparams = {
        "lr": common_hparams["lr"],  # learning rate
        "n_step": common_hparams["n_step"],
        "start_timesteps": common_hparams["start_timesteps"],
        "gae_lambda": 0.95,
        "vf_coef": 0.5,
        "ent_coef": 0.001,
        "eps_clip": 0.1,
        "value_clip": False,
        "dual_clip": None,
        "advantage_normalization": True,
        "recompute_advantage": False,
        # llm hparam
        "llm_mode": [
            {"llm": "Qwen2-0.5B-Instruct", "token_dim": 896},
            {"llm": "Qwen2-1.5B-Instruct", "token_dim": 1536},
        ],
        # prompt hparam
        "sum_prob": [0, 0.1, 0.2, 0.4],
    }


class LLMInference_HyperParams(OffPolicyRLHyperParameterSpace):
    _supported_algos = ("llm",)
    _general_hparams = {
        # general parameters
        "seed": common_hparams["seed"],
    }
    # policy hyperparameter search space
    _policy_hparams = {
        "llm_mode": [
            {"llm": "Qwen2.5-1.5B-Instruct", "context_window": 32768},
            # todo: llama3 7b
            # qwen 7b
            # {"llm": "internlm2_5-7b-chat", "context_window": 32768},
            # {"llm": "Phi-3-small-128k-instruct", "context_window": 131072},
            # {"llm": "Yi-1.5-9B-Chat", "context_window": 4096},
        ],
    }
