import DTRBench.src.offpolicyRLHparams as offpolicyRLHparams
from DTRBench.src.offpolicyRLHparams import common_hparams
from DTRBench.src.offpolicyRLHparams import OffPolicyRLHyperParameterSpace
import DTRBench.src.onpolicyRLHparams as onpolicyRLHparams

'''
Open LLM Leaderboard Top 3 Average Performance Model under 10B (2024.7.7):
1. internlm/internlm2_5-7b-chat
2. microsoft/Phi-3-small-128k-instruct
3. 01-ai/Yi-1.5-9B-Chat
Open LLM Leaderboard Top 1 Average Performance Model under 1B (2024.7.7):
1. Qwen/Qwen2-1.5B-Instruct
'''


class LLM_DQN_HyperParams(offpolicyRLHparams.DQNHyperParams):
    # "internlm2_5-7b-chat", "Phi-3-small-128k-instruct",
    # "Yi-1.5-9B-Chat", "Qwen2-1.5B-Instruct"
    # "llama-2-13b", "llama-13b",
    # "llama-3-8b", "llama-2-7b", "llama-7b",
    # "gpt2"
    _supported_algos = ("llm-dqn",)
    _general_hparams = {
        # general parameters
        "seed": common_hparams["seed"],
        "batch_size": 64,  #common_hparams["batch_size"],
        "step_per_collect": common_hparams["step_per_collect"],  # number of steps per collect. refer to tianshou's doc
        "update_per_step": common_hparams["update_per_step"],
        # number of frames to concatenate, cannot be used with stack_num or rnn, must be specified in the child class
        "gamma": common_hparams["gamma"],
        "start_timesteps": common_hparams["start_timesteps"],
    }
    _policy_hparams = {
        "lr": common_hparams["lr"],  # learning rate
        "eps_train": common_hparams["eps_train"],
        "eps_test": common_hparams["eps_test"],
        "eps_train_final": common_hparams["eps_train_final"],
        "n_step": common_hparams["n_step"],
        "target_update_freq": common_hparams["target_update_freq"],
        "is_double": False,
        "use_dueling": False,

        # llm hparam
        "llm_mode":
            {"llm": "Qwen2-0.5B-Instruct",
             "token_dim": 896},
        "llm_modal": "text_only",
        # {"llm": "Qwen2-1.5B-Instruct",
        #            "token_dim": 1536},

        # prompt hparam
        "summary_prob": 0.,
        "gradient_accumulation": 7,
    }


class LLM_PPO_HyperParams(onpolicyRLHparams.PPOHyperParams):
    _supported_algos = ("llm-ppo",)
    _policy_hparams = {"gae_lambda": 0.95,
                       "vf_coef": 0.5,
                       "ent_coef": 0.01,
                       "eps_clip": [0.1, 0.2],
                       "value_clip": False,
                       "dual_clip": None,
                       "advantage_normalization": True,
                       "recompute_advantage": False, }


class LLMInference_HyperParams(OffPolicyRLHyperParameterSpace):
    _supported_algos = ("llm",)
    _general_hparams = {
        # general parameters
        "seed": common_hparams["seed"],
    }
    # policy hyperparameter search space
    _policy_hparams = {
        "need_summary": [True, False],
        "num_try": 2,
        "llm_mode":
            [
            {"llm": "Qwen2-7B-Instruct",
             "context_window": 32768},
            {"llm": "internlm2_5-7b-chat",
             "context_window": 32768},
            {"llm": "Meta-Llama-3.1-8B-Instruct",
             "context_window": 131072},],
    }
