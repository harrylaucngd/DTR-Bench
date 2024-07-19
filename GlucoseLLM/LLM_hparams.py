import DTRBench.src.offpolicyRLHparams as offpolicyRLHparams
from DTRBench.src.offpolicyRLHparams import common_hparams

'''
Open LLM Leaderboard Top 3 Average Performance Model under 10B (2024.7.7):
1. internlm/internlm2_5-7b-chat
2. microsoft/Phi-3-small-128k-instruct
3. 01-ai/Yi-1.5-9b-Chat
Open LLM Leaderboard Top 1 Average Performance Model under 1B (2024.7.7):
1. Qwen/Qwen2-1.5B-Instruct
'''

token_dim_table = {
    "internlm2_5-7b-chat": {"token_dim": 4096},
    "Phi-3-small-128k-instruct": {"token_dim": 4096},
    "Yi-1.5-9b-Chat": {"token_dim": 4096},
    "Qwen2-1.5B-Instruct": {"token_dim": 1536},
    "llama-2-13b": {"token_dim": 5120},
    "llama-13b": {"token_dim": 5120},
    "llama-3-8b": {"token_dim": 4096},
    "llama-2-7b": {"token_dim": 4096},
    "llama-7b": {"token_dim": 4096},
    "gpt2": {"token_dim": 768}
}


class LLM_DQN_HyperParams(offpolicyRLHparams.DQNHyperParams):
    # "internlm2_5-7b-chat", "Phi-3-small-128k-instruct",
    # "Yi-1.5-9b-Chat", "Qwen2-1.5B-Instruct"
    # "llama-2-13b", "llama-13b",
    # "llama-3-8b", "llama-2-7b", "llama-7b",
    # "gpt2"
    _supported_algos = ("llm-dqn", "llm-ddqn")
    _policy_hparams = {
        "lr": common_hparams["lr"],  # learning rate
        # "stack_num": common_hparams["stack_num"],
        # "cat_num": common_hparams["cat_num"],
        "eps_test": common_hparams["eps_test"],
        "eps_train": common_hparams["eps_test"],
        "eps_train_final": 0.001,
        "n_step": common_hparams["n_step"],
        "target_update_freq": common_hparams["target_update_freq"],
        "is_double": False,
        "use_dueling": False,

        # llm hparam, TODO: The code here violates aesthetical requirements
        "llm": ["Qwen2-1.5B-Instruct", "internlm2_5-7b-chat"],
        "token_dim": 1536,

        # prompt hparam
        "need_obs_explain": [True, False],
        "need_act_explain": [True, False],
        "need_summary": [True, False],
        "exp_freq": [0, 12, 24, 36],
    }
