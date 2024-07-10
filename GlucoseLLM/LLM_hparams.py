import DTRBench.src.offpolicyRLHparams as offpolicyRLHparams
from DTRBench.src.offpolicyRLHparams import common_hparams

llm_dim_table = {
    "internlm2_5-7b-chat": {"llm_dim": 4096},
    "Phi-3-small-128k-instruct": {"llm_dim": 4096},
    "Yi-1.5-9b-Chat": {"llm_dim": 4096},
    "Qwen2-1.5B-Instruct": {"llm_dim": 1536},
}

class LLM_DQN_HyperParams(offpolicyRLHparams.DQNHyperParams):
    # todo: add LLM prompt engineering hyperparameters
    # "internlm2_5-7b-chat",
    # "Phi-3-small-128k-instruct",
    # "Yi-1.5-9b-Chat",
    # "Qwen2-1.5B-Instruct"
    _supported_algos = ("llm-dqn", "llm-ddqn")
    _policy_hparams = {
        "lr": common_hparams["lr"],  # learning rate
        "eps_test": common_hparams["eps_test"],
        "eps_train": common_hparams["eps_test"],
        "eps_train_final": 0.005,
        "n_step": common_hparams["n_step"],
        "target_update_freq": common_hparams["target_update_freq"],
        "is_double": False,
        "use_dueling": False,
    }