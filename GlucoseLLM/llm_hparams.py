import DTRBench.src.offpolicyRLHparams as offpolicyRLHparams
from DTRBench.src.offpolicyRLHparams import common_hparams

llm_dim_table = {
    "llama-2-13b": {"llm_dim": 5120},
    "llama-13b": {"llm_dim": 5120},
    "llama-3-8b": {"llm_dim": 4096},
    "llama-2-7b": {"llm_dim": 4096},
    "llama-7b": {"llm_dim": 4096},
    "gpt2": {"llm_dim": 768}
}

class LLM_DQN_HyperParams(offpolicyRLHparams.DQNHyperParams):
    # todo: add LLM prompt engineering hyperparameters
    # "llama-2-13b", "llama-13b",
    # "llama-3-8b", "llama-2-7b", "llama-7b",
    # "gpt2"
    _policy_hparams = {
        "lr": common_hparams["lr"],  # learning rate
        "stack_num": common_hparams["stack_num"],
        "cat_num": common_hparams["cat_num"],
        "eps_test": common_hparams["eps_test"],
        "eps_train": common_hparams["eps_test"],
        "eps_train_final": 0.005,
        "n_step": common_hparams["n_step"],
        "target_update_freq": common_hparams["target_update_freq"],
        "is_double": False,
        "use_dueling": False,
    }