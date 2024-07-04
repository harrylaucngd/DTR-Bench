import DTRBench.src.offpolicyRLHparams as offpolicyRLHparams
from DTRBench.src.offpolicyRLHparams import common_hparams

class LLM_DQN_HyperParams(offpolicyRLHparams.DQNHyperParams):
    # todo: add LLM prompt engineering hyperparameters
    _policy_hparams = {
        "lr": common_hparams["lr"],  # learning rate
        "stack_num": 1,
        "cat_num": 1,
        "eps_test": 0.005,
        "eps_train": 1,
        "eps_train_final": 0.005,
        "n_step": common_hparams["n_step"],
        "target_update_freq": common_hparams["target_update_freq"],
        "is_double": False,
        "use_dueling": False,
    }