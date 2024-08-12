from DTRBench.src.offpolicyRLHparams import OffPolicyRLHyperParameterSpace
from DTRBench.src.base_hparams import common_hparams


class BaselineHyperParams(OffPolicyRLHyperParameterSpace):
    _meta_hparams = ["obs_window",
    ]

    # general hyperparameter search space
    _general_hparams = {
        # general parameters
        "seed": common_hparams["seed"],
        "policy_name": ["zero_drug", "constant0.02", "random0.1", "random0.5",
                        "pulse60-0.05", "pulse60-0.1", "pulse60-0.2"],
        "env_name": "SimGlucoseEnv-adult1" # final evaluation is the same so we only need one
    }
    # policy hyperparameter search space
    _policy_hparams = {
    }
    _supported_algos = ("zero_drug", "constant0.02", "random0.1", "random0.5", "pulse60-0.05", "pulse60-0.1")

    def __init__(self, obs_window):
        self.obs_window = obs_window