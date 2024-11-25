from DTRBench.src.offpolicyRLHparams import OffPolicyRLHyperParameterSpace
from DTRBench.src.base_hparams import common_hparams


class OnPolicyRLHyperParameterSpace(OffPolicyRLHyperParameterSpace):
    _meta_hparams = [
        "algo_name",  # name of the algorithm
        "log_dir",  # directory to save logs
        "training_num",  # number of training envs
        "test_num",  # number of test envs
        "epoch",
        "step_per_epoch",  # number of training steps per epoch
        "buffer_size",  # size of replay buffer
        "num_actions",  # number of actions, only used for discrete action space
        "linear",  # whether to use linear approximation as network
        "obs_window", # number of frames to concatenate or stack, depending on obs_mode
    ]

    # general hyperparameter search space
    _general_hparams = {
        # general parameters
        "seed": common_hparams["seed"],
        "batch_size": common_hparams["batch_size"],
        "step_per_collect": common_hparams["onpolicy_step_per_collect"],
        # number of steps per collect. refer to tianshou's doc
        "repeat_per_collect": common_hparams["repeat_per_collect"],
        # number of steps per collect. refer to tianshou's doc
        # number of frames to concatenate, cannot be used with stack_num or rnn, must be specified in the child class
        "gamma": common_hparams["gamma"],
        "obs_mode": common_hparams["obs_mode"],
    }
    # policy hyperparameter search space
    _policy_hparams = {
    }
    _supported_algos = ()


class PPOHyperParams(OnPolicyRLHyperParameterSpace):
    _supported_algos = ("ppo",)
    _policy_hparams = {"lr": common_hparams["lr"],  # learning rate
                       "n_step": common_hparams["n_step"],
                       "start_timesteps": common_hparams["start_timesteps"],
                       "gae_lambda": common_hparams["gamma"],
                       "conditioned_sigma": [True, False],
                       "vf_coef": 0.5,
                       "ent_coef": 0.001,
                       "eps_clip": 0.1,
                       "value_clip": False,
                       "dual_clip": None,
                       "advantage_normalization": True,
                       "recompute_advantage": False, }
