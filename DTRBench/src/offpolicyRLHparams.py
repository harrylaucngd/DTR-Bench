
from DTRBench.src.base_hparams import common_hparams
import numpy as np


class OffPolicyRLHyperParameterSpace:
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
    ]

    # general hyperparameter search space
    _general_hparams = {
        # general parameters
        "seed": common_hparams["seed"],
        "batch_size": common_hparams["batch_size"],
        "step_per_collect": common_hparams["step_per_collect"],  # number of steps per collect. refer to tianshou's doc
        "update_per_step": common_hparams["update_per_step"],
        # number of frames to concatenate, cannot be used with stack_num or rnn, must be specified in the child class
        "gamma": common_hparams["gamma"],
        "obs_mode": common_hparams["obs_mode"],
        "start_timesteps": common_hparams["start_timesteps"],
    }
    # policy hyperparameter search space
    _policy_hparams = {
    }
    _supported_algos = ()

    def __init__(self,
                 algo_name,  # name of the algorithm
                 log_dir,  # directory to save logs
                 training_num,  # number of training envs
                 test_num,  # number of test envs
                 epoch,
                 step_per_epoch,  # number of training steps per epoch
                 buffer_size,  # size of replay buffer
                 num_actions=None,  # number of actions, only used for discrete action space
                 linear=False
                 ):
        if algo_name.lower() not in [i.lower() for i in self.__class__._supported_algos]:
            raise NotImplementedError(f"algo_name {algo_name} not supported, support {self.__class__._supported_algos}")
        self.algo_name = algo_name
        self.log_dir = log_dir
        self.training_num = training_num
        self.test_num = test_num
        self.epoch = epoch
        self.step_per_epoch = step_per_epoch
        self.buffer_size = buffer_size
        self.num_actions = num_actions
        self.linear = linear

    def check_illegal(self):
        """
        This function makes sure all hyperparameters are defined.
        all hyperparameters should be defined in _meta_hparams, _general_hparams and _policy_hparams. If not, raise error
        and list the undefined hyperparameters.
        :return: list of undefined hyperparameters
        """
        all_hparams = list(self._meta_hparams) + list(self._general_hparams.keys()) + list(self._policy_hparams.keys())
        undefined_hparams = [h for h in all_hparams if not hasattr(self, h)]
        unknown_hparams = [h for h in self.__dict__() if h not in all_hparams]
        if len(undefined_hparams) > 0:
            printout1 = f"undefined hyperparameters: {undefined_hparams}"
        else:
            printout1 = ""
        if len(unknown_hparams) > 0:
            printout2 = f"unknown hyperparameters: {unknown_hparams}"
        else:
            printout2 = ""
        if len(printout1) > 0 or len(printout2) > 0:
            raise ValueError(f"{printout1}\n{printout2}")

    def get_search_space(self):
        search_space = {}
        search_space.update(self._general_hparams)
        search_space.update(self._policy_hparams)
        space = {}
        for k, v in search_space.items():
            if isinstance(v, (int, float, bool, str, dict, list, tuple)):
                if not hasattr(v, "__len__") or len(v) == 1:
                    space[k] = {"value": v}
                else:
                    space[k] = {"values": v}
            else:
                raise NotImplementedError(f"unsupported type {type(v)} for hyperparameter {k}")
        return space

    def sample(self, mode="first"):
        if mode == "first":
            sample_fn = lambda x: x[0]
        else:
            sample_fn = lambda x: np.random.choice(x)
        search_space = self.get_search_space()
        result = {}
        for k, v in search_space.items():
            if "values" in v:
                result[k] = sample_fn(v["values"])
            elif "value" in v:
                result[k] = v["value"]
            else:
                raise NotImplementedError
        return result

    def get_meta_params(self):
        return {k: getattr(self, k) for k in self._meta_hparams}

    def get_general_params(self):
        return {k: getattr(self, k) for k in self._general_hparams.keys()}

    def get_policy_params(self):
        return {k: getattr(self, k) for k in self._policy_hparams.keys()}

    def get_all_params(self):
        result = {}
        dict_args = [self.get_general_params(), self.get_policy_params(), self.get_meta_params(), ]
        # if args in both general and meta, meta will overwrite general (seed)
        for dictionary in dict_args:
            result.update(dictionary)
        return result

    def keys(self):
        return self.__dict__()

    def __dict__(self):
        return {k for k in dir(self) if not k.startswith('__') and not callable(getattr(self, k))}

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __repr__(self):
        return self.__str__()

    def __iter__(self):
        for key in dir(self):
            if not key.startswith('__') and not callable(getattr(self, key)):
                yield key, getattr(self, key)

    def __str__(self):
        # This will combine the dict representation with the class's own attributes
        class_attrs = {k: getattr(self, k) for k in dir(self) if
                       not k.startswith('__') and not callable(getattr(self, k))}
        all_attrs = {**self, **class_attrs}
        return str(all_attrs)


class DQNHyperParams(OffPolicyRLHyperParameterSpace):
    _supported_algos = ("dqn", "ddqn")
    _policy_hparams = {
        "lr": common_hparams["lr"],  # learning rate
        "n_step": common_hparams["n_step"],
        "target_update_freq": common_hparams["target_update_freq"],
        "is_double": False,
        "use_dueling": False,
        "eps_test": common_hparams["eps_test"],
        "eps_train": common_hparams["eps_train"],
        "eps_train_final": common_hparams["eps_train_final"],
    }


# todo: add rainbow
# todo: add cat and rnn version of the following policies


class C51HyperParams(OffPolicyRLHyperParameterSpace):
    _supported_algos = ("c51", "c51-rnn")
    _policy_hparams = {"lr": common_hparams["lr"],
                       "num_atoms": 51,
                       "v_min": -20,
                       "v_max": 20,
                       "estimation_step": common_hparams["n_step"],
                       "target_update_freq": common_hparams["target_update_freq"], }


class SACHyperParams(OffPolicyRLHyperParameterSpace):
    _supported_algos = ("sac", "discrete-sac", "discrete-sac-rnn")
    _policy_hparams = {
        "actor_lr": common_hparams["lr"],
        "critic_lr": common_hparams["lr"],
        "alpha": [0.05, 0.1, 0.2],
        "n_step": common_hparams["n_step"],
        "tau": common_hparams["tau"],
        "start_timesteps": common_hparams["start_timesteps"],
    }


class TD3HyperParams(OffPolicyRLHyperParameterSpace):
    _supported_algos = ("td3",)
    _policy_hparams = {
        # "actor_lr": common_hparams["lr"],  # manually set to 0.1*critic_lr

        "critic_lr": common_hparams["lr"],
        "n_step": common_hparams["n_step"],
        "exploration_noise": common_hparams["exploration_noise"],
        "tau": common_hparams["tau"],
        "start_timesteps": common_hparams["start_timesteps"],
        "update_actor_freq": common_hparams["update_actor_freq"],
        "policy_noise": 0.05,  # todo: TBD
        "noise_clip": 0.1,
    }
