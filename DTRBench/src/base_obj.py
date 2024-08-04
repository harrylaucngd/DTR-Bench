import os
from tqdm import tqdm
import wandb
from DTRBench.src.collector import GlucoseCollector as Collector
from dataclasses import asdict
from tianshou.policy.base import BasePolicy
from DTRBench.utils.wandb import WandbLogger
from torch.utils.tensorboard import SummaryWriter
from tianshou.trainer.utils import test_episode
from DTRGym.base import make_env
from DTRBench.utils.misc import set_global_seed
from DTRBench.src.offpolicyRLHparams import OffPolicyRLHyperParameterSpace


class RLObjective:
    def __init__(self, env_name, env_args:dict, hyperparam: OffPolicyRLHyperParameterSpace, device, **kwargs):
        # define high level parameters
        self.env_name = env_name
        self.env_args = env_args
        self.hyperparam = hyperparam
        self.meta_param = self.hyperparam.get_meta_params()
        self.device = device

    def get_search_space(self):
        return self.hyperparam.get_search_space()

    def prepare_env(self, seed, env_name, **env_kwargs):
        # prepare env
        self.env, self.train_envs, self.test_envs = make_env(env_name, int(seed),
                                                             self.meta_param["training_num"], 1, **env_kwargs)
        state_shape = self.env.observation_space.shape or self.env.observation_space.n
        self.state_space = self.env.observation_space
        action_shape = self.env.action_space.shape or self.env.action_space.n
        self.action_space = self.env.action_space
        if isinstance(state_shape, (tuple, list)):
            if len(state_shape) > 1:
                raise NotImplementedError("state shape > 1 not supported yet")
            self.state_shape = state_shape[0]
        else:
            self.state_shape = int(state_shape)
        if isinstance(action_shape, (tuple, list)):
            if len(action_shape) > 1:
                raise NotImplementedError("action shape > 1 not supported yet")
            self.action_shape = action_shape
        else:
            self.action_shape = int(action_shape)

    def wandb_search(self):
        # init wandb to get hparams
        self.logger = WandbLogger(train_interval=10, update_interval=100)
        hparams = wandb.config
        # get names
        hp_name = "-".join([f"{v}" if not isinstance(v, dict) else f"{list(v.keys())[0]}"
                            for k, v in hparams.items() if
                            k not in self.meta_param.keys() or k != "wandb_project_name"])
        self.log_path = os.path.join(self.meta_param["log_dir"], f"{hp_name}")

        print(f"logging to {self.log_path}")
        os.makedirs(self.log_path, exist_ok=True)
        # writer = SummaryWriter(log_dir=self.log_path)
        #
        # self.logger.load(writer)

        self.prepare_env(int(hparams["seed"]), self.env_name, **self.env_args)
        set_global_seed(int(hparams["seed"]))

        # start training
        print("prepare policy")
        self.policy = self.define_policy(**{**hparams, **self.meta_param})
        print("start training!")
        best_policy, test_fn = self.run(self.policy, **{**hparams, **self.meta_param})

        # test on all envs
        self.test_all_patients(best_policy, test_fn, int(hparams["seed"]), self.logger, n_episode=20)

    def test_all_patients(self, policy, test_fn, seed, logger, n_episode=20):
        for patient_name in tqdm(["adolescent#001", "adolescent#002", "adolescent#003", "adolescent#004",
                                  "adult#001", "adult#002", "adult#003", "adult#004",
                                  "child#001", "child#002", "child#003", "child#004", "child#005"],
                                 desc="final_testing"):
            self.prepare_env(seed, "SimGlucoseEnv-single-patient", patient_name=patient_name, **self.env_args)
            test_collectors = Collector(policy, self.test_envs, exploration_noise=True)
            result = test_episode(policy, test_collectors, n_episode=n_episode, test_fn=test_fn, epoch=0)
            result_dict = self.logger.prepare_dict_for_logging(asdict(result), f"final_test/{patient_name}")
            logger.write("this arg doesn't matter", 0, result_dict)

    def search_once(self, hparams: dict, metric="best_reward"):
        # todo: this is wrong
        raise NotImplementedError

    def early_stop_fn(self, mean_rewards):
        # todo: early stopping is not working for now, because stop_fn is called at each training step
        return False

    # no checkpoint saving needed
    def save_checkpoint_fn(self, epoch, env_step, gradient_step):
        return

    def define_policy(self, *args, **kwargs) -> BasePolicy:
        raise NotImplementedError

    def run(self, *args, **kwargs):
        raise NotImplementedError
