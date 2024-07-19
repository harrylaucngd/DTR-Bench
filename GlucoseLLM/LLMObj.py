import torch
import wandb
from GlucoseLLM.LLM_policy import LLM_DQN_Policy, LLM_Policy
from GlucoseLLM.LLM_hparams import LLM_HyperParams
from GlucoseLLM.models.llm_net import define_llm_network, define_llm
from DTRBench.src.offpolicyRLHparams import OffPolicyRLHyperParameterSpace
from DTRBench.src.RLObj import DQNObjective
from DTRBench.src.base_obj import RLObjective
from DTRBench.utils.wandb import WandbLogger
from DTRBench.utils.misc import set_global_seed


obs_exp_prompt = ("The Simglucose environment is a simulation environment designed to mimic the physiological dynamics "
                  "of glucose metabolism in humans, often used in research of glucose control. "
                  "The primary goal in the Simglucose environment is to maintain a patient's blood glucose levels "
                  "(the observation) within a target range through the administration of insulin (the action). "
                  "The reason for a high value observation (high Blood Glucose Level (BG): the current blood glucose "
                  "concentration in mg/dL) is typically in last/last several timestep, more insulin (action) was "
                  "injected, a raising or high level of action is witnessed, and vice versa. The following would be "
                  "conversation history between user and you as an assistant, where the user gives blood glucose "
                  "observation and the agent's action (Insulin Bolus Dose) at every timestep, and next timestep's "
                  "observation, then the assistant give explaination to the observation at every next timestep.")  # expertised system prompt of background knowledge for observation explanation

Q_prompt = ("The Simglucose environment is a simulation environment designed to mimic the physiological dynamics of "
            "glucose metabolism in humans, often used in research of glucose control. The primary goal in the "
            "Simglucose environment is to maintain a patient's blood glucose levels (the observation) within a target "
            "range through the administration of insulin (the action). 5 number of actions (Insulin Bolus Dose) "
            "represents 5 degrees of insulin injection to restrain high blood glucose level. In Q-learning, "
            "the Q-value represents the expected future rewards for taking a given action in a given state, "
            "with high Q-values indicating more favorable actions and low Q-values indicating less favorable actions. "
            "So for a q-learning agent, if the blood glucose level is observed to be high, the q value of the high "
            "value action should be high, and q value of the low value action should be low, and vice versa for low "
            "blood glucose level. The following would be conversation history between user and you as an assistant, "
            "where the user gives blood glucose observation and the agent's action (Insulin Bolus Dose) at every timestep, "
            "then the assistant give explaination to both the observation and action at every timestep.")  # expertised system prompt for series information description and Q value prediction

act_exp_prompt = ("The Simglucose environment is a simulation environment designed to mimic the physiological dynamics "
                  "of glucose metabolism in humans, often used in research of glucose control. The primary goal in the "
                  "Simglucose environment is to maintain a patient's blood glucose levels (the observation) within a "
                  "target range through the administration of insulin (the action). The reason for a high value action "
                  "(high Insulin Bolus Dose measured in units (U) of insulin) is typically in current timestep or the "
                  "past several timesteps, a relatively high value of Blood Glucose Level (BG): the current blood "
                  "glucose concentration in mg/dL is observed (low observation), thus the patient needs more insulin "
                  "to prevent the blood glucose from getting too high, and vice versa. The following would be conversation "
                  "history between user and you as an assistant, where the user gives blood glucose observation and "
                  "the agent's action (Insulin Bolus Dose) at every timestep, then the assistant give explaination to the "
                  "action at every timestep.")  # expertised system prompt of background knowledge for action explanation
summary_prompt = ("The Simglucose environment is a simulation environment designed to mimic the physiological dynamics "
                  "of glucose metabolism in humans, often used in research of glucose control. The primary goal in the "
                  "Simglucose environment is to maintain a patient's blood glucose levels (the observation) within a "
                  "target range through the administration of insulin (the action). The reason for a high value action "
                  "(high Insulin Bolus Dose measured in units (U) of insulin) is typically in current timestep or the "
                  "past several timesteps, a relatively high value of Blood Glucose Level (BG): the current blood "
                  "glucose concentration in mg/dL is observed (low observation), thus the patient needs more insulin "
                  "to prevent the blood glucose from getting too high, and vice versa. The reason for a high value "
                  "observation (high Blood Glucose Level (BG): the current blood glucose concentration in mg/dL) is "
                  "typically in last/last several timestep, more insulin (action) was injected, a raising or high level "
                  "of action is witnessed, and vice versa.")  # expertised system prompt of background knowledge for regulation summary


class LLM_DQN_Objective(DQNObjective):
    def __init__(self, env_name, env_args, hparam_space: OffPolicyRLHyperParameterSpace, device, **kwargs):
        super().__init__(env_name, env_args, hparam_space, device, **kwargs)

    def define_policy(self,
                      # general hp
                      gamma, lr,
                      # dqn hp
                      n_step, target_update_freq, is_double,
                      # llm prompt
                      llm_mode, need_obs_explain, need_act_explain, need_summary, exp_freq,
                      *args, **kwargs
                      ):
        # define model
        net = define_llm_network(self.state_shape, self.action_shape,
                                 device=self.device, llm=llm_mode["llm"], token_dim=llm_mode["token_dim"],
                                 obs_exp_prompt=obs_exp_prompt, Q_prompt=Q_prompt, act_exp_prompt=act_exp_prompt, summary_prompt=summary_prompt)
        optim = torch.optim.Adam(net.parameters(), lr=lr)
        # define policy
        policy = LLM_DQN_Policy(
            net,
            optim,
            gamma,
            n_step,
            target_update_freq=target_update_freq,
            is_double=is_double,
            action_space=self.action_space,
            observation_space=self.state_space,
            need_obs_explain=need_obs_explain,
            need_act_explain=need_act_explain,
            need_summary=need_summary,
            exp_freq=exp_freq,
        )
        return policy


class LLM_Objective(RLObjective):
    def __init__(self, env_name, env_args, hparam_space: LLM_HyperParams, device, **kwargs):
        super().__init__(env_name, env_args, hparam_space, device=device, **kwargs)

    def define_policy(self, llm_mode, **kwargs):
        net = define_llm(llm=llm_mode["llm"], context_window=llm_mode["context_window"],
                                 device=self.device)
        return LLM_Policy(
            net,
            action_space=self.action_space,
            observation_space=self.state_space,
        )

    def wandb_search(self):
        self.logger = WandbLogger(train_interval=24 * 15)
        self.meta_param["training_num"] = 1
        self.meta_param["num_actions"] = None
        hparams = wandb.config

        self.prepare_env(int(hparams["seed"]), self.env_name, **self.env_args)
        set_global_seed(int(hparams["seed"]))

        # start training
        print("prepare policy")
        self.policy = self.define_policy(**{**hparams, **self.meta_param})

        # test on all envs
        self.test_all_patients(self.policy, None, int(hparams["seed"]), self.logger, n_episode=20)
