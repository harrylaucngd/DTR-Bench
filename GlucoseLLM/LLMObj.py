import torch
import wandb
from GlucoseLLM.LLM_policy import LLM_DQN_Policy, LLM_Policy
from GlucoseLLM.LLM_hparams import LLMInference_HyperParams
from GlucoseLLM.models.llm_net import define_llm_network, LLM
from DTRBench.src.offpolicyRLHparams import OffPolicyRLHyperParameterSpace
from DTRBench.src.RLObj import DQNObjective
from DTRBench.src.base_obj import RLObjective
from DTRBench.utils.wandb import WandbLogger
from DTRBench.utils.misc import set_global_seed

universal_sys_prompt = ("You are a clinical specialist working with Type-1 Diabetic patients. Your primary goal is to"
                        " maintain a patient's blood glucose levels (the observation, received every 5 minutes) within"
                        " 70-140 mg/dL through the administration of insulin (the action). Insulin will reduce blood "
                        "glucose levels, while food intake, which is hidden, will increase blood glucose levels. You will"
                        "be penalized for blood glucose <70 or >140, and high insulin doses. Notably, low blood glucose"
                        "levels are much more dangerous. You should take caution to avoid overdosing insulin, thus"
                        "to avoid hypoglycemia. The insulin is given per 5 minutes and given in units/hour, ranging from 0 to 0.5.")

sys_obs_exp_prompt = universal_sys_prompt  # expertised system prompt of background knowledge for observation explanation

sys_Q_prompt = universal_sys_prompt + ("Please generate the expected discounted reward (i.e., Q(s, a)) for each insulin bins in the order of "
                "the following insulin dosage bins: [0, 0-0.05, 0.05-0.1, 0.1-0.15, 0.15-0.2, 0.2-0.25,"
                " 0.25-0.3, 0.3-0.35, 0.35-0.4, 0.4-0.45, 0.45-0.5]")  # expertised system prompt for series information description and Q value prediction

sys_act_exp_prompt = universal_sys_prompt  # expertised system prompt of background knowledge for action explanation

sys_summary_prompt = ("You are a clinical specialist working with Type-1 Diabetic patients. Your primary goal is to"
                        "summarize history glucose record and drug usage. You need to extract information such as"
                      " glucose record trend, drug dosage history, abnormal glucose signs and possible misuse of insulin."
                      " Please extract as much information as possible while keeping the answer short.")  # expertised system prompt of background knowledge for regulation summary

sys_llm_only_prompt = universal_sys_prompt  # expertised system prompt of background knowledge for action decision


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
                                 obs_exp_prompt=sys_obs_exp_prompt, Q_prompt=sys_Q_prompt, act_exp_prompt=sys_act_exp_prompt, summary_prompt=sys_summary_prompt)
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
    def __init__(self, env_name, env_args, hparam_space: LLMInference_HyperParams, device, **kwargs):
        super().__init__(env_name, env_args, hparam_space, device=device, **kwargs)

    def define_policy(self, llm_mode, **kwargs):
        net = LLM(llm=llm_mode["llm"], context_window=llm_mode["context_window"],
                  device=self.device, system_prompt=sys_llm_only_prompt).to(self.device)
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