import torch
import wandb
from GlucoseLLM.LLM_policy import LLM_DQN_Policy, LLM_PPO_Policy, LLM_Policy
from GlucoseLLM.LLM_hparams import LLMInference_HyperParams
from GlucoseLLM.models.llm_net import define_llm_dqn, define_llm_ppo, LLM
from DTRBench.src.offpolicyRLHparams import OffPolicyRLHyperParameterSpace
from DTRBench.src.onpolicyRLHparams import OnPolicyRLHyperParameterSpace
from DTRBench.src.RLObj import DQNObjective, PPOObjective
from DTRBench.src.base_obj import RLObjective
from DTRBench.utils.wandb import WandbLogger
from DTRBench.utils.misc import set_global_seed
from tianshou.utils.net.common import ActorCritic
from DTRBench.utils.network import define_continuous_critic
from torch.distributions import Distribution, Independent, Normal

universal_sys_prompt = ("You are a clinical specialist working with Type-1 Diabetic patients. Your primary goal is to"
                        " maintain a patient's blood glucose levels (the observation, received every 5 minutes) within"
                        " 70-140 mg/dL through the administration of insulin (the action). Insulin will reduce blood "
                        "glucose levels, while food intake, which is hidden, will increase blood glucose levels. You will"
                        "be penalized for blood glucose <70 or >140, and high insulin doses. Notably, low blood glucose"
                        "levels are much more dangerous. You should take caution to avoid overdosing insulin, thus"
                        "to avoid hypoglycemia. The insulin is given per 5 minutes and given in units/hour, ranging from 0 to 0.5. ")

sys_Q_prompt = universal_sys_prompt + ("Please generate the expected discounted reward (i.e., Q(s, a)) for each insulin bins in the order of "
                "the following insulin dosage bins: [0, 0-0.05, 0.05-0.1, 0.1-0.15, 0.15-0.2, 0.2-0.25,"
                " 0.25-0.3, 0.3-0.35, 0.35-0.4, 0.4-0.45, 0.45-0.5]. ")  # expertised system prompt for series information description and Q value prediction

sys_act_prompt = universal_sys_prompt + ("Please generate the expected discounted reward (i.e., Q(s, a)) for each insulin bins in the order of "
                "the following insulin dosage bins: [0, 0-0.05, 0.05-0.1, 0.1-0.15, 0.15-0.2, 0.2-0.25,"
                " 0.25-0.3, 0.3-0.35, 0.35-0.4, 0.4-0.45, 0.45-0.5]. ")  # expertised system prompt for series information description and action prediction

sys_summary_prompt = ("You are a clinical specialist working with Type-1 Diabetic patients. Your primary goal is to"
                        "summarize history glucose record and drug usage. You need to extract information such as"
                      " glucose record trend, drug dosage history, abnormal glucose signs and possible misuse of insulin."
                      " Please extract as much information as possible while keeping the answer short. ")  # expertised system prompt of background knowledge for regulation summary

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
                      llm_mode, sum_prob,
                      *args, **kwargs
                      ):
        # define model
        net = define_llm_dqn(self.action_shape, device=self.device, llm=llm_mode["llm"],
                             token_dim=llm_mode["token_dim"], Q_prompt=sys_Q_prompt,
                             summary_prompt=sys_summary_prompt)
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
            sum_prob=sum_prob,
        )
        return policy


class LLM_PPO_Objective(PPOObjective):
    def __init__(self, env_name, env_args, hparam_space: OffPolicyRLHyperParameterSpace, device, **kwargs):
        super().__init__(env_name, env_args, hparam_space, device, **kwargs)

    def define_policy(self, gamma, lr, gae_lambda, vf_coef, ent_coef, eps_clip, value_clip, dual_clip,
                      advantage_normalization, recompute_advantage, n_step, epoch, batch_size, linear, llm_mode, sum_prob, **kwargs):
        cat_num, stack_num = 48, 1
        actor = define_llm_ppo(self.action_shape, unbounded=True,
                               device=self.device, llm=llm_mode["llm"], token_dim=llm_mode["token_dim"],
                               summary_prompt=sys_summary_prompt, act_prompt=sys_act_prompt).to(self.device)
        critic = define_continuous_critic(self.state_shape, self.action_shape, linear=linear, use_rnn=stack_num > 1,
                                          cat_num=cat_num, use_action_net=False, state_net_hidden_size=128, 
                                          device=self.device)
        actor_critic = ActorCritic(actor, critic)
        optim = torch.optim.Adam(actor_critic.parameters(), lr=lr)

        # torch.nn.init.constant_(actor.sigma_param, -0.5)
        # for m in actor_critic.modules():
        #     if isinstance(m, torch.nn.Linear):
        #         # orthogonal initialization
        #         torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
        #         torch.nn.init.zeros_(m.bias)
        # do last policy layer scaling, this will make initial actions have (close to)
        # 0 mean and std, and will help boost performances,
        # see https://arxiv.org/abs/2006.05990, Fig.24 for details
        for m in actor.mu.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.zeros_(m.bias)
                m.weight.data.copy_(0.01 * m.weight.data)

        def dist(*loc_scale: tuple[torch.Tensor, torch.Tensor]) -> Distribution:
            loc, scale = loc_scale
            return Independent(Normal(loc, scale), 1)

        policy: LLM_PPO_Policy = LLM_PPO_Policy(
            actor=actor,
            critic=critic,
            optim=optim,
            dist_fn=dist,
            discount_factor=gamma,
            gae_lambda=float(gae_lambda),
            vf_coef=vf_coef,
            ent_coef=ent_coef,
            action_scaling=True,
            action_bound_method='clip',
            action_space=self.action_space,
            eps_clip=eps_clip,
            value_clip=value_clip,
            dual_clip=dual_clip,
            advantage_normalization=advantage_normalization,
            recompute_advantage=recompute_advantage,
            sum_prob=sum_prob,
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
