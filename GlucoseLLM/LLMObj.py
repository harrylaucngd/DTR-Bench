import os
import torch
import wandb
from GlucoseLLM.LLM_policy import LLM_DQN_Policy, LLM_PPO_Policy, LLM_Policy
from GlucoseLLM.LLM_hparams import LLMInference_HyperParams
from GlucoseLLM.models.llm_net import LLMPPO
from GlucoseLLM.models.timeLLM import timeLLM, LLMInference
from DTRBench.src.offpolicyRLHparams import OffPolicyRLHyperParameterSpace
from DTRBench.src.onpolicyRLHparams import OnPolicyRLHyperParameterSpace
from DTRBench.src.RLObj import DQNObjective, PPOObjective
from DTRBench.src.base_obj import RLObjective
from DTRBench.utils.wandb_fn import WandbLogger
from DTRBench.utils.misc import set_global_seed
from tianshou.utils.net.common import ActorCritic
from DTRBench.utils.network import define_continuous_critic
from torch.distributions import Distribution, Independent, Normal
from transformers import AdamW, get_scheduler


class LLM_DQN_Objective(DQNObjective):
    def __init__(self, env_name, env_args, hparam_space: OffPolicyRLHyperParameterSpace, device, **kwargs):
        super().__init__(env_name, env_args, hparam_space, device, **kwargs)

    def define_policy(
        self,
        # general hp
        gamma,
        lr,
        obs_mode,
        # dqn hp
        n_step,
        target_update_freq,
        is_double,
        # llm prompt
        llm_mode,
        sum_prob,
        gradient_accumulation_steps,
        *args,
        **kwargs,
    ):
        cat_num, stack_num = obs_mode[list(obs_mode.keys())[0]]["cat_num"], obs_mode[list(obs_mode.keys())[0]]["stack_num"]
        seq_len = cat_num * stack_num  # calculate sequence length
        # define model
        net = timeLLM(
            llm_name=llm_mode["llm"],
            action_size=self.action_shape,
            seq_len=seq_len,
            token_dim=llm_mode["token_dim"],
            patch_len=24,
            stride=6,
            d_model=2,
            dropout=0,
            n_heads=4,
            d_ff=64,
            dtype=torch.bfloat16,
            max_new_tokens=256,
        ).to(self.device)

        trainable_params = [p for p in net.parameters() if p.requires_grad]
        optimizer = AdamW(trainable_params, lr=lr, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)

        num_training_steps = self.meta_param["step_per_epoch"] * self.meta_param["epoch"]  # Total training steps
        num_warmup_steps = 0.05 * num_training_steps  # 5% of steps for warmup
        scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

        # define policy
        policy = LLM_DQN_Policy(
            net,
            optimizer,
            gamma,
            n_step,
            target_update_freq=target_update_freq,
            is_double=is_double,
            action_space=self.action_space,
            observation_space=self.state_space,
            summary_prob=sum_prob,
            lr_scheduler=scheduler,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        return policy


class LLM_PPO_Objective(PPOObjective):
    def __init__(self, env_name, env_args, hparam_space: OffPolicyRLHyperParameterSpace, device, **kwargs):
        super().__init__(env_name, env_args, hparam_space, device, **kwargs)

    def define_policy(
        self,
        gamma,
        lr,
        gae_lambda,
        vf_coef,
        ent_coef,
        eps_clip,
        value_clip,
        dual_clip,
        advantage_normalization,
        recompute_advantage,
        n_step,
        epoch,
        batch_size,
        linear,
        llm_mode,
        sum_prob,
        **kwargs,
    ):
        cat_num, stack_num = 48, 1
        actor = define_llm_ppo(
            self.state_shape,
            self.action_shape,
            unbounded=True,
            device=self.device,
            llm=llm_mode["llm"],
            token_dim=llm_mode["token_dim"],
            summary_prompt=sys_summary_prompt,
            act_prompt=sys_act_prompt,
        ).to(self.device)
        critic = define_continuous_critic(
            self.state_shape,
            self.action_shape,
            linear=linear,
            use_rnn=stack_num > 1,
            cat_num=cat_num,
            use_action_net=False,
            state_net_hidden_size=128,
            device=self.device,
        )
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
            action_bound_method="clip",
            action_space=self.action_space,
            eps_clip=eps_clip,
            value_clip=value_clip,
            dual_clip=dual_clip,
            advantage_normalization=advantage_normalization,
            recompute_advantage=recompute_advantage,
            sum_prob=sum_prob,
        )
        return policy


class LLMInferenceObjective(RLObjective):
    def __init__(self, env_name, env_args, hparam_space: LLMInference_HyperParams, device, **kwargs):
        super().__init__(env_name, env_args, hparam_space, device=device, **kwargs)

    def define_policy(self, llm_mode, **kwargs):
        net = LLMInference(llm=llm_mode["llm"], context_window=llm_mode["context_window"], device=self.device).to(self.device)
        return LLM_Policy(
            net,
            action_space=self.action_space,
            observation_space=self.state_space,
        )

    def wandb_search(self):
        self.logger = WandbLogger(train_interval=10, update_interval=100)
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

    def search_once(self, hparams: dict):
        # Define paths for logging
        hp_name = "-".join([f"{v}" for k, v in hparams.items() if k not in ["wandb_project_name", "log_dir"]])
        wandb.init(project=hparams["wandb_project_name"], config=hparams)
        self.log_path = os.path.join(self.meta_param["log_dir"], f"search_once/{hp_name}")
        os.makedirs(self.log_path, exist_ok=True)
        self.logger = WandbLogger(train_interval=10, update_interval=100)

        # Prepare the environment using the given hyperparameters
        self.prepare_env(int(hparams["seed"]), self.env_name, **self.env_args)
        set_global_seed(int(hparams["seed"]))

        # Define and train the policy
        print("Prepare policy")
        policy = self.define_policy(**{**hparams, **self.meta_param})
        print("Start testing")

        # Test the policy
        self.test_all_patients(policy, None, int(hparams["seed"]), self.logger, n_episode=20)
