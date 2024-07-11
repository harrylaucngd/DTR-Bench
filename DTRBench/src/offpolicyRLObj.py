import os
import numpy as np
import torch
from tianshou.data import Collector, VectorReplayBuffer, ReplayBuffer
from tianshou.exploration import GaussianNoise
from GlucoseLLM.LLM_policy import LLM_DQN_Policy
from tianshou.policy import DDPGPolicy, \
    TD3Policy, SACPolicy, REDQPolicy, C51Policy, DiscreteSACPolicy
from tianshou.policy import PPOPolicy
from tianshou.policy.modelfree.dqn import DQNPolicy
from tianshou.trainer import OffpolicyTrainer
from DTRBench.src.base_obj import RLObjective
from DTRBench.src.offpolicyRLHparams import OffPolicyRLHyperParameterSpace
from DTRBench.utils.network import define_llm_network, define_single_network, Actor, Critic, define_continuous_critic


class DQNObjective(RLObjective):
    def __init__(self, env_name, env_args, hparam_space: OffPolicyRLHyperParameterSpace, device, **kwargs):
        super().__init__(env_name, env_args, hparam_space, device, **kwargs)

    def define_policy(self,
                      # general hp
                      gamma,
                      lr,
                      obs_mode,
                      linear,

                      # dqn hp
                      n_step,
                      target_update_freq,
                      is_double,
                      use_dueling,
                      **kwargs
                      ):
        # define model
        cat_num, stack_num = obs_mode[list(obs_mode.keys())[0]]["cat_num"], obs_mode[list(obs_mode.keys())[0]]["stack_num"]
        if obs_mode == "cat":
            cat_num, stack_num = 1, 1  # cat is the deprecated version of cur, we will use cur instead
        net = define_single_network(self.state_shape, self.action_shape, use_dueling=use_dueling,
                                    use_rnn=stack_num > 1, device=self.device, linear=linear, cat_num=cat_num)
        optim = torch.optim.Adam(net.parameters(), lr=lr)
        # define policy
        policy = DQNPolicy(
            model=net,
            optim=optim,
            discount_factor=gamma,
            estimation_step=n_step,
            target_update_freq=target_update_freq,
            is_double=is_double,  # we will have a separate runner for double dqn
            action_space=self.action_space,
            observation_space=self.state_space,
        )
        return policy

    def run(self, policy,
            seed,
            eps_test,
            eps_train,
            eps_train_final,
            obs_mode,
            step_per_collect,
            update_per_step,
            batch_size,
            **kwargs
            ):
        def save_best_fn(policy):
            torch.save(policy.state_dict(), os.path.join(self.log_path, "best_policy.pth"))

        def train_fn(epoch, env_step):
            # nature DQN setting, linear decay in the first 10k steps
            if env_step <= self.meta_param["epoch"] * self.meta_param["step_per_epoch"] * 0.95:
                eps = eps_train - env_step / (self.meta_param["epoch"] * self.meta_param["step_per_epoch"] * 0.95) * \
                      (eps_train - eps_train_final)
            else:
                eps = eps_train_final
            policy.set_eps(eps)
            if env_step % 1000 == 0:
                self.logger.write("train/env_step", env_step, {"train/eps": eps})

        def test_fn(epoch, env_step):
            policy.set_eps(eps_test)
        cat_num, stack_num = obs_mode[list(obs_mode.keys())[0]]["cat_num"], obs_mode[list(obs_mode.keys())[0]]["stack_num"]
        if obs_mode == "cat":
            cat_num, stack_num = 1, 1 # cat is the deprecated version of cur, we will use cur instead
        assert not (cat_num > 1 and stack_num > 1), "does not support both categorical and frame stack"
        stack_num = max(stack_num, cat_num)
        # seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        # replay buffer: `save_last_obs` and `stack_num` can be removed together
        # when you have enough RAM
        if self.meta_param["training_num"] > 1:
            buffer = VectorReplayBuffer(
                self.meta_param["buffer_size"],
                buffer_num=len(self.train_envs),
                ignore_obs_next=False,
                save_only_last_obs=False,
                stack_num=stack_num
            )
        else:
            buffer = ReplayBuffer(self.meta_param["buffer_size"],
                                  ignore_obs_next=False,
                                  save_only_last_obs=False,
                                  stack_num=stack_num)

        # collector
        train_collector = Collector(policy, self.train_envs, buffer, exploration_noise=True)
        test_collector = Collector(policy, self.test_envs, exploration_noise=False)

        OffpolicyTrainer(
            policy,
            max_epoch=self.meta_param["epoch"],
            batch_size=batch_size,
            train_collector=train_collector,
            test_collector=test_collector,
            step_per_epoch=self.meta_param["step_per_epoch"],
            step_per_collect=step_per_collect,
            episode_per_test=self.meta_param["test_num"],
            train_fn=train_fn,
            test_fn=test_fn,
            stop_fn=self.early_stop_fn,
            save_best_fn=save_best_fn,
            logger=self.logger,
            update_per_step=update_per_step,
            save_checkpoint_fn=self.save_checkpoint_fn,
        ).run()

        # load the best policy to test again
        policy.load_state_dict(torch.load(os.path.join(self.log_path, "best_policy.pth")))
        return policy, test_fn


class LLM_DQN_Objective(DQNObjective):
    def __init__(self, env_name, env_args, hparam_space: OffPolicyRLHyperParameterSpace, device, llm, llm_dim,
                 need_obs_explain, need_act_explain, need_summary, exp_freq, **kwargs):
        super().__init__(env_name, env_args, hparam_space, device, **kwargs)
        self.llm = llm
        self.llm_dim = llm_dim
        self.need_obs_explain = need_obs_explain
        self.need_act_explain = need_act_explain
        self.need_summary = need_summary
        self.exp_freq = exp_freq

    def define_policy(self,
                      # general hp
                      gamma,
                      lr,

                      # dqn hp
                      n_step,
                      target_update_freq,
                      **kwargs
                      ):
        # define model
        net = define_llm_network(self.state_shape, self.action_shape,   # Changing to GlucoseLLM
                                    device=self.device, llm=self.llm, llm_dim=self.llm_dim)
        optim = torch.optim.Adam(net.parameters(), lr=lr)
        # define policy
        policy = LLM_DQN_Policy(
            net,
            optim,
            gamma,
            n_step,
            target_update_freq=target_update_freq,
            need_obs_explain=self.need_obs_explain,
            need_act_explain=self.need_act_explain,
            need_summary=self.need_summary,
            exp_freq=self.exp_freq,
            action_space=self.action_space,
            observation_space=self.state_space,
        )
        return policy

    def run(self, policy,
            seed,
            eps_test,
            eps_train,
            eps_train_final,
            step_per_collect,
            update_per_step,
            batch_size,
            **kwargs
            ):
        def save_best_fn(policy):
            torch.save(policy.state_dict(), os.path.join(self.log_path, "policy.pth"))

        def train_fn(epoch, env_step):
            # nature DQN setting, linear decay in the first 10k steps
            if env_step <= self.meta_param["epoch"] * self.meta_param["step_per_epoch"] * 0.95:
                eps = eps_train - env_step / (self.meta_param["epoch"] * self.meta_param["step_per_epoch"] * 0.95) * \
                      (eps_train - eps_train_final)
            else:
                eps = eps_train_final
            policy.set_eps(eps)
            if env_step % 1000 == 0:
                self.logger.write("train/env_step", env_step, {"train/eps": eps})

        def test_fn(epoch, env_step):
            policy.set_eps(eps_test)

        # seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        # replay buffer: `save_last_obs` and `stack_num` can be removed together
        # when you have enough RAM
        if self.meta_param["training_num"] > 1:
            buffer = VectorReplayBuffer(
                self.meta_param["buffer_size"],
                buffer_num=len(self.train_envs),
                ignore_obs_next=False,
                save_only_last_obs=False
            )
        else:
            buffer = ReplayBuffer(self.meta_param["buffer_size"],
                                  ignore_obs_next=False,
                                  save_only_last_obs=False
                                  )

        # collector
        train_collector = Collector(policy, self.train_envs, buffer, exploration_noise=True)
        test_collector = Collector(policy, self.test_envs, exploration_noise=False)

        OffpolicyTrainer(
            policy,
            max_epoch=self.meta_param["epoch"],
            batch_size=batch_size,
            train_collector=train_collector,
            test_collector=test_collector,
            step_per_epoch=self.meta_param["step_per_epoch"],
            step_per_collect=step_per_collect,
            episode_per_test=self.meta_param["test_num"],
            train_fn=train_fn,
            test_fn=test_fn,
            stop_fn=self.early_stop_fn,
            save_best_fn=save_best_fn,
            logger=self.logger,
            update_per_step=update_per_step,
            save_checkpoint_fn=self.save_checkpoint_fn,
        ).run()


        # load the best policy to test again
        policy.load_state_dict(torch.load(os.path.join(self.log_path, "best_policy.pth")))
        return policy, test_fn


# class SACObjective(RLObjective):
#     # todo: linear does not work
#     def __init__(self, env_name, hparam_space: OffPolicyRLHyperParameterSpace, device,
#                  **kwargs):
#         super().__init__(env_name, hparam_space, device, **kwargs)
#
#     def define_policy(self, gamma,
#                       stack_num,
#                       actor_lr,
#                       critic_lr,
#                       alpha,
#                       n_step,
#                       tau,
#                       cat_num,
#                       linear,
#                       **kwargs, ):
#         hidden_sizes = [256, 256, 256] if not linear else []
#
#         # model
#         net_a = Net(self.state_shape, hidden_sizes=hidden_sizes, device=self.device, cat_num=cat_num)
#         actor = ActorProb(
#             net_a,
#             self.action_shape,
#             device=self.device,
#             unbounded=True,
#             conditioned_sigma=True,
#         ).to(self.device)
#         actor_optim = torch.optim.Adam(actor.parameters(), lr=actor_lr)
#         net_c1 = Net(
#             self.state_shape,
#             self.action_shape,
#             hidden_sizes=hidden_sizes,
#             concat=True,
#             device=self.device,
#             cat_num=cat_num
#         )
#         net_c2 = Net(
#             self.state_shape,
#             self.action_shape,
#             hidden_sizes=hidden_sizes,
#             concat=True,
#             device=self.device,
#             cat_num=cat_num
#         )
#         critic1 = Critic(net_c1, device=self.device).to(self.device)
#         critic1_optim = torch.optim.Adam(critic1.parameters(), lr=critic_lr)
#         critic2 = Critic(net_c2, device=self.device).to(self.device)
#         critic2_optim = torch.optim.Adam(critic2.parameters(), lr=critic_lr)
#
#         policy = SACPolicy(
#             actor,
#             actor_optim,
#             critic1,
#             critic1_optim,
#             critic2,
#             critic2_optim,
#             tau=tau,
#             gamma=gamma,
#             alpha=alpha,
#             estimation_step=n_step,
#             action_space=self.action_space,
#         )
#         return policy
#
#     def run(self, policy,
#             stack_num,
#             cat_num,
#             step_per_collect,
#             update_per_step,
#             batch_size,
#             start_timesteps,
#             **kwargs):
#         assert not (cat_num > 1 and stack_num > 1), "does not support both categorical and frame stack"
#         stack_num = max(stack_num, cat_num)
#         # collector
#         if self.meta_param["training_num"] > 1:
#             buffer = VectorReplayBuffer(self.meta_param["buffer_size"], len(self.train_envs), stack_num=stack_num)
#         else:
#             buffer = ReplayBuffer(self.meta_param["buffer_size"], stack_num=stack_num)
#         train_collector = Collector(policy, self.train_envs, buffer, exploration_noise=True)
#         test_collector = Collector(policy, self.train_envs)
#         if start_timesteps > 0:
#             train_collector.collect(n_step=start_timesteps, random=True)
#
#         def save_best_fn(policy):
#             torch.save(policy.state_dict(), os.path.join(self.log_path, "policy.pth"))
#
#         result = offpolicy_trainer(
#             policy,
#             train_collector,
#             test_collector,
#             self.meta_param["epoch"],
#             self.meta_param["step_per_epoch"],
#             step_per_collect,
#             self.meta_param["test_num"],
#             batch_size,
#             save_best_fn=save_best_fn,
#             logger=self.logger,
#             update_per_step=update_per_step,
#             stop_fn=self.early_stop_fn,
#             save_checkpoint_fn=self.save_checkpoint_fn
#         )
#         return result


class TD3Objective(RLObjective):
    # todo: linear does not work
    def __init__(self, env_name, env_args, hparam_space: OffPolicyRLHyperParameterSpace, device, **kwargs):
        super().__init__(env_name, env_args, hparam_space, device, **kwargs)

    def define_policy(self, gamma,
                      actor_lr,
                      critic_lr,
                      n_step,
                      obs_mode,

                      tau,
                      update_actor_freq,
                      policy_noise,
                      noise_clip,
                      exploration_noise,
                      linear,
                      **kwargs, ):

        cat_num, stack_num = (obs_mode[list(obs_mode.keys())[0]]["cat_num"],
                              obs_mode[list(obs_mode.keys())[0]]["stack_num"])
        min_action, max_action = self.action_space.low[0], self.action_space.high[0]
        net_a = define_single_network(self.state_shape, self.action_shape,
                                      use_rnn=stack_num > 1, device=self.device, linear=linear, cat_num=cat_num,
                                      use_dueling=False,)
        actor = Actor(net_a, min_action=min_action, max_action=max_action).to(self.device)
        actor_optim = torch.optim.Adam(actor.parameters(), lr=actor_lr)

        critic1 = define_continuous_critic(self.state_shape, self.action_shape, use_rnn=stack_num > 1, linear=linear,
                                           device=self.device, cat_num=cat_num)
        critic1_optim = torch.optim.Adam(critic1.parameters(), lr=critic_lr)
        critic2 = define_continuous_critic(self.state_shape, self.action_shape, use_rnn=stack_num > 1, linear=linear,
                                           device=self.device, cat_num=cat_num)
        critic2_optim = torch.optim.Adam(critic2.parameters(), lr=critic_lr)

        policy = TD3Policy(
            actor=actor,
            actor_optim=actor_optim,
            critic=critic1,
            critic_optim=critic1_optim,
            critic2=critic2,
            critic2_optim=critic2_optim,
            tau=tau,
            gamma=gamma,
            exploration_noise=GaussianNoise(sigma=exploration_noise),
            policy_noise=policy_noise,
            update_actor_freq=update_actor_freq,
            noise_clip=noise_clip,
            estimation_step=n_step,
            action_space=self.action_space,
        )
        return policy

    def run(self, policy,
            stack_num,
            cat_num,
            step_per_collect,
            update_per_step,
            batch_size,
            start_timesteps,
            **kwargs):
        assert not (cat_num > 1 and stack_num > 1), "does not support both categorical and frame stack"
        stack_num = max(stack_num, cat_num)
        # collector
        if self.meta_param["training_num"] > 1:
            buffer = VectorReplayBuffer(self.meta_param["buffer_size"], len(self.train_envs), stack_num=stack_num)
        else:
            buffer = ReplayBuffer(self.meta_param["buffer_size"], stack_num=stack_num)
        train_collector = Collector(policy, self.train_envs, buffer, exploration_noise=True)
        test_collector = Collector(policy, self.train_envs)
        if start_timesteps > 0:
            train_collector.collect(n_step=start_timesteps, random=True)

        def save_best_fn(policy):
            torch.save(policy.state_dict(), os.path.join(self.log_path, "policy.pth"))

        result = offpolicy_trainer(
            policy,
            train_collector,
            test_collector,
            self.meta_param["epoch"],
            self.meta_param["step_per_epoch"],
            step_per_collect,
            self.meta_param["test_num"],
            batch_size,
            save_best_fn=save_best_fn,
            logger=self.logger,
            update_per_step=update_per_step,
            stop_fn=self.early_stop_fn,
            save_checkpoint_fn=self.save_checkpoint_fn
        )
        return result

    # todo: add PPO objective
