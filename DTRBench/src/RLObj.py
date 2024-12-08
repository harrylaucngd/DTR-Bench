import os
import torch
from tianshou.data import VectorReplayBuffer, ReplayBuffer
from tianshou.exploration import GaussianNoise
from tianshou.policy import TD3Policy
from tianshou.policy import PPOPolicy
from tianshou.policy.modelfree.dqn import DQNPolicy
from tianshou.trainer import OffpolicyTrainer, OnpolicyTrainer
from DTRBench.src.base_obj import RLObjective
from DTRBench.src.offpolicyRLHparams import OffPolicyRLHyperParameterSpace
from DTRBench.src.onpolicyRLHparams import OnPolicyRLHyperParameterSpace
from DTRBench.utils.network import define_single_network, Critic, define_continuous_critic, Actor
from tianshou.utils.net.continuous import ActorProb
from tianshou.utils.net.common import ActorCritic
from torch.distributions import Distribution, Independent, Normal
from torch.distributions import Normal, TransformedDistribution, SigmoidTransform, Independent
from DTRBench.src.collector import GlucoseCollector as Collector
from DTRBench.naive_baselines.naive_baselines import ConstantPolicy, RandomPolicy
import torch.nn as nn
import numpy as np
from tianshou.policy import DQNPolicy
import torch.nn.functional as F


class DQNPolicyWithKnowledge(DQNPolicy):
    def exploration_noise(
            self,
            act,
            batch,
    ):
        if isinstance(act, np.ndarray) and not np.isclose(self.eps, 0.0):
            bsz = len(act)
            rand_mask = np.random.rand(bsz) < self.eps
            assert (
                    self.max_action_num is not None
            ), "Can't call this method before max_action_num was set in first forward"

            # Define the probability distribution
            p = [0] * self.max_action_num
            p[0] = (1 / (self.max_action_num - 1) * 2 + 1) * self.max_action_num
            total_sum = p[0] + (self.max_action_num - 1)  # Total sum for normalisation
            p[0] /= total_sum  # Normalise probability for action 0
            rest_probability = (1 - p[0]) / (self.max_action_num - 1)  # Split rest evenly
            p[1:] = [rest_probability] * (self.max_action_num - 1)

            # Sample actions based on the probability distribution
            rand_act = np.random.choice(self.max_action_num, size=bsz, p=p)

            if hasattr(batch.obs, "mask"):
                q = np.random.rand(bsz, self.max_action_num)  # [0, 1]
                q += batch.obs.mask
                rand_act = q.argmax(axis=1)  # Fallback to max if masking is present

            act[rand_mask] = rand_act[rand_mask]
        return act

class DQNObjective(RLObjective):
    def __init__(self, env_name, env_args, hparam_space: OffPolicyRLHyperParameterSpace, device, **kwargs):
        super().__init__(env_name, env_args, hparam_space, device, **kwargs)

    def define_policy(
        self,
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
        use_knowledge,
        *args,
        **kwargs,
    ):
        # define model
        cat_num, stack_num = obs_mode[list(obs_mode.keys())[0]]["cat_num"], obs_mode[list(obs_mode.keys())[0]]["stack_num"]
        net = define_single_network(
            self.state_shape,
            self.action_shape,
            use_dueling=use_dueling,
            use_rnn=stack_num > 1,
            device=self.device,
            linear=linear,
            cat_num=cat_num,
        )
        optim = torch.optim.Adam(net.parameters(), lr=lr)
        # define policy
        policy_cls = DQNPolicyWithKnowledge if use_knowledge else DQNPolicy
        policy = policy_cls(
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

    def run(self, policy, eps_test, eps_train, eps_train_final, step_per_collect, update_per_step, batch_size, start_timesteps, **kwargs):
        def save_best_fn(policy):
            torch.save(policy.state_dict(), os.path.join(self.log_path, "best_policy.pth"))

        def train_fn(epoch, env_step):
            # nature DQN setting, linear decay in the first 10k steps
            if env_step <= self.meta_param["epoch"] * self.meta_param["step_per_epoch"] * 0.95:
                eps = eps_train - env_step / (self.meta_param["epoch"] * self.meta_param["step_per_epoch"] * 0.95) * (
                    eps_train - eps_train_final
                )
            else:
                eps = eps_train_final
            policy.set_eps(eps)
            if env_step % 1000 == 0:
                self.logger.write("train/env_step", env_step, {"train/eps": eps})

        def test_fn(epoch, env_step):
            policy.set_eps(eps_test)

        # replay buffer: `save_last_obs` and `stack_num` can be removed together
        # when you have enough RAM
        if self.meta_param["training_num"] > 1:
            buffer = VectorReplayBuffer(
                self.meta_param["buffer_size"],
                buffer_num=len(self.train_envs),
                ignore_obs_next=False,
                save_only_last_obs=False,
                stack_num=1,  # stack is implemented in the env
            )
        else:
            buffer = ReplayBuffer(self.meta_param["buffer_size"], ignore_obs_next=False, save_only_last_obs=False, stack_num=1)
        if start_timesteps > 0:
            if os.path.exists(f"warmup_random_{start_timesteps}.hdf5"):
                buffer = buffer.load_hdf5(f"warmup_random_{start_timesteps}.hdf5")
            else:
                print(f"warmup with random policy for {start_timesteps} steps..")
                warmup_policy = ConstantPolicy(action_space=self.action_space, dose=0)
                warmup_collector = Collector(warmup_policy, self.train_envs, buffer, exploration_noise=True)
                warmup_collector.collect(n_step=start_timesteps)
                buffer.save_hdf5(f"warmup_random_{start_timesteps}.hdf5")

        # collector
        train_collector = Collector(policy, self.train_envs, buffer, exploration_noise=True)
        test_collector = Collector(policy, self.test_envs, exploration_noise=True)

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


class TD3Objective(RLObjective):
    # todo: linear does not work
    def __init__(self, env_name, env_args, hparam_space: OffPolicyRLHyperParameterSpace, device, **kwargs):
        super().__init__(env_name, env_args, hparam_space, device, **kwargs)

    def define_policy(
        self,
        gamma,
        critic_lr,
        n_step,
        obs_mode,
        tau,
        update_actor_freq,
        policy_noise,
        noise_clip,
        exploration_noise,
        linear,
        **kwargs,
    ):
        actor_lr = critic_lr * 0.1
        cat_num, stack_num = (obs_mode[list(obs_mode.keys())[0]]["cat_num"], obs_mode[list(obs_mode.keys())[0]]["stack_num"])
        min_action, max_action = self.action_space.low[0], self.action_space.high[0]
        net_a = define_single_network(
            self.state_shape,
            128,
            use_rnn=stack_num > 1,
            device=self.device,
            linear=linear,
            cat_num=cat_num,
            use_dueling=False,
        )
        actor = Actor(
            net_a,
            action_shape=self.action_shape,
            device=self.device,
            # last_layer_init=-10,
            final_activation=nn.Tanh(),
            preprocess_net_output_dim=128,
        ).to(self.device)

        # # init actor with orthogonal initialization and zeros bias
        # for m in actor.modules():
        #     if isinstance(m, torch.nn.Linear):
        #         torch.nn.init.zeros_(m.bias)
        #         m.weight.data.copy_(0.01 * m.weight.data)

        actor_optim = torch.optim.Adam(actor.parameters(), lr=actor_lr)

        critic1 = define_continuous_critic(
            self.state_shape,
            self.action_shape,
            linear=linear,
            use_rnn=stack_num > 1,
            cat_num=cat_num,
            state_net_hidden_size=127,
            action_net_hidden_size=1,
            device=self.device,
        )
        critic1_optim = torch.optim.Adam(critic1.parameters(), lr=critic_lr)
        critic2 = define_continuous_critic(
            self.state_shape,
            self.action_shape,
            linear=linear,
            use_rnn=stack_num > 1,
            cat_num=cat_num,
            state_net_hidden_size=127,
            action_net_hidden_size=1,
            device=self.device,
        )
        critic2_optim = torch.optim.Adam(critic2.parameters(), lr=critic_lr)

        exploration_noise *= max_action
        policy_noise *= max_action
        noise_clip *= max_action
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

    def run(
        self,
        policy,
        step_per_collect,
        update_per_step,
        batch_size,
        start_timesteps,
        # exploration_noise,
        # exploration_noise_final,
        **kwargs,
    ):

        # collector
        if self.meta_param["training_num"] > 1:
            buffer = VectorReplayBuffer(
                self.meta_param["buffer_size"], buffer_num=len(self.train_envs), ignore_obs_next=False, save_only_last_obs=False, stack_num=1
            )
        else:
            buffer = ReplayBuffer(self.meta_param["buffer_size"], ignore_obs_next=False, save_only_last_obs=False, stack_num=1)

        # collector
        train_collector = Collector(policy, self.train_envs, buffer, exploration_noise=True)
        test_collector = Collector(policy, self.test_envs, exploration_noise=False)
        if start_timesteps > 0:
            # todo: collect with random
            train_collector.collect(n_step=start_timesteps, random=True)

        # def train_fn(epoch, env_step):
        #     # nature DQN setting, linear decay in the first 10k steps
        #     if env_step <= self.meta_param["epoch"] * self.meta_param["step_per_epoch"] * 0.95:
        #         eps = exploration_noise - env_step / (self.meta_param["epoch"] * self.meta_param["step_per_epoch"] * 0.95) * \
        #               (exploration_noise - exploration_noise_final)
        #     else:
        #         eps = eps_train_final
        #     policy.set_exp_noise(eps)
        #     if env_step % 1000 == 0:
        #         self.logger.write("train/env_step", env_step, {"train/eps": eps})
        #
        # def test_fn(epoch, env_step):
        #     policy.set_eps(eps_test)
        def save_best_fn(policy):
            torch.save(policy.state_dict(), os.path.join(self.log_path, "best_policy.pth"))

        OffpolicyTrainer(
            policy,
            max_epoch=self.meta_param["epoch"],
            batch_size=batch_size,
            train_collector=train_collector,
            test_collector=test_collector,
            step_per_epoch=self.meta_param["step_per_epoch"],
            step_per_collect=step_per_collect,
            episode_per_test=self.meta_param["test_num"],
            train_fn=lambda epoch, env_step: None,
            test_fn=lambda epoch, env_step: None,
            stop_fn=self.early_stop_fn,
            save_best_fn=save_best_fn,
            logger=self.logger,
            update_per_step=update_per_step,
            save_checkpoint_fn=self.save_checkpoint_fn,
        ).run()

        # load the best policy to test again
        policy.load_state_dict(torch.load(os.path.join(self.log_path, "best_policy.pth")))
        return policy, lambda epoch, env_step: None


class PPOObjective(RLObjective):
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
        conditioned_sigma,
        advantage_normalization,
        recompute_advantage,
        n_step,
        epoch,
        batch_size,
        obs_mode,
        use_knowledge,
        linear,
        **kwargs,
    ):
        # todo use knowledge for PPO not done
        cat_num, stack_num = obs_mode[list(obs_mode.keys())[0]]["cat_num"], obs_mode[list(obs_mode.keys())[0]]["stack_num"]
        net_a = define_single_network(
            self.state_shape,
            self.action_shape,
            use_dueling=False,
            num_layer=3,
            hidden_size=128,
            use_rnn=stack_num > 1,
            device=self.device,
            cat_num=cat_num,
        )
        actor = ActorProb(
            net_a,
            self.action_shape,
            conditioned_sigma=conditioned_sigma,
            unbounded=True,
            device=self.device,
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

        def normal_dist(*loc_scale: tuple[torch.Tensor, torch.Tensor]) -> Distribution:
            loc, scale = loc_scale
            return Independent(Normal(loc, scale), 1)
        #
        # def logit_normal_dist(loc: torch.Tensor, scale: torch.Tensor) -> torch.distributions.Distribution:
        #     scale = F.softplus(scale) + 1e-5
        #     base_dist = Normal(loc, scale)
        #     transforms = [SigmoidTransform()]
        #     logit_normal = TransformedDistribution(base_dist, transforms)
        #     return Independent(logit_normal, 1)

        policy: PPOPolicy = PPOPolicy(
            actor=actor,
            critic=critic,
            optim=optim,
            dist_fn=normal_dist,
            discount_factor=gamma,
            gae_lambda=float(gae_lambda),
            vf_coef=vf_coef,
            ent_coef=ent_coef,
            action_scaling=True,
            action_bound_method="tanh" if use_knowledge else "clip",
            action_space=self.action_space,
            eps_clip=eps_clip,
            value_clip=value_clip,
            dual_clip=dual_clip,
            advantage_normalization=advantage_normalization,
            recompute_advantage=recompute_advantage,
        )
        return policy

    def run(self, policy, obs_mode, step_per_collect, repeat_per_collect, batch_size, start_timesteps, **kwargs):
        def save_best_fn(policy):
            torch.save(policy.state_dict(), os.path.join(self.log_path, "best_policy.pth"))

        # collector
        if self.meta_param["training_num"] > 1:
            buffer = VectorReplayBuffer(
                self.meta_param["buffer_size"], buffer_num=len(self.train_envs), ignore_obs_next=False, save_only_last_obs=False, stack_num=1
            )
        else:
            buffer = ReplayBuffer(self.meta_param["buffer_size"], ignore_obs_next=False, save_only_last_obs=False, stack_num=1)


        if start_timesteps > 0:
            if start_timesteps == 100000:
                buffer = buffer.load_hdf5("warmup_random_100000.hdf5")
            else:
                print(f"warmup with random policy for {start_timesteps} steps..")
                warmup_policy = RandomPolicy(min_act=0, max_act=5 if self.env_args["discrete"] else 0.1, action_space=self.action_space)
                warmup_collector = Collector(warmup_policy, self.train_envs, buffer, exploration_noise=True)
                warmup_collector.collect(n_step=start_timesteps)
                buffer.save_hdf5(f"warmup_random_{start_timesteps}.hdf5")
        # collector
        train_collector = Collector(policy, self.train_envs, buffer, exploration_noise=True)
        test_collector = Collector(policy, self.test_envs, exploration_noise=False)

        OnpolicyTrainer(
            policy,
            batch_size=batch_size,
            train_collector=train_collector,
            test_collector=test_collector,
            max_epoch=self.meta_param["epoch"],
            step_per_epoch=self.meta_param["step_per_epoch"],
            step_per_collect=step_per_collect,
            repeat_per_collect=repeat_per_collect,
            episode_per_test=self.meta_param["test_num"],
            train_fn=None,
            test_fn=None,
            stop_fn=self.early_stop_fn,
            save_best_fn=save_best_fn,
            logger=self.logger,
            save_checkpoint_fn=self.save_checkpoint_fn,
        ).run()

        # load the best policy to test again
        policy.load_state_dict(torch.load(os.path.join(self.log_path, "best_policy.pth")))
        return policy, lambda epoch, env_step: None
