from typing import Any, Literal, cast

import random
import gymnasium as gym
import numpy as np
import torch

from tianshou.data import Batch, to_numpy
from tianshou.data.batch import BatchProtocol
from tianshou.data.types import (
    ModelOutputBatchProtocol,
    ObsBatchProtocol,
    RolloutBatchProtocol,
    DistBatchProtocol,
)
from tianshou.policy import BasePolicy, DQNPolicy, PPOPolicy
from tianshou.policy.base import TLearningRateScheduler, TTrainingStats
from tianshou.policy.modelfree.pg import TDistributionFunction
from tianshou.utils.net.common import ActorCritic

from GlucoseLLM.models.llm_net import LLMDQN, LLMPPO

from GlucoseLLM.prompt_pipeline import summary_reprogramming, q_prompt_reprogramming, act_prompt_reprogramming, obs2text, text2act


class LLM_DQN_Policy(DQNPolicy):
    """
    Implementation of LLM-DQN policy.
    """

    def __init__(
            self,
            model: LLMDQN,
            optim: torch.optim.Optimizer,
            discount_factor: float = 0.99,
            estimation_step: int = 1,
            target_update_freq: int = 0,
            reward_normalization: bool = False,
            is_double: bool = True,
            clip_loss_grad: bool = False,
            action_space: gym.spaces.Discrete | None = None,
            observation_space: gym.Space | None = None,
            lr_scheduler: TLearningRateScheduler | None = None,
            sum_prob=0,
    ) -> None:
        BasePolicy.__init__(
            self,
            action_space=action_space,
            observation_space=observation_space,
            action_scaling=False,
            action_bound_method=None,
            lr_scheduler=lr_scheduler,
        )
        self.model = model
        self.optim = optim
        self.eps = 0.0
        assert (
                0.0 <= discount_factor <= 1.0
        ), f"discount factor should be in [0, 1] but got: {discount_factor}"
        self.gamma = discount_factor
        assert (
                estimation_step > 0
        ), f"estimation_step should be greater than 0 but got: {estimation_step}"
        self.n_step = estimation_step
        self._target = target_update_freq > 0
        self.freq = target_update_freq
        self._iter = 0
        self.rew_norm = reward_normalization
        self.is_double = is_double
        self.clip_loss_grad = clip_loss_grad
        self.sum_prob = sum_prob

    def _target_q(self, buffer, indices) -> torch.Tensor:
        obs_next_batch = Batch(
            obs=buffer[indices].obs_next,
            info=buffer[indices].info,
        )  # obs_next: s_{t+n}
        result = self(obs_next_batch)
        if self._target:
            # target_Q = Q_old(s_, argmax(Q_new(s_, *)))
            self.model.active_branch = "model_old"
            target_q = self(obs_next_batch, model="model", input="obs_next").logits
        else:
            target_q = result.logits
        if self.is_double:
            return target_q[np.arange(len(result.act)), result.act]
        # Nature DQN, over estimate
        return target_q.max(dim=1)[0]

    def sync_weight(self) -> None:
        """Synchronize the non-LLM weights for the target network."""
        attributes = ["patch_embedding", "mapping_layer", "reprogramming_layer", "output_projection"]
        for attr in attributes:
            old_attr = f"{attr}_old"
            old_attr_obj = getattr(self.model, old_attr)
            attr_obj = getattr(self.model, attr)
            old_attr_obj.load_state_dict(attr_obj.state_dict())

    def forward(
            self,
            batch: ObsBatchProtocol,
            state: dict | BatchProtocol | np.ndarray | None = None,
            model: Literal["model", "model_old"] = "model",
            **kwargs: Any,
    ) -> ModelOutputBatchProtocol:
        """Compute action over the given batch data and summarize rules."""
        model = getattr(self, model)

        # rules summarization
        obs = batch.obs
        batch_size = len(obs)
        need_summary = random.choices([True, False], weights=[self.sum_prob, 1 - self.sum_prob], k=batch_size)
        conversations = summary_reprogramming(batch)
        conversations_T = [conversations[i] for i in range(batch_size) if need_summary[i]]
        summaries_T = model.summarize(conversations_T, mode='str') if conversations_T!=[] else []
        summaries = ["" for _ in range(batch_size)]
        true_index = 0
        for i in range(batch_size):
            if need_summary[i]:
                summaries[i] = summaries_T[true_index]
                true_index += 1

        # Q value prediction
        series, conversations = q_prompt_reprogramming(obs[:, :, 0], obs[:, :, 1], summaries)
        logits = model.q_pred(series, conversations, mode='Q')
        q = self.compute_q_value(logits, getattr(obs, "mask", None))
        if not hasattr(self, "max_action_num"):
            self.max_action_num = q.shape[1]
        act = to_numpy(q.max(dim=1)[1])

        result = Batch(logits=logits, act=act, state=state)
        return cast(ModelOutputBatchProtocol, result)


class LLM_PPO_Policy(PPOPolicy):
    """
    Implementation of LLM-DQN policy.
    """
    def __init__(
        self,
        *,
        actor: LLMPPO,
        critic: torch.nn.Module,
        optim: torch.optim.Optimizer,
        dist_fn: TDistributionFunction,
        action_space: gym.Space,
        eps_clip: float = 0.2,
        dual_clip: float | None = None,
        value_clip: bool = False,
        advantage_normalization: bool = True,
        recompute_advantage: bool = False,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float | None = None,
        gae_lambda: float = 0.95,
        max_batchsize: int = 256,
        discount_factor: float = 0.99,
        reward_normalization: bool = False,
        deterministic_eval: bool = False,
        observation_space: gym.Space | None = None,
        action_scaling: bool = True,
        action_bound_method: Literal["clip", "tanh"] | None = "clip",
        lr_scheduler: TLearningRateScheduler | None = None,
        sum_prob=0,
    ) -> None:
        assert (
            dual_clip is None or dual_clip > 1.0
        ), f"Dual-clip PPO parameter should greater than 1.0 but got {dual_clip}"

        super().__init__(
            actor=actor,
            critic=critic,
            optim=optim,
            dist_fn=dist_fn,
            action_space=action_space,
            eps_clip=eps_clip,
            dual_clip=dual_clip,
            value_clip=value_clip,
            advantage_normalization=advantage_normalization,
            recompute_advantage=recompute_advantage,
            vf_coef=vf_coef,
            ent_coef=ent_coef,
            max_grad_norm=max_grad_norm,
            gae_lambda=gae_lambda,
            max_batchsize=max_batchsize,
            discount_factor=discount_factor,
            reward_normalization=reward_normalization,
            deterministic_eval=deterministic_eval,
            observation_space=observation_space,
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
            lr_scheduler=lr_scheduler,
        )
        self.sum_prob = sum_prob
    
    def forward(
        self,
        batch: ObsBatchProtocol,
        state: dict | BatchProtocol | np.ndarray | None = None,
        **kwargs: Any,
    ) -> DistBatchProtocol:
        """Compute action over the given batch data by applying the actor.

        Will sample from the dist_fn, if appropriate.
        Returns a new object representing the processed batch data
        (contrary to other methods that modify the input batch inplace).

        .. seealso::

            Please refer to :meth:`~tianshou.policy.BasePolicy.forward` for
            more detailed explanation.
        """
        # rules summarization
        obs = batch.obs
        batch_size = len(obs)
        need_summary = random.choices([True, False], weights=[self.sum_prob, 1 - self.sum_prob], k=batch_size)
        conversations = summary_reprogramming(batch)
        conversations_T = [conversations[i] for i in range(batch_size) if need_summary[i]]
        summaries_T = self.actor.summarize(conversations_T, mode='str') if conversations_T!=[] else []
        summaries = ["" for _ in range(batch_size)]
        true_index = 0
        for i in range(batch_size):
            if need_summary[i]:
                summaries[i] = summaries_T[true_index]
                true_index += 1

        # assumptions about the order of the output and on distribution type
        series, conversations = act_prompt_reprogramming(obs[:, :, 0], obs[:, :, 1], summaries)
        logits = self.actor.act_pred(series, conversations, mode='act')
        if isinstance(logits, tuple):
            dist = self.dist_fn(*logits)
        else:
            dist = self.dist_fn(logits)

        # in this case, the dist is unused!
        if self.deterministic_eval and not self.training:
            act = dist.mode
        else:
            act = dist.sample()
        result = Batch(logits=logits, act=act, state=state, dist=dist)
        return cast(DistBatchProtocol, result)


class LLM_Policy(BasePolicy):
    """
    Implementation of pure LLM policy.
    """

    def __init__(
            self,
            model: torch.nn.Module,
            action_space: gym.Space,
            observation_space: gym.Space | None = None,
    ) -> None:
        super().__init__(
            action_space=action_space,
            observation_space=observation_space,
            action_scaling=False,
            action_bound_method=None,
        )
        self.model = model

    def forward(
            self,
            batch: ObsBatchProtocol,
            state: dict | BatchProtocol | np.ndarray | None = None,
            model: Literal["model", "model_old"] = "model",
            **kwargs: Any,
    ) -> ModelOutputBatchProtocol:
        """Decide action over the given batch data."""
        model = getattr(self, model)

        prompt = obs2text(batch)
        logits = model(prompt)
        act = [text2act(logits)]
        result = Batch(act=act, state=state)
        return cast(ModelOutputBatchProtocol, result)

    def learn(self, batch: RolloutBatchProtocol, *args: Any, **kwargs: Any) -> TTrainingStats:
        raise NotImplementedError("LLM_Policy does not support learning.")
