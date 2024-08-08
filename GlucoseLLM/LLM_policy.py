from typing import Any, Literal, cast, List
from tianshou.data import Batch, ReplayBuffer, to_numpy, to_torch_as
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
from tianshou.policy.modelfree.dqn import DQNTrainingStats, TDQNTrainingStats
from tianshou.policy import BasePolicy, DQNPolicy, PPOPolicy
from tianshou.policy.base import TLearningRateScheduler, TTrainingStats
from tianshou.policy.modelfree.pg import TDistributionFunction
from tianshou.utils.net.common import ActorCritic
from GlucoseLLM.model.net import timeLLM
from GlucoseLLM.model.llm_net import LLMPPO
from GlucoseLLM.prompts import (obs2text, SYSTEM_PROMPT, ACTOR_INSTRUCTION_PROMPT, SUMMARY_INSTRUCTION_PROMPT,
                                LLM_INFERENCE_INSTRUCTION_PROMPT, get_Q_instruction, get_patient_info_prompt,
                                LLM_INFERENCE_RETRY_PROMPT)
from GlucoseLLM.prompt_utils import text2act, Conversation


class LLM_DQN_Policy(DQNPolicy):
    def __init__(
            self,
            model: timeLLM,
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
            gradient_accumulation: int = 1,
            summary_prob=0,
    ) -> None:
        """
        gradient_accumulation will divide the loss by the number of accumulation steps, and accumulate the gradient.
        """
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
        self.sum_prob = summary_prob
        self.q_prompt = get_Q_instruction(self.action_space.n, 0.5)
        self.max_action_num = self.action_space.n
        self.gradient_accumulation = gradient_accumulation

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

        batch_size = len(batch)
        need_summary = np.random.choice([True, False], p=[self.sum_prob, 1 - self.sum_prob], size=batch_size)
        if self.sum_prob > 0 and need_summary.any():
            summary = self.text_summarize(batch[need_summary])
            summaries = [summary if need else None for need in need_summary]
        else:
            summaries = [None] * batch_size
        # Q value prediction

        logits, _ = self.get_q(batch, summaries, model)
        q = self.compute_q_value(logits, getattr(batch.obs, "mask", None))
        if not hasattr(self, "max_action_num"):
            self.max_action_num = q.shape[1]
        act = to_numpy(q.max(dim=1)[1])

        result = Batch(logits=logits, act=act, state=state)
        return cast(ModelOutputBatchProtocol, result)

    def get_q(self, batch: ObsBatchProtocol, summaries: List[List[str] | None] | List[None], model) -> torch.Tensor:
        """Compute Q value over the given batch data."""
        prompt_list = []
        ts = torch.from_numpy(batch.obs).bfloat16().to(self.model.device)
        for i in range(len(batch)):
            messages = Conversation()
            messages.insert_component("system", SYSTEM_PROMPT, 0)
            if summaries[i] is not None:
                messages.insert_component("assistant", summaries[i], -1)
            messages.insert_component("user", self.q_prompt, -1)
            prompt_list.append(messages.get())
        logits = self.model.forward(ts, prompt_list, model=model)
        q = self.compute_q_value(logits, getattr(batch.obs, "mask", None))
        return q

    def text_summarize(self, batch: Batch) -> List[str]:
        """Summarize the rules based on the given observation."""
        prompt_list = []
        for i in range(len(batch)):
            obs_prompt = obs2text(batch[i])
            messages = Conversation()
            messages.insert_component("system", SYSTEM_PROMPT, 0)
            messages.insert_component("user", obs_prompt + SUMMARY_INSTRUCTION_PROMPT, -1)
            prompt_list.append(messages.get())
        summary = self.model.generate_text(None, prompt_list)
        return summary

    def learn(self, batch: RolloutBatchProtocol, *args: Any,
              **kwargs: Any) -> TDQNTrainingStats:
        if self._target and self._iter % self.freq == 0:
            self.sync_weight()

        # Calculate the chunk size
        chunk_size = len(batch) // self.gradient_accumulation

        # Initialize overall loss
        total_loss = 0.0

        for i in range(self.gradient_accumulation):
            # Get the current chunk
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size if (i + 1) * chunk_size < len(batch) else len(batch)
            chunk = batch[start_idx:end_idx]

            weight = chunk.pop("weight", 1.0)
            q = self(chunk).logits
            q = q[np.arange(len(q)), chunk.act]
            returns = to_torch_as(chunk.returns.flatten(), q)
            td_error = returns - q

            if self.clip_loss_grad:
                y = q.reshape(-1, 1)
                t = returns.reshape(-1, 1)
                loss = torch.nn.functional.huber_loss(y, t, reduction="mean")
            else:
                loss = (td_error.pow(2) * weight).mean()

            chunk.weight = td_error  # prio-buffer

            # Accumulate loss and divide by accumulation_steps
            total_loss += loss.item()
            loss = loss / self.gradient_accumulation
            loss.backward()

            # Perform optimizer step after accumulating gradients over chunks
            if (i + 1) % self.gradient_accumulation == 0 or (i + 1) == len(batch):
                self.optim.step()
                self.optim.zero_grad()

        self._iter += 1

        return DQNTrainingStats(loss=total_loss / self.gradient_accumulation)  # type: ignore[return-value]


class LLM_PPO_Policy(PPOPolicy):
    """
    Implementation of LLM-DQN policy.
    """

    def __init__(
            self,
            *,
            actor: LLMPPO,
            critic: LLMPPO,
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
        self.eps_clip = eps_clip
        self.dual_clip = dual_clip
        self.value_clip = value_clip
        self.norm_adv = advantage_normalization
        self.recompute_adv = recompute_advantage
        self._actor_critic: ActorCritic

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
        # TODO: rename? It's not really logits and there are particular
        #  assumptions about the order of the output and on distribution type
        logits, hidden = self.actor(batch.obs, state=state, info=batch.info)
        if isinstance(logits, tuple):
            dist = self.dist_fn(*logits)
        else:
            dist = self.dist_fn(logits)

        # in this case, the dist is unused!
        if self.deterministic_eval and not self.training:
            act = dist.mode
        else:
            act = dist.sample()
        result = Batch(logits=logits, act=act, state=hidden, dist=dist)
        return cast(DistBatchProtocol, result)


class LLMInference_Policy(BasePolicy):
    """
    Implementation of pure LLM policy.
    """

    def __init__(
            self,
            model: torch.nn.Module,
            action_space: gym.Space,
            observation_space: gym.Space | None = None,
            need_summary: bool = False,
            num_try: int = 1,
    ) -> None:
        super().__init__(
            action_space=action_space,
            observation_space=observation_space,
            action_scaling=False,
            action_bound_method=None,
        )
        self.model = model
        self.need_summary = need_summary
        self.num_try = num_try
        if num_try < 1:
            raise ValueError("num_try should be greater than 0")

    def forward(
            self,
            batch: ObsBatchProtocol,
            state: dict | BatchProtocol | np.ndarray | None = None,
            model: Literal["model", "model_old"] = "model",
            **kwargs: Any,
    ) -> ModelOutputBatchProtocol:
        """Decide action over the given batch data."""
        if batch.obs.shape[0] != 1:
            raise ValueError("LLMInference_Policy only supports batch size of 1 at inference time.")
        model = getattr(self, model)

        obs_prompt = obs2text(batch[0])
        messages = Conversation()
        messages.insert_component("system", SYSTEM_PROMPT, 0)
        if self.need_summary and (batch.obs[:, :, 0] == -1).mean() < 0.8:
            messages.insert_component("user", obs_prompt + SUMMARY_INSTRUCTION_PROMPT, -1)
            summary = model(messages.get())
            messages.insert_component("assistant", summary, -1)
            messages.insert_component("user", LLM_INFERENCE_INSTRUCTION_PROMPT, -1)
            action_text = model(messages.get())
        else:
            messages.insert_component("user", obs_prompt + LLM_INFERENCE_INSTRUCTION_PROMPT, -1)
            action_text = model(messages.get())

        use_random = True
        for _ in range(self.num_try):
            act = text2act(action_text, self.action_space)
            if act is not None:
                use_random = False
                break
            messages.insert_component("assistant", action_text, -1)
            messages.insert_component("user", LLM_INFERENCE_RETRY_PROMPT, -1)
            action_text = model(messages.get())

        # use random action if no valid action is found
        if use_random:
            act = self.action_space.sample()

        result = Batch(act=[act], state=state)
        # print("policy act", act)
        return cast(ModelOutputBatchProtocol, result)

    def learn(self, batch: RolloutBatchProtocol, *args: Any, **kwargs: Any) -> TTrainingStats:
        raise NotImplementedError("LLM_Policy does not support learning.")
