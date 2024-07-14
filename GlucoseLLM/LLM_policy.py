from dataclasses import dataclass
from typing import Any, Generic, Literal, Self, TypeVar, cast

import numpy as np
import torch

from tianshou.data import Batch, ReplayBuffer, to_numpy, to_torch_as
from tianshou.policy import BasePolicy, DQNPolicy
from tianshou.policy.base import TLearningRateScheduler, TrainingStats
from tianshou.data.batch import BatchProtocol
from tianshou.data.types import (
    BatchWithReturnsProtocol,
    ModelOutputBatchProtocol,
    ObsBatchProtocol,
    RolloutBatchProtocol,
)

from DTRBench.utils.network import LLMNet

from DTRBench.utils.prompt_pipeline import obs_prompt_reprogramming, q_prompt_reprogramming, act_prompt_reprogramming, summary_reprogramming


@dataclass(kw_only=True)
class DQNTrainingStats(TrainingStats):
    loss: float


TDQNTrainingStats = TypeVar("TDQNTrainingStats", bound=DQNTrainingStats)


class LLM_DQN_Policy(DQNPolicy):
    """Implementation of Deep Q Network. arXiv:1312.5602.

    Implementation of Double Q-Learning. arXiv:1509.06461.

    Implementation of Dueling DQN. arXiv:1511.06581 (the dueling DQN is
    implemented in the network side, not here).

    :param torch.nn.Module model: a model following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer optim: a torch.optim for optimizing the model.
    :param float discount_factor: in [0, 1].
    :param int estimation_step: the number of steps to look ahead. Default to 1.
    :param int target_update_freq: the target network update frequency (0 if
        you do not use the target network). Default to 0.
    :param bool reward_normalization: normalize the reward to Normal(0, 1).
        Default to False.
    :param bool is_double: use double dqn. Default to True.
    :param bool clip_loss_grad: clip the gradient of the loss in accordance
        with nature14236; this amounts to using the Huber loss instead of
        the MSE loss. Default to False.
    :param lr_scheduler: a learning rate scheduler that adjusts the learning rate in
        optimizer in each policy.update(). Default to None (no lr_scheduler).

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        model: LLMNet,
        optim: torch.optim.Optimizer,
        discount_factor: float = 0.99,
        estimation_step: int = 1,
        target_update_freq: int = 0,
        reward_normalization: bool = False,
        is_double: bool = True,
        clip_loss_grad: bool = False,
        need_obs_explain = True,
        need_act_explain = True,
        need_summary = True,
        exp_freq = 1,
        **kwargs: Any,
    ) -> None:
        BasePolicy.__init__(self, **kwargs)
        self.model = model
        self.optim = optim
        self.eps = 0.0
        assert 0.0 <= discount_factor <= 1.0, "discount factor should be in [0, 1]"
        self._gamma = discount_factor
        assert estimation_step > 0, "estimation_step should be greater than 0"
        self._n_step = estimation_step
        self._target = target_update_freq > 0
        self._freq = target_update_freq
        self._iter = 0
        self._rew_norm = reward_normalization
        self._is_double = is_double
        self._clip_loss_grad = clip_loss_grad
        self.need_obs_explain = need_obs_explain
        self.need_act_explain = need_act_explain
        self.need_summary = need_summary
        self.exp_freq = exp_freq

    def sync_weight(self) -> None:
        """Synchronize the non-LLM weights for the target network."""
        attributes = ["patch_embedding", "mapping_layer", "reprogramming_layer", "output_projection", "normalized_layer"]
        for attr in attributes:
            old_attr = f"{attr}_old"
            old_attr_obj = getattr(self.model, old_attr)
            attr_obj = getattr(self.model, attr)
            old_attr_obj.load_state_dict(attr_obj.state_dict())

    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        batch = buffer[indices]  # batch.obs_next: s_{t+n}
        result = self(batch, input="obs_next")
        if self._target:
            # target_Q = Q_old(s_, argmax(Q_new(s_, *)))
            self.model.active_branch = "model_old"
            target_q = self(batch, model="model", input="obs_next").logits
        else:
            target_q = result.logits
        if self._is_double:
            return target_q[np.arange(len(result.act)), result.act]
        else:  # Nature DQN, over estimate
            return target_q.max(dim=1)[0]
    
    def compress_state(self, state):
        separator = "|||"
        compressed_state = {
            'obs': [separator.join(map(str, state.obs))],
            'act': [separator.join(map(str, state.act))],
            'obs_exp': [separator.join(map(str, state.obs_exp))],
            'act_exp': [separator.join(map(str, state.act_exp))],
            'summary': [separator.join(map(str, state.summary))],
        }
        return compressed_state

    def extract_state(self, compressed_state):
        separator = "|||"
        extracted_state = Batch(
            obs=compressed_state['obs'][0].split(separator),
            act=compressed_state['act'][0].split(separator),
            obs_exp=compressed_state['obs_exp'][0].split(separator),
            act_exp=compressed_state['act_exp'][0].split(separator),
            summary=compressed_state['summary'][0].split(separator),
        )

        # Convert numerical strings back to their respective types
        extracted_state.obs = [eval(i) if i.replace('.','',1).isdigit() else i for i in extracted_state.obs]
        extracted_state.act = [eval(i) if i.replace('.','',1).isdigit() else i for i in extracted_state.act]

        return extracted_state

    def forward(
        self,
        batch: ObsBatchProtocol,
        state: dict | BatchProtocol | np.ndarray | None = None,
        model: Literal["model", "model_old"] = "model",
        **kwargs: Any,
    ) -> ModelOutputBatchProtocol:
        """
        todo: improve state indices for better locate episode_id and step
        """
        model = getattr(self, model)
        if (state is None) or any(value==[None] for value in state.values()):
            state = Batch(obs=[], act=[], obs_exp=[], act_exp=[], summary=[])
            summ = None
        else:
            state = self.extract_state(state)
            summ = state.summary[-1]

        # decide to explain or not
        step = batch.info["step"]
        attributes = ['need_obs_explain', 'need_act_explain', 'need_summary']
        for attr in attributes:
            explain_bool = getattr(self, attr)
            if (self.exp_freq == 0) or (step % self.exp_freq != 0):
                explain_bool = False
            elif step % self.exp_freq == 0:
                explain_bool = True
            setattr(self, attr, explain_bool)
        
        # obs and obs explanation
        obs = batch.obs
        obs_next = obs.obs if hasattr(obs, "obs") else obs
        state.obs = np.append(state.obs, obs_next)
        conversation = obs_prompt_reprogramming(state.obs, state.act, state.obs_exp)
        obs_explain = model.explain_obs(conversation, summ, mode='str') if self.need_obs_explain else ""
        state.obs_exp = np.append(state.obs_exp, obs_explain)

        # Q value prediction
        series, conversation = q_prompt_reprogramming(state.obs, state.act, state.obs_exp, state.act_exp)
        logits = model.q_pred(series, conversation, summ, mode='Q')
        q = self.compute_q_value(logits, getattr(obs, "mask", None))
        if not hasattr(self, "max_action_num"):
            self.max_action_num = q.shape[1]
        act = to_numpy(q.max(dim=1)[1])
        state.act = np.append(state.act, act)
        
        # act explanation
        conversation = act_prompt_reprogramming(state.obs, state.act, state.act_exp)
        act_explain = model.explain_act(conversation, summ, mode='str') if self.need_act_explain else ""
        state.act_exp = np.append(state.act_exp, act_explain)

        # update summary
        conversation = summary_reprogramming(state.obs, state.act, state.summary)
        summary = model.summarize(conversation, mode='str') if self.need_summary else ""
        state.summary = np.append(state.summary, summary)

        # compress state batch to len 1
        state = self.compress_state(state)

        result = Batch(logits=logits, act=act, state=state);print(step)
        return cast(ModelOutputBatchProtocol, result)

    def learn(self, batch: RolloutBatchProtocol, *args: Any, **kwargs: Any) -> TDQNTrainingStats:
        if self._target and self._iter % self.freq == 0:
            self.sync_weight()
        self.optim.zero_grad()
        weight = batch.pop("weight", 1.0);import pdb;pdb.set_trace()
        q = self(batch).logits
        q = q[np.arange(len(q)), batch.act]
        returns = to_torch_as(batch.returns.flatten(), q)
        td_error = returns - q

        if self.clip_loss_grad:
            y = q.reshape(-1, 1)
            t = returns.reshape(-1, 1)
            loss = torch.nn.functional.huber_loss(y, t, reduction="mean")
        else:
            loss = (td_error.pow(2) * weight).mean()

        batch.weight = td_error  # prio-buffer
        loss.backward()
        self.optim.step()
        self._iter += 1

        return DQNTrainingStats(loss=loss.item())  # type: ignore[return-value]
