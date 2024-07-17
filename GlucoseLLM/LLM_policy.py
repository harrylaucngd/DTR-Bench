from dataclasses import dataclass
from typing import Any, Generic, Literal, Self, TypeVar, cast, Optional, Union, Sequence

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

from DTRBench.utils.prompt_pipeline import obs_prompt_reprogramming, q_prompt_reprogramming, act_prompt_reprogramming, \
    summary_reprogramming


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
            need_obs_explain=True,
            need_act_explain=True,
            need_summary=True,
            exp_freq=1,
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
        self.state = {}

    def sync_weight(self) -> None:
        """Synchronize the non-LLM weights for the target network."""
        attributes = ["patch_embedding", "mapping_layer", "reprogramming_layer", "output_projection",
                      "normalized_layer"]
        for attr in attributes:
            old_attr = f"{attr}_old"
            old_attr_obj = getattr(self.model, old_attr)
            attr_obj = getattr(self.model, attr)
            old_attr_obj.load_state_dict(attr_obj.state_dict())

    def get_state(self, epi_ids, steps):
        """Get history states for given episodes and steps."""
        states = []
        for ep, st in zip(epi_ids, steps):
            state = {"obs": [], "act": [], "obs_exp": [], "act_exp": [], "summary": []}
            if ep in self.state and st > 0:
                for s in range(st):
                    if s in self.state[ep]:
                        state["obs"].append(self.state[ep][s]["obs"])
                        state["act"].append(self.state[ep][s]["act"])
                        state["obs_exp"].append(self.state[ep][s]["obs_exp"])
                        state["act_exp"].append(self.state[ep][s]["act_exp"])
                        state["summary"].append(self.state[ep][s]["summary"])
            states.append(state)
        return states

    def is_learn(self, epi_ids, steps):
        """Check if the current forward is in the learning process."""
        for ep, st in zip(epi_ids, steps):
            if ep not in self.state:
                return False
            if st not in self.state[ep]:
                return False
        return True

    def insert_state(self, curr_states, epi_ids, steps):
        """Insert current states to self.state if not in learning process."""
        for ep, st, curr_state in zip(epi_ids, steps, curr_states):
            if ep not in self.state:
                assert ep == len(self.state[ep]), "Episode should be the next episode"
                assert st == 0, "Step should be zero in a new episode"
            else:
                assert st == len(self.state[ep]), "Step should be the next step"
            self.state[ep][st] = curr_state

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

        # get state
        epi_ids, steps = batch.info["episode_id"], batch.info["step"]
        assert len(epi_ids)==len(steps), "Inequal lengths of epi_ids and steps!"
        batch_size = len(epi_ids)
        _is_learn = self.is_learn(epi_ids, steps)
        states = self.get_state(epi_ids, steps)
        curr_states, summ = [{"obs": [], "act": [], "obs_exp": [], "act_exp": [], "summary": []} for _ in epi_ids], []
        for i in range(batch_size):
            try:
                summ.append(states[i]["summary"][-1])
            except IndexError:
                summ.append(None)

        # decide to explain or not
        need_obs_explain, need_act_explain, need_summary = [], [], []
        for step in steps:
            if (self.exp_freq == 0) or (step % self.exp_freq != 0):
                need_obs_explain.append(False)
                need_act_explain.append(False)
                need_summary.append(False)
            elif step % self.exp_freq == 0:
                need_obs_explain.append(True)
                need_act_explain.append(True)
                need_summary.append(True)

        # obs and obs explanation
        obs = batch.obs
        obs_next = obs.obs[:, 0] if hasattr(obs, "obs") else obs[:, 0]
        for i in range(batch_size):
            states[i]["obs"] = np.append(states[i]["obs"], obs_next[i])
            curr_states[i]["obs"] = obs_next[i]
            if not _is_learn:
                conversation = obs_prompt_reprogramming(states[i]["obs"], states[i]["act"], states[i]["obs_exp"])
                obs_explain = model.explain_obs(conversation, summ[i], mode='str') if need_obs_explain[i] else ""
                states[i]["obs_exp"] = np.append(states[i]["obs_exp"], obs_explain)
                curr_states[i]["obs_exp"] = obs_explain

        # Q value prediction
        series, conversation = [], []
        for i in range(batch_size):
            ser, con = q_prompt_reprogramming(states[i]["obs"], states[i]["act"], states[i]["obs_exp"],
                                                        states[i]["act_exp"])
            series.append(ser)
            conversation.append(con)
        conversation = torch.stack(series)
        logits = model.q_pred(series, conversation, summ, mode='Q')
        q = self.compute_q_value(logits, getattr(obs, "mask", None))
        if not hasattr(self, "max_action_num"):
            self.max_action_num = q.shape[1]
        act = to_numpy(q.max(dim=1)[1])
        for i in range(batch_size):
            states["act"] = np.append(states[i]["act"], act[i])
            curr_states[i]["act"] = act[i]

        # act explanation and update summary
        for i in range(batch_size):
            if not _is_learn:
                conversation = act_prompt_reprogramming(states[i]["obs"], states[i]["act"], states[i]["act_exp"])
                act_explain = model.explain_act(conversation, summ[i], mode='str') if need_act_explain[i] else ""
                states[i]["act_exp"] = np.append(states[i]["act_exp"], act_explain)
                curr_states[i]["act_exp"] = act_explain

                conversation = summary_reprogramming(states[i]["obs"], states[i]["act"], states[i]["summary"])
                summary = model.summarize(conversation, mode='str') if need_summary[i] else ""
                states[i]["summary"] = np.append(states[i]["summary"], summary)
                curr_states[i]["summary"] = summary

        self.insert_state(curr_states, epi_ids, steps)
        result = Batch(logits=logits, act=act, state=state);print(steps)
        return cast(ModelOutputBatchProtocol, result)

    def learn(self, batch: RolloutBatchProtocol, *args: Any, **kwargs: Any) -> TDQNTrainingStats:
        if self._target and self._iter % self.freq == 0:
            self.sync_weight()
        self.optim.zero_grad()
        weight = batch.pop("weight", 1.0)
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
