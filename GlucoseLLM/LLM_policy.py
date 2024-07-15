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
        self.episode = -1

    def sync_weight(self) -> None:
        """Synchronize the non-LLM weights for the target network."""
        attributes = ["patch_embedding", "mapping_layer", "reprogramming_layer", "output_projection",
                      "normalized_layer"]
        for attr in attributes:
            old_attr = f"{attr}_old"
            old_attr_obj = getattr(self.model, old_attr)
            attr_obj = getattr(self.model, attr)
            old_attr_obj.load_state_dict(attr_obj.state_dict())

    def get_state(self, ep_ids: Union[Sequence[int]], steps: Union[Sequence[int]]):
        if isinstance(ep_ids, int):
            assert isinstance(steps, int), "step should be an integer when episode is an integer"
            ep_ids = [ep_ids]
            steps = [steps]
        else:
            assert isinstance(steps, Sequence), "step should be a list when episode is a list"
            assert len(ep_ids) == len(steps), "episode and step should have the same length"

        states = Batch(**{"obs": [], "act": [], "obs_exp": [], "act_exp": [], "summary": []})

        if self.state == {}:
            return states

        for ep_id, step in zip(ep_ids, steps):
            ep_data = self.state.get(ep_id, states)  #todo: may need to skip step=0
            states["obs"].append(ep_data[:step].get("obs", None))
            states["act"].append(ep_data[:step].get("act", None))
            states["obs_exp"].append(ep_data[:step].get("obs_exp", None))
            states["act_exp"].append(ep_data[:step].get("act_exp", None))
            states["summary"].append(ep_data[:step].get("summary", None))
        return states

    def is_learn(self, episode: Union[int, Sequence[int]], step: Union[int, Sequence[int]]):
        if isinstance(episode, int):
            assert isinstance(step, int), "step should be an integer when episode is an integer"
            episode = [episode]
            step = [step]
        else:
            assert isinstance(step, Sequence), "step should be a list when episode is a list"
            assert len(episode) == len(step), "episode and step should have the same length"

        for ep, st in zip(episode, step):
            if ep not in self.state:
                return False
            if st not in self.state[ep]:
                return False
        return True

    def insert_state(self, curr_state: Batch, episode: int, step: int):
        if episode not in self.state:
            self.state[episode] = curr_state
        else:
            assert step == len(self.state[episode]), "step should be the next step"
            self.state[episode] = Batch.cat([self.state[episode], curr_state])

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
        states = self.get_state(epi_ids)
        curr_state = {}
        summ = None if states["summary"] == [] else states["summary"][-1]

        # decide to explain or not
        step = batch.info["step"][0]
        if step == 0:
            self.episode += 1
            states = {"obs": [], "act": [], "obs_exp": [], "act_exp": [], "summary": []}
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
        states["obs"] = np.append(states["obs"], obs_next[0][0])
        curr_state["obs"] = obs_next[0][0]
        conversation = obs_prompt_reprogramming(states["obs"], states["act"], states["obs_exp"])
        obs_explain = model.explain_obs(conversation, summ, mode='str') if self.need_obs_explain else ""
        states["obs_exp"] = np.append(states["obs_exp"], obs_explain)
        curr_state["obs_exp"] = obs_explain

        # Q value prediction
        series, conversation = q_prompt_reprogramming(states["obs"], states["act"], states["obs_exp"],
                                                      states["act_exp"])
        logits = model.q_pred(series, conversation, summ, mode='Q')
        q = self.compute_q_value(logits, getattr(obs, "mask", None))
        if not hasattr(self, "max_action_num"):
            self.max_action_num = q.shape[1]
        act = to_numpy(q.max(dim=1)[1])
        states["act"] = np.append(states["act"], act[0])
        curr_state["act"] = act[0]

        # act explanation
        conversation = act_prompt_reprogramming(states["obs"], states["act"], states["act_exp"])
        act_explain = model.explain_act(conversation, summ, mode='str') if self.need_act_explain else ""
        states["act_exp"] = np.append(states["act_exp"], act_explain)
        curr_state["act_exp"] = act_explain

        # update summary
        conversation = summary_reprogramming(states["obs"], states["act"], states["summary"])
        summary = model.summarize(conversation, mode='str') if self.need_summary else ""
        states["summary"] = np.append(states["summary"], summary)
        curr_state["summary"] = summary

        self.insert_state(Batch(**curr_state), self.episode, step)
        result = Batch(logits=logits, act=act, state=None)
        print(step)
        return cast(ModelOutputBatchProtocol, result)

    def learn(self, batch: RolloutBatchProtocol, *args: Any, **kwargs: Any) -> TDQNTrainingStats:
        if self._target and self._iter % self.freq == 0:
            self.sync_weight()
        self.optim.zero_grad()
        weight = batch.pop("weight", 1.0);
        import pdb;
        pdb.set_trace()
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
