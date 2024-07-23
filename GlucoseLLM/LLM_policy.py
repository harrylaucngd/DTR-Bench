from typing import Any, Literal, cast

import re
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
)
from tianshou.policy import BasePolicy, DQNPolicy
from tianshou.policy.base import TLearningRateScheduler, TTrainingStats

from GlucoseLLM.models.llm_net import LLMNet

from GlucoseLLM.prompt_pipeline import obs_prompt_reprogramming, q_prompt_reprogramming, act_prompt_reprogramming, \
    summary_reprogramming


class LLM_DQN_Policy(DQNPolicy):
    """
    Implementation of LLM-DQN policy.
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
            action_space: gym.spaces.Discrete | None = None,
            observation_space: gym.Space | None = None,
            lr_scheduler: TLearningRateScheduler | None = None,
            need_obs_explain=True,
            need_act_explain=True,
            need_summary=True,
            exp_freq=0,
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
        self.need_obs_explain = need_obs_explain
        self.need_act_explain = need_act_explain
        self.need_summary = need_summary
        self.exp_freq = exp_freq
        self.state = {}

    def _target_q(self, buffer, indices) -> torch.Tensor:
        obs_next_batch = Batch(
            obs=buffer[indices].obs_next,
            info=buffer[indices].info,
        )  # obs_next: s_{t+n}
        result = self(obs_next_batch)
        if self._target:
            # target_Q = Q_old(s_, argmax(Q_new(s_, *)))
            target_q = self(obs_next_batch, model="model_old").logits
        else:
            target_q = result.logits
        if self.is_double:
            return target_q[np.arange(len(result.act)), result.act]
        # Nature DQN, over estimate
        return target_q.max(dim=1)[0]

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
                assert st == 0, "Step should be zero in a new episode"
                self.state[ep] = {}
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
        """Compute action over the given batch data and give explanations to the decision."""
        model = getattr(self, model)

        # get state
        epi_ids, steps = batch.info["episode_id"], batch.info["step"]
        assert len(epi_ids) == len(steps), "Inequal lengths of epi_ids and steps!"
        batch_size = len(epi_ids)
        _is_learn = self.is_learn(epi_ids, steps)
        states = self.get_state(epi_ids, steps)
        curr_states, summ = [{"obs": [], "act": [], "obs_exp": [], "act_exp": [], "summary": []} for _ in epi_ids], []
        for i in range(batch_size):
            try:
                summ.append(next((state for state in reversed(states[i]["summary"]) if state != ""), None))
            except IndexError:
                summ.append(None)

        # decide to explain or not
        need_obs_explain, need_act_explain, need_summary = [], [], []
        for step in steps:
            if (_is_learn) or (self.exp_freq == 0) or (step % self.exp_freq != 0):
                need_obs_explain.append(False)
                need_act_explain.append(False)
                need_summary.append(False)
            elif step % self.exp_freq == 0:
                need_obs_explain.append(True)
                need_act_explain.append(True)
                need_summary.append(True)

        # obs and obs explanation
        obs = batch.obs
        obs_next = obs.obs[:, -1, 0] if hasattr(obs, "obs") else obs[:, -1, 0]
        for i in range(batch_size):
            states[i]["obs"] = np.append(states[i]["obs"], obs_next[i])
            curr_states[i]["obs"] = obs_next[i]
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
        series = torch.stack(series)
        logits = model.q_pred(series, conversation, summ, mode='Q')
        q = self.compute_q_value(logits, getattr(obs, "mask", None))
        if not hasattr(self, "max_action_num"):
            self.max_action_num = q.shape[1]
        act = to_numpy(q.max(dim=1)[1])
        for i in range(batch_size):
            states[i]["act"] = np.append(states[i]["act"], act[i])
            curr_states[i]["act"] = act[i]

        # act explanation and update summary
        for i in range(batch_size):
            conversation = act_prompt_reprogramming(states[i]["obs"], states[i]["act"], states[i]["act_exp"])
            act_explain = model.explain_act(conversation, summ[i], mode='str') if need_act_explain[i] else ""
            states[i]["act_exp"] = np.append(states[i]["act_exp"], act_explain)
            curr_states[i]["act_exp"] = act_explain

            conversation = summary_reprogramming(states[i]["obs"], states[i]["act"], states[i]["summary"])
            summary = model.summarize(conversation, mode='str') if need_summary[i] else ""
            states[i]["summary"] = np.append(states[i]["summary"], summary)
            curr_states[i]["summary"] = summary

        if not _is_learn:
            self.insert_state(curr_states, epi_ids, steps)
        result = Batch(logits=logits, act=act, state=state);
        print(step)
        return cast(ModelOutputBatchProtocol, result)


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

    def obs2text(self, batch):
        obs = batch.obs
        length = obs.shape[1]
        time = batch.info["time"]
        glucose = obs[:, :, 0]
        insulin = obs[:, :, 1]

        descriptions = []
        # todo:  check time shift
        for i in range(length):
            if glucose == -1:
                continue
            if i == 0:
                descriptions.append(f"Time:{time - length * 5},insulin:{insulin[0]}. ")
            if i < length - 1:
                descriptions.append(
                    f"Time:{time - (length - i - 1) * 5},glucose:{glucose[i]},insulin:{insulin[i] + 1}. ")
            else:
                descriptions.append("Please determine the current insulin dosage, giving a number in 0-0.5,"
                                    " without anything else. ")
                descriptions.append(f"Current time: {time},glucose:{glucose[i]}, insulin:")
        return " ".join(descriptions)

    def text2act(self, logits):
        numbers = re.findall(r'-?\d+\.?\d*', logits)  # todo: make sure it select the correct number with 0.
        numbers = [float(num) for num in numbers]

        if len(numbers) == 0:
            return self.action_space.sample()

        # always select the first number
        return np.clip(numbers[0], 0, 0.5)

    def forward(
            self,
            batch: ObsBatchProtocol,
            state: dict | BatchProtocol | np.ndarray | None = None,
            model: Literal["model", "model_old"] = "model",
            **kwargs: Any,
    ) -> ModelOutputBatchProtocol:
        """Decide action over the given batch data."""
        model = getattr(self, model)

        prompt = self.obs2text(batch)
        logits = model(prompt)
        act = [self.text2act(logits)]
        result = Batch(act=act, state=state)
        return cast(ModelOutputBatchProtocol, result)

    def learn(self, batch: RolloutBatchProtocol, *args: Any, **kwargs: Any) -> TTrainingStats:
        raise NotImplementedError("LLM_Policy does not support learning.")
