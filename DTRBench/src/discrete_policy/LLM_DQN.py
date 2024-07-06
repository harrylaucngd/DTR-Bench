from copy import deepcopy
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import time

from tianshou.data import Batch, ReplayBuffer, to_numpy, to_torch_as
from tianshou.policy import BasePolicy, DQNPolicy
from DTRBench.utils.network import LLMNet, get_target_model, sync_target_model

from DTRBench.utils.prompt_pipeline import Conversation, obs_prompt_reprogramming, q_prompt_reprogramming, act_prompt_reprogramming, summary_reprogramming


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
        if self._target:
            # self.model_old = get_target_model(self.model)
            self.model_old = deepcopy(self.model)
            self.model_old.eval()
        self._rew_norm = reward_normalization
        self._is_double = is_double
        self._clip_loss_grad = clip_loss_grad
        self.need_obs_explain = need_obs_explain
        self.need_act_explain = need_act_explain
        self.need_summary = need_summary

    def sync_weight(self) -> None:
        """Synchronize the non-LLM weight for the target network."""
        self.model_old.load_state_dict(self.model.state_dict())
    
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
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        model: str = "model",
        input: str = "obs",
        **kwargs: Any,
    ) -> Batch:
        """
        todo: 1. modify sync_weight
        todo: 2. modify is_double
        """
        model = getattr(self, model)
        if state is None:
            state = Batch(obs=[], act=[], obs_exp=[], act_exp=[], summary=[])
            summ = None
        else:
            state = self.extract_state(state)
            summ = state.summary[-1]

        # obs and obs explanation
        obs = batch[input]
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

        return Batch(logits=logits, act=act, state=state)
