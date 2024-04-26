from copy import deepcopy
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from tianshou.data import Batch, ReplayBuffer, to_numpy, to_torch_as
from tianshou.policy import DQNPolicy

from DTRBench.utils.prompt_info import min_values, max_values, median, lags


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
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        discount_factor: float = 0.99,
        estimation_step: int = 1,
        target_update_freq: int = 0,
        reward_normalization: bool = False,
        is_double: bool = True,
        clip_loss_grad: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
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
            self.model_old = deepcopy(self.model)
            self.model_old.eval()
        self._rew_norm = reward_normalization
        self._is_double = is_double
        self._clip_loss_grad = clip_loss_grad

    def set_eps(self, eps: float) -> None:
        """Set the eps for epsilon-greedy exploration."""
        self.eps = eps

    def train(self, mode: bool = True) -> "LLM_DQN_Policy":
        """Set the module in training mode, except for the target network."""
        self.training = mode
        self.model.train(mode)
        return self

    def sync_weight(self) -> None:
        """Synchronize the weight for the target network."""
        self.model_old.load_state_dict(self.model.state_dict())

    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        batch = buffer[indices]  # batch.obs_next: s_{t+n}
        result = self(batch, input="obs_next")
        if self._target:
            # target_Q = Q_old(s_, argmax(Q_new(s_, *)))
            target_q = self(batch, model="model_old", input="obs_next").logits
        else:
            target_q = result.logits
        if self._is_double:
            return target_q[np.arange(len(result.act)), result.act]
        else:  # Nature DQN, over estimate
            return target_q.max(dim=1)[0]

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        """Compute the n-step return for Q-learning targets.

        More details can be found at
        :meth:`~tianshou.policy.BasePolicy.compute_nstep_return`.
        """
        batch = self.compute_nstep_return(
            batch, buffer, indices, self._target_q, self._gamma, self._n_step,
            self._rew_norm
        )
        return batch

    def compute_q_value(
        self, logits: torch.Tensor, mask: Optional[np.ndarray]
    ) -> torch.Tensor:
        """Compute the q value based on the network's raw output and action mask."""
        if mask is not None:
            # the masked q value should be smaller than logits.min()
            min_value = logits.min() - logits.max() - 1.0
            logits = logits + to_torch_as(1 - mask, logits) * min_value
        return logits

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        model: str = "model",
        input: str = "obs",
        **kwargs: Any,
    ) -> Batch:
        """
        todo: 1. modify forward
        todo: 2. modify sync_weight
        todo: 3. modify is_double
        todo: 4. change network definition
        todo: 5. in network, add functions to control which mode to use
        todo: 6. transformers.pipeline   [{"role": "system", "content": "blablabla"},
                                          {"role": "user", "content": "blabla"},
                                          {"role": "assistent", "content": "blabla"}]
                                                https://huggingface.co/docs/transformers/main/en/chat_templating
        todo: 7.https://tianshou.org/en/v0.4.7/tutorials/batch.html

        Define Q generation as self(prompt, obs, mode=Q), define string generation as self(prompt, obs, mode=str)
        1. build prompt
        system_prompt = blablabla
        history_prompt = state

        1. observation explanation
        if need_obs_explain:
            obs_explain = model.explain_obs(system_prompt, obs, mode=str)
        else:
            obs_explain = []
        new_prompt = system_prompt + history_prompt + obs_explain    #add obs_explain to prompt

        2. action explanation
        if need_act_explain:
            act_exp_prompt = new_prompt + self.take_action_from_info(state) + question_prompt
            act_explain = model.explain_act(system_prompt, act, mode=str)
        else:
            act_explain = []

        3. Q inference
        logits, state = model(new_prompt, obs, mode=Q)


        4. save obs_explain, act_explain, to state

        """
        # todo: state stores all previous histories, so use state
        # todo: all prompts go to a prompt bank

        obs_prompt, action_prompt = [], []  # TODO:Where to get/store history? (Using Collector? Like NStepReturn? Or storing it locally with unique identifier?)
        model = getattr(self, model)
        obs = batch[input]
        obs_next = obs.obs if hasattr(obs, "obs") else obs

        # Generate observation explanation
        min_values_str = str(min_values(obs_prompt))
        max_values_str = str(max_values(obs_prompt))
        median_values_str = str(median(obs_prompt))
        lags_values_str = str(lags(obs_prompt))
        prompt_ = (
            f"<|start_prompt|>Data description: {self.state_description}"
            f"Task description: forecast the next {str(self.output_size)} states given the previous {str(self.input_size)} steps information; "
            "Input statistics: "
            f"min value {min_values_str}, "
            f"max value {max_values_str}, "
            f"median value {median_values_str}, "
            f"the trend of input is {'upward' if obs - obs_prompt[-1].obs > 0 else 'downward'}, "
            f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
        )
        obs_prompt.append(
            {"prompt": prompt_,
             "min_values": min(obs_next),
             "max_values": max(obs_next),
             "median_values": median(obs_next),
             "lags_values": lags(obs_next)}
        )
        obs_explain = model.explain_obs(obs_prompt, obs_next)

        # Compute logits and gain action
        logits, state = model(obs_next, state=state, info=batch.info)
        q = self.compute_q_value(logits, getattr(obs, "mask", None))
        if not hasattr(self, "max_action_num"):
            self.max_action_num = q.shape[1]
        act = to_numpy(q.max(dim=1)[1])

        # Generate action explanation
        min_values_str = str(min_values(obs_prompt))
        max_values_str = str(max_values(obs_prompt))
        median_values_str = str(median(obs_prompt))
        lags_values_str = str(lags(obs_prompt))
        prompt_ = (
            f"<|start_prompt|>Data description: {self.action_description}"
            f"Task description: forecast the next {str(self.output_size)} action given the previous {str(self.input_size)} steps information; "
            "Input statistics: "
            f"min value {min_values_str}, "
            f"max value {max_values_str}, "
            f"median value {median_values_str}, "
            f"the trend of input is {'upward' if act - action_prompt[-1].act > 0 else 'downward'}, "
            f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
        )
        action_prompt.append(
            {"prompt": prompt_,
             "min_values": min(act),
             "max_values": max(act),
             "median_values": median(act),
             "lags_values": lags(act)}
        )
        action_explain = model.explain_action(action_prompt, act)

        # Construct new s dict
        if isinstance(obs, Batch):
            s = Batch(obs=obs, obs_explain=obs_explain, action_explain=action_explain)
        else:
            s = {'obs': obs, 'obs_explain': obs_explain, 'action_explain': action_explain}
        return Batch(logits=logits, act=act, state=s)

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        if self._target and self._iter % self._freq == 0:
            self.sync_weight() # todo: only sync the Q module
        self.optim.zero_grad()
        weight = batch.pop("weight", 1.0)
        q = self(batch).logits
        q = q[np.arange(len(q)), batch.act]
        returns = to_torch_as(batch.returns.flatten(), q)
        td_error = returns - q

        if self._clip_loss_grad:
            y = q.reshape(-1, 1)
            t = returns.reshape(-1, 1)
            loss = torch.nn.functional.huber_loss(y, t, reduction="mean")
        else:
            loss = (td_error.pow(2) * weight).mean()

        batch.weight = td_error  # prio-buffer
        loss.backward()
        self.optim.step()
        self._iter += 1
        return {"loss": loss.item()}

    def exploration_noise(
        self,
        act: Union[np.ndarray, Batch],
        batch: Batch,
    ) -> Union[np.ndarray, Batch]:
        if isinstance(act, np.ndarray) and not np.isclose(self.eps, 0.0):
            bsz = len(act)
            rand_mask = np.random.rand(bsz) < self.eps
            q = np.random.rand(bsz, self.max_action_num)  # [0, 1]
            if hasattr(batch.obs, "mask"):
                q += batch.obs.mask
            rand_act = q.argmax(axis=1)
            act[rand_mask] = rand_act[rand_mask]
        return act