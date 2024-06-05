from copy import deepcopy
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from tianshou.data import Batch, ReplayBuffer, to_numpy, to_torch_as
from tianshou.policy import DQNPolicy
from DTRBench.utils.network import LLMNet, get_target_model, sync_target_model

from DTRBench.utils.prompt_pipeline import Conversation, act_prompt_reprogramming, obs_prompt_reprogramming, q_prompt_reprogramming


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
        need_act_explain = True,
        need_obs_explain = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, optim, discount_factor, estimation_step, target_update_freq, reward_normalization, is_double, clip_loss_grad, **kwargs)
        if self._target:
            #self.model_old = get_target_model(self.model)
            self.model_old = deepcopy(self.model)
            self.model_old.eval()
        self.need_act_explain = need_act_explain
        self.need_obs_explain = need_obs_explain

    def concat_batches(self, batch1, batch2):
        obs_concat = np.append(batch1.obs, batch2.obs)
        act_concat = np.append(batch1.act, batch2.act)
        act_exp_concat = np.append(batch1.act_exp, batch2.act_exp)
        obs_exp_concat = np.append(batch1.obs_exp, batch2.obs_exp)
        return Batch(obs=obs_concat, act=act_concat, act_exp=act_exp_concat, obs_exp=obs_exp_concat)
    
    def align_state(self, state):
        max_len = max(len(state.obs), len(state.act), len(state.act_exp), len(state.obs_exp))

        def pad_list(lst, pad_value, max_len):
            return np.append(lst, [pad_value] * (max_len - len(lst)))
        
        state.obs = pad_list(state.obs, -np.inf, max_len)
        state.act = pad_list(state.act, -np.inf, max_len)
        state.act_exp = pad_list(state.act_exp, "N/A", max_len)
        state.obs_exp = pad_list(state.obs_exp, "N/A", max_len)
        return state

    def extract_state(self, state):
        def trim_list(lst, trim_value):
            return [x for x in lst if x != trim_value]

        state.obs = trim_list(state.obs, -np.inf)
        state.act = trim_list(state.act, -np.inf)
        state.act_exp = trim_list(state.act_exp, "N/A")
        state.obs_exp = trim_list(state.obs_exp, "N/A")
        return state

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        model: str = "model",
        input: str = "obs",
        **kwargs: Any,
    ) -> Batch:
        """
        todo: 0. modify env function, a) sample_time=10 b) action_list append in step and reset in self.reset
        todo: 1. modify forward
        todo: 2. modify sync_weight
        todo: 3. modify is_double
        todo: 4. change network definition
        todo: 5. in network, add functions to control which mode to use
        todo: 6. transformers.pipeline   [{"role": "system", "content": "blablabla"},
                                          {"role": "user", "content": "blabla"},
                                          {"role": "assistant", "content": "blabla"}]
                                                https://huggingface.co/docs/transformers/main/en/chat_templating
        todo: 7.https://tianshou.org/en/v0.4.7/tutorials/batch.html
        """
        # todo: when state is None, it is the first step of an episode, so you can initialize the state and prompt using state is None
        # TODO: How to play with batch.info? batch.info has action history. You can assume prev_act_list = batch.info["prev_act_list"]
        model = getattr(self, model)
        if state is None:
            state = Batch(obs=[], act=[], act_exp=[], obs_exp=[])
            is_first_round = True
        else:
            for attribute in state.keys():
                state[attribute] = state[attribute][0]
            state = self.extract_state(state)
            is_first_round = False

        curr_state = Batch(obs=[], act=[], act_exp=[], obs_exp=[])
        act_exp_prompt = "" # TODO: expertised background knowledge for action explanation
        obs_exp_prompt = "" # TODO: expertised background knowledge for observation explanation
        Q_prompt = ""       # TODO: expertised prompt for series information description and Q value prediction
        if is_first_round:    # first round, no explanation needed
            obs = batch[input]
            obs_next = obs.obs if hasattr(obs, "obs") else obs
            curr_state.obs = np.append(curr_state.obs, obs_next)
            conversation = Conversation()
            conversation.add_component("user", Q_prompt)
            logits = model.q_pred(obs_next, conversation, mode='Q')

            q = self.compute_q_value(logits, getattr(obs, "mask", None))
            if not hasattr(self, "max_action_num"):
                self.max_action_num = q.shape[1]
            act = to_numpy(q.max(dim=1)[1])
        else:
            # act explanation
            curr_state.act = np.append(curr_state.act, batch.act[-1]) # add last step act from info outside forward
            conversation = act_prompt_reprogramming(state.obs, np.append(state.act, curr_state.act), state.act_exp)
            conversation.append_description(act_exp_prompt)
            act_explain = model.explain_act(conversation, mode='str') if self.need_act_explain else ""
            curr_state.act_exp = np.append(curr_state.act_exp, act_explain)

            # obs explanation
            obs = batch[input]
            obs_next = obs.obs if hasattr(obs, "obs") else obs
            curr_state.obs = np.append(curr_state.obs, obs_next)
            conversation = obs_prompt_reprogramming(np.append(state.obs, curr_state.obs), np.append(state.act, curr_state.act), state.act_exp)
            conversation.append_description(obs_exp_prompt)
            obs_explain = model.explain_obs(conversation, mode='str') if self.need_obs_explain else ""
            curr_state.obs_exp = np.append(curr_state.obs_exp, obs_explain)

            # Q value prediction
            series, conversation = q_prompt_reprogramming(np.append(state.obs, curr_state.obs), np.append(state.act, curr_state.act), act_explain, obs_explain)
            conversation.append_description(Q_prompt)
            logits = model.q_pred(series, conversation, mode='Q')
            q = self.compute_q_value(logits, getattr(obs, "mask", None))
            if not hasattr(self, "max_action_num"):
                self.max_action_num = q.shape[1]
            act = to_numpy(q.max(dim=1)[1])

        # append curr_state to state history and align lengths of each attributes
        state = self.align_state(self.concat_batches(state, curr_state))
        for attribute in state.keys():
            state[attribute] = [state[attribute]]

        return Batch(logits=logits, act=act, state=state)

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        if self._target and self._iter % self._freq == 0:
            #self.model_old = sync_target_model(self.model, self.model_old)
            self.sync_weight()
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