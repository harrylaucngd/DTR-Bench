import numpy as np
import argparse
import torch
import torch.nn as nn
from GlucoseLLM.models import GlucoseLLM
from tianshou.utils.net.common import ActorCritic, MLP
from tianshou.utils.net.continuous import Actor as tianshouActor
from typing import Union, List, Tuple, Optional, Callable, Sequence, Dict, Any
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    no_type_check,
)
from transformers import LogitsProcessorList, TopKLogitsWarper, TopPLogitsWarper, TemperatureLogitsWarper, \
    StoppingCriteriaList, MaxLengthCriteria
import torch
from torch import nn
import torch.nn.functional as F
from tianshou.data.batch import Batch

ModuleType = Type[nn.Module]
ArgsType = Union[Tuple[Any, ...], Dict[Any, Any], Sequence[Tuple[Any, ...]],
Sequence[Dict[Any, Any]]]

# todo: move the following llm code to a separate file
llm_context_window = {
    "llama-2-13b": 4096,
    "llama-13b": 2048,
    "llama-3-8b": 4096,
    "llama-2-7b": 4096,
    "llama-7b": 2048,
    "gpt2": 1024
}

obs_exp_prompt = ("The Simglucose environment is a simulation environment designed to mimic the physiological dynamics "
                  "of glucose metabolism in humans, often used in research of glucose control. "
                  "The primary goal in the Simglucose environment is to maintain a patient's blood glucose levels "
                  "(the observation) within a target range through the administration of insulin (the action). "
                  "The reason for a high value observation (high Blood Glucose Level (BG): the current blood glucose "
                  "concentration in mg/dL) is typically in last/last several timestep, more insulin (action) was "
                  "injected, a raising or high level of action is witnessed, and vice versa.")  # expertised system prompt of background knowledge for observation explanation
Q_prompt = ("The Simglucose environment is a simulation environment designed to mimic the physiological dynamics of "
            "glucose metabolism in humans, often used in research of glucose control. The primary goal in the "
            "Simglucose environment is to maintain a patient's blood glucose levels (the observation) within a target "
            "range through the administration of insulin (the action). 5 number of actions (Insulin Bolus Dose) "
            "represents 5 degrees of insulin injection to restrain high blood glucose level. In Q-learning, "
            "the Q-value represents the expected future rewards for taking a given action in a given state, "
            "with high Q-values indicating more favorable actions and low Q-values indicating less favorable actions. "
            "So for a q-learning agent, if the blood glucose level is observed to be high, the q value of the high "
            "value action should be high, and q value of the low value action should be low, and vice versa for low "
            "blood glucose level.")  # expertised system prompt for series information description and Q value prediction
act_exp_prompt = ("The Simglucose environment is a simulation environment designed to mimic the physiological dynamics "
                  "of glucose metabolism in humans, often used in research of glucose control. The primary goal in the "
                  "Simglucose environment is to maintain a patient's blood glucose levels (the observation) within a "
                  "target range through the administration of insulin (the action). The reason for a high value action "
                  "(high Insulin Bolus Dose measured in units (U) of insulin) is typically in current timestep or the "
                  "past several timesteps, a relatively high value of Blood Glucose Level (BG): the current blood "
                  "glucose concentration in mg/dL is observed (low observation), thus the patient needs more insulin "
                  "to prevent the blood glucose from getting too high, and vice versa.")  # expertised system prompt of background knowledge for action explanation


class LLMNet(GlucoseLLM.Model):
    def __init__(
            self,
            configs: argparse.Namespace,
            state_shape: Union[int, Sequence[int]],
            action_shape: Union[int, Sequence[int]] = 0,
            device: Union[str, int, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
            llm: str = "gpt2",
            llm_dim: int = 768,
            need_llm: bool = False,
    ) -> None:
        if isinstance(action_shape, int):
            self.num_actions = action_shape
        elif isinstance(action_shape, Sequence):
            self.num_actions = 1
            for dim in action_shape:
                self.num_actions *= dim
        configs.pred_len = self.num_actions
        configs.seq_len = state_shape  # TODO: The seq_len would be dynamic in our case. How to modify GlucoseLLM accordingly?
        configs.llm_model = llm
        configs.llm_dim = llm_dim
        super().__init__(configs, need_llm=need_llm)
        self.configs = configs
        self.input_shape = state_shape
        self.output_shape = action_shape
        self.device = device
        self.llm = llm

    def forward_Q(self, series, prompt):
        # Inference the whole network
        dec_out = self.forecast(series, prompt)
        return dec_out[:, -self.pred_len:, :].squeeze(-1), []

    def forward_text(self, prompt, max_length=128):
        # todo: remove the following code and use pipeline to infer
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True,
                                max_length=max_length).input_ids.to(self.llm_model.device)
        attention_mask = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True,
                                        max_length=max_length).attention_mask.to(self.llm_model.device)

        # Encode prompt
        outputs = self.llm_model(input_ids=inputs, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        logits = hidden_states[:, -1, :]
        predicted_token_id = torch.argmax(logits, dim=-1).item()
        generated_text = self.tokenizer.decode(predicted_token_id)
        return generated_text

    def forward(self, series, messages, max_length=100, mode='Q'):
        logits, state = None, None
        prompt = self.tokenizer.apply_chat_template(messages.conversation, tokenize=False, add_generation_prompt=True,
                                                    return_tensors="pt")
        if mode == 'Q':
            logits, state = self.forward_Q(series, prompt)
            llm_output = ""
        elif mode == 'str':
            llm_output = self.forward_text(prompt, max_length=max_length)
        else:
            raise ValueError("Unsupported mode! Use 'Q' for full network inference or 'str' for llm_model inference.")
        return logits, state, llm_output

    def freeze_llm_model(self):
        """Ensure all llm_model parameters are frozen."""
        for param in self.llm_model.parameters():
            param.requires_grad = False

    def unfreeze_llm_model(self):
        """Unfreeze all llm_model parameters, allowing them to be updated during training."""
        for param in self.llm_model.parameters():
            param.requires_grad = True

    def explain_obs(self, conversation, mode='str'):
        prompt = conversation.clip(llm_context_window[self.llm] - 300, self.tokenizer)
        prompt.insert_component("system", obs_exp_prompt, 0)
        prompt.insert_component("system", "Please analyze the current state within 100 words:", -1)
        series = torch.tensor([]).to(self.device)
        _, _, response = self.forward(series, prompt, max_length=256, mode=mode)
        return "Analysis of the current state: " + response

    def q_pred(self, series, conversation, mode='Q'):
        prompt = conversation.clip(llm_context_window[self.llm] - 300, self.tokenizer)
        prompt.insert_component("system", Q_prompt, 0)
        prompt.insert_component("system",
                                f"user: Please predict the q value for the {self.num_actions} possible actions in the next timestep:",
                                -1)
        series = torch.tensor(series, dtype=torch.float32).unsqueeze(-1).to(self.device)
        q_list, _, _ = self.forward(series, prompt, max_length=256, mode=mode)
        return q_list

    def explain_act(self, conversation, mode='str'):
        prompt = conversation.clip(llm_context_window[self.llm] - 300, self.tokenizer)
        prompt.insert_component("system", act_exp_prompt, 0)
        prompt.insert_component("system", "Please explain why the agent chose the last action within 100 words:", -1)
        series = torch.tensor([]).to(self.device)
        _, _, response = self.forward(series, prompt, max_length=256, mode=mode)
        return "Explanation of the last action: " + response


def get_target_model(model):
    """
    Initialize the target network according to the q-network.
    Exclude the llm_model part from initialization.
    """
    model_old = LLMNet(configs=model.configs, state_shape=model.input_shape, action_shape=model.output_shape,
                       device="cuda" if torch.cuda.is_available() else "cpu", llm=model.llm, llm_dim=model.d_llm,
                       need_llm=False).to(device="cuda" if torch.cuda.is_available() else "cpu")

    # Copy all layers and parameters from model to model_old except for llm_model
    for name, param in model.named_parameters():
        if 'llm_model' not in name:
            getattr(model_old, name.split('.')[0]).weight = param.data

    # Copy the rest of the attributes excluding llm_model
    excluded_attributes = ['llm_model', 'tokenizer']
    for attr_name in dir(model):
        if not attr_name.startswith('__') and attr_name not in excluded_attributes and not callable(
                getattr(model, attr_name)):
            setattr(model_old, attr_name, getattr(model, attr_name))

    # Set model_old's llm_model to reference model's llm_model
    model_old.llm_model = model.llm_model
    model_old.tokenizer = model.tokenizer

    # Ensure llm_model parameters are not trainable in model_old
    for param in model_old.llm_model.parameters():
        param.requires_grad = False
    return model_old


def sync_target_model(model, model_old):
    """
    Synchronize the parameters of model_old with model.
    Exclude the llm_model part from synchronization.
    """
    model_dict = model.state_dict()
    model_old_dict = model_old.state_dict()

    # Filter out llm_model parameters and update model_old's parameters
    model_dict = {k: v for k, v in model_dict.items() if 'llm_model' not in k}
    model_old_dict.update(model_dict)
    model_old.load_state_dict(model_old_dict)
    return model_old


def define_llm_network(input_shape: int, output_shape: int,
                       device="cuda" if torch.cuda.is_available() else "cpu", llm="gpt2", llm_dim=768
                       ):
    configs = argparse.Namespace(
        d_ff=32,
        patch_len=9,  # TODO: Adaptive value?
        stride=8,  # TODO: Adaptive value?
        llm_layers=6,
        d_model=16,
        dropout=0.1,
        n_heads=8,
        enc_in=7,
        prompt_domain=0,
        content="",
    )
    net = LLMNet(configs=configs, state_shape=input_shape, action_shape=output_shape,
                 device=device, llm=llm, llm_dim=llm_dim, need_llm=True).to(device)
    return net


class Net(nn.Module):
    def __init__(
            self,
            state_shape: Union[int, Sequence[int]],
            action_shape: Union[int, Sequence[int]] = 0,
            hidden_sizes: Sequence[int] = (),
            norm_layer: Optional[Union[ModuleType, Sequence[ModuleType]]] = None,
            norm_args: Optional[ArgsType] = None,
            activation: Optional[Union[ModuleType, Sequence[ModuleType]]] = nn.ReLU,
            act_args: Optional[ArgsType] = None,
            device: Union[str, int, torch.device] = "cpu",
            softmax: bool = False,
            concat: bool = False,
            num_atoms: int = 1,
            dueling_param: Optional[Tuple[Dict[str, Any], Dict[str, Any]]] = None,
            linear_layer: Type[nn.Linear] = nn.Linear,
            cat_num: int = 1,
    ) -> None:
        super().__init__()
        self.device = device
        self.softmax = softmax
        self.num_atoms = num_atoms
        self.cat_num = cat_num
        input_dim = int(np.prod(state_shape)) * cat_num
        action_dim = int(np.prod(action_shape)) * num_atoms
        if concat:
            input_dim += action_dim
        self.use_dueling = dueling_param is not None
        output_dim = action_dim if not self.use_dueling and not concat else 0
        self.model = MLP(
            input_dim, output_dim, hidden_sizes, norm_layer, norm_args, activation,
            act_args, device, linear_layer
        )
        self.output_dim = self.model.output_dim
        if self.use_dueling:  # dueling DQN
            q_kwargs, v_kwargs = dueling_param  # type: ignore
            q_output_dim, v_output_dim = 0, 0
            if not concat:
                q_output_dim, v_output_dim = action_dim, num_atoms
            q_kwargs: Dict[str, Any] = {
                **q_kwargs, "input_dim": self.output_dim,
                "output_dim": q_output_dim,
                "device": self.device
            }
            v_kwargs: Dict[str, Any] = {
                **v_kwargs, "input_dim": self.output_dim,
                "output_dim": v_output_dim,
                "device": self.device
            }
            self.Q, self.V = MLP(**q_kwargs), MLP(**v_kwargs)
            self.output_dim = self.Q.output_dim

    def forward(
            self,
            obs: Union[np.ndarray, torch.Tensor],
            state: Any = None,
            info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        """Mapping: obs -> flatten (inside MLP)-> logits."""
        obs = torch.as_tensor(
            obs,
            device=self.device,
            dtype=torch.float32,
        )
        if obs.ndim == 3:
            obs = obs.view(obs.shape[0], -1)  # cat. ATTENTION: this is a temporary solution.
        logits = self.model(obs)
        bsz = logits.shape[0]
        if self.use_dueling:  # Dueling DQN
            q, v = self.Q(logits), self.V(logits)
            if self.num_atoms > 1:
                q = q.view(bsz, -1, self.num_atoms)
                v = v.view(bsz, -1, self.num_atoms)
            logits = q - q.mean(dim=1, keepdim=True) + v
        elif self.num_atoms > 1:
            logits = logits.view(bsz, -1, self.num_atoms)
        if self.softmax:
            logits = torch.softmax(logits, dim=-1)
        return logits, state


class Recurrent(nn.Module):
    def __init__(
            self,
            layer_num: int,
            state_shape: Union[int, Sequence[int]],
            action_shape: Union[int, Sequence[int]],
            device: Union[str, int, torch.device] = "cpu",
            hidden_layer_size: int = 128,
            dropout: float = 0.0,
            num_atoms: int = 1,
            last_step_only: bool = True,
            ignore_state: bool = True,
    ) -> None:
        super().__init__()
        self.device = device
        self.nn = nn.LSTM(
            input_size=hidden_layer_size,
            hidden_size=hidden_layer_size,
            num_layers=layer_num,
            dropout=dropout,
            batch_first=True,
        )
        self.num_atoms = num_atoms
        self.action_dim = int(np.prod(action_shape)) * num_atoms
        self.fc1 = nn.Linear(int(np.prod(state_shape)), hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, self.action_dim)
        self.use_last_step = last_step_only
        self.output_dim = self.action_dim
        self.ignore_state = ignore_state  # whether to ignore the state input. We already set rnn style obs in env, so state should be ignored.

    def forward(
            self,
            obs: Union[np.ndarray, torch.Tensor],
            state: Optional[Dict[str, torch.Tensor]] = None,
            info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        obs = torch.as_tensor(
            obs,
            device=self.device,
            dtype=torch.float32,
        )
        # obs [bsz, len, dim] (training) or [bsz, dim] (evaluation)
        # In short, the tensor's shape in training phase is longer than which
        # in evaluation phase.
        if len(obs.shape) == 2:
            obs = obs.unsqueeze(-2)
        obs = self.fc1(obs)
        self.nn.flatten_parameters()
        if state is None or self.ignore_state:
            obs, (hidden, cell) = self.nn(obs)
        else:
            # we store the stack data in [bsz, len, ...] format
            # but pytorch rnn needs [len, bsz, ...]
            obs, (hidden, cell) = self.nn(
                obs, (
                    state["hidden"].transpose(0, 1).contiguous(),
                    state["cell"].transpose(0, 1).contiguous()
                )
            )
        if self.use_last_step:
            obs = self.fc2(obs[:, -1])
        else:
            obs = self.fc2(obs)

        if self.num_atoms > 1:
            obs = obs.view(obs.shape[0], -1, self.num_atoms)

        return obs, {
            "hidden": hidden.transpose(0, 1).detach(),
            "cell": cell.transpose(0, 1).detach()
        }


# class Actor(nn.Module):
#     def __init__(self, preprocess_net, min_action=-1, max_action=1, activation="Tanh"):
#         super(Actor, self).__init__()
#         self.preprocess_net = preprocess_net
#         self.activation = getattr(nn, activation)()
#         self.device = self.preprocess_net.device
#         self.min_action = min_action
#         self.max_action = max_action
#
#     def forward(self, obs, state=None, info={}):
#         obs = torch.as_tensor(
#             obs,
#             device=self.device,
#             dtype=torch.float32,
#         )
#         obs, state = self.preprocess_net(obs, state)
#         action = self.activation(obs)
#         action = self.min_action + ((action + 1) * (self.max_action - self.min_action) / 2)
#         return action, state

class Actor(tianshouActor):
    """Simple actor network.

    It will create an actor operated in continuous action space with structure of preprocess_net ---> action_shape.

    :param preprocess_net: a self-defined preprocess_net which output a
        flattened hidden state.
    :param action_shape: a sequence of int for the shape of action.
    :param hidden_sizes: a sequence of int for constructing the MLP after
        preprocess_net. Default to empty sequence (where the MLP now contains
        only a single linear layer).
    :param max_action: the scale for the final action logits. Default to
        1.
    :param preprocess_net_output_dim: the output dimension of
        preprocess_net.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.

    .. seealso::

        Please refer to :class:`~tianshou.utils.net.common.Net` as an instance
        of how preprocess_net is suggested to be defined.
    """

    def __init__(
            self,
            preprocess_net: nn.Module,
            action_shape,
            hidden_sizes: Sequence[int] = (),
            max_action: float = 1.0,
            device: str | int | torch.device = "cpu",
            preprocess_net_output_dim: int | None = None,
            final_activation: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
            last_layer_init_scale: float = 1.
    ) -> None:
        super().__init__(preprocess_net, action_shape, hidden_sizes, max_action, device, preprocess_net_output_dim)
        self.activation = final_activation
        self.last_layer_init_scale = last_layer_init_scale

        # last layer init rescale
        if self.last_layer_init_scale != 1.:
            # todo: should fix this
            self.last.weight.data.copy_(self.last_layer_init_scale * self.last.weight.data)

    def forward(
            self,
            obs: np.ndarray | torch.Tensor,
            state: Any = None,
            info: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, Any]:
        """Mapping: obs -> logits -> action."""
        if info is None:
            info = {}
        logits, hidden = self.preprocess(obs, state)
        logits = self.last(logits)
        if self.activation is not None:
            logits = self.activation(logits)
        logits = self.max_action * logits
        return logits, hidden


class Critic(nn.Module):

    def __init__(
            self,
            state_net: nn.Module,
            action_net: nn.Module = None,
            cat_size: int = 256,
            fuse_hidden_sizes: Sequence[int] = (256,),
            device: str | int | torch.device = "cpu",
    ) -> None:
        super().__init__()

        self.obs_net = state_net
        self.act_net = action_net
        self.device = device
        self.output_dim = 1
        self.last = MLP(
            cat_size,
            1,
            fuse_hidden_sizes,
            device=self.device,
            linear_layer=nn.Linear,
            flatten_input=True,
        )

    def forward(
            self,
            obs: np.ndarray | torch.Tensor,
            act: np.ndarray | torch.Tensor = None,
            state: Optional[Dict[str, torch.Tensor]] = None,
            info: dict[str, Any] | None = {},
    ):
        """Mapping: (s, a) -> logits -> Q(s, a)."""
        obs, state = self.obs_net(obs, state)  # state here won't be useful anyway since tianshou does not RNN-critic
        if act is not None:
            act, _ = self.act_net(act)
            obs = torch.cat([obs, act], dim=1)
        else:
            assert self.act_net is None
        value = self.last(obs)
        return value


def define_single_network(input_shape: int, output_shape: int, hidden_size=256, num_layer=4,
                          use_rnn=False, use_dueling=False, cat_num: int = 1, linear=False,
                          device="cuda" if torch.cuda.is_available() else "cpu"
                          ):
    assert num_layer > 1 or linear
    if use_dueling and use_rnn:
        raise NotImplementedError("rnn and dueling are not implemented together")

    if use_dueling:
        if linear:
            dueling_params = ({"hidden_sizes": (), "activation": None},
                              {"hidden_sizes": (), "activation": None})
        else:
            dueling_params = ({"hidden_sizes": (hidden_size, hidden_size), "activation": nn.ReLU},
                              {"hidden_sizes": (hidden_size, hidden_size), "activation": nn.ReLU})
    else:
        dueling_params = None
    if not use_rnn:
        net = Net(state_shape=input_shape, action_shape=output_shape,
                  hidden_sizes=(hidden_size,) * num_layer if not linear else (),
                  activation=nn.ReLU if not linear else None,
                  device=device, dueling_param=dueling_params, cat_num=cat_num).to(device)
    else:
        net = Recurrent(layer_num=num_layer - 1,
                        state_shape=input_shape,
                        action_shape=output_shape,
                        device=device,
                        hidden_layer_size=hidden_size,
                        ).to(device)

    return net


def define_continuous_critic(state_shape: int, action_shape,
                             state_net_n_layer=2,
                             state_net_hidden_size=128,
                             action_net_n_layer=1,
                             action_net_hidden_size=128,
                             fuse_net_n_layer=1,
                             linear=False,
                             use_rnn=False,
                             cat_num=1, use_action_net=True,
                             device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Since Tianshou's critic network does not support RNN style network, we use a simple MLP network here.
    """
    if use_rnn:
        obs_net = Recurrent(layer_num=state_net_n_layer,
                            state_shape=state_shape,
                            action_shape=state_net_hidden_size,
                            device=device,
                            hidden_layer_size=state_net_hidden_size,
                            ).to(device)
    else:
        obs_net = Net(state_shape=state_shape, action_shape=state_net_hidden_size,
                      hidden_sizes=(state_net_hidden_size,) * state_net_n_layer if not linear else (),
                      activation=nn.ReLU if not linear else None,
                      device=device, dueling_param=None, cat_num=cat_num).to(device)
    if use_action_net:
        act_net = Net(state_shape=action_shape, action_shape=action_net_hidden_size,
                      hidden_sizes=action_net_n_layer * [action_net_hidden_size],
                      activation=nn.ReLU,
                      device=device, cat_num=1).to(device)
        critic = Critic(obs_net, act_net, cat_size=state_net_hidden_size + action_net_hidden_size,
                        fuse_hidden_sizes=[state_net_hidden_size + action_net_hidden_size] * fuse_net_n_layer,
                        device=device).to(device)

        return critic
    else:
        critic = Critic(obs_net, None, cat_size=state_net_hidden_size,
                        fuse_hidden_sizes=[state_net_hidden_size] * fuse_net_n_layer,
                        device=device).to(device)
        return critic
