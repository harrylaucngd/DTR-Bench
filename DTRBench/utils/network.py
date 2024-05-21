import numpy as np
import argparse
import torch
import torch.nn as nn
from GlucoseLLM.models import GlucoseLLM
from tianshou.utils.net.common import ActorCritic, MLP
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
from transformers import pipeline
import torch
from torch import nn
import torch.nn.functional as F
from tianshou.data.batch import Batch
from transformers import pipelines

ModuleType = Type[nn.Module]
ArgsType = Union[Tuple[Any, ...], Dict[Any, Any], Sequence[Tuple[Any, ...]],
Sequence[Dict[Any, Any]]]


llm_tokenization_table = {
    "llama-2-13b": "./model_hub/llama-2-13b",
    "llama-13b": "./model_hub/llama-13b",
    "llama-3-8b": "./model_hub/llama-3-8b",
    "llama-2-7b": "./model_hub/llama-2-7b",
    "llama-7b": "./model_hub/llama-7b",
    "gpt2": "./model_hub/gpt2"
}


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
        configs.seq_len = state_shape   # TODO: The seq_len would be dynamic in our case. How to modify GlucoseLLM accordingly?
        configs.llm_model = llm
        configs.llm_dim = llm_dim
        super().__init__(configs, need_llm=need_llm)
        self.configs = configs
        self.input_shape = state_shape
        self.output_shape = action_shape
        self.llm = llm

    def forward_Q(self, series, messages):
        pass

    def forward_text(self, messages, temp=0.2, max_length=300, top_p=0.3):
        pass

    def forward(self, series, prompt, temp=0.2, max_length=300, top_p=0.3, mode='Q', mask=None, state=None, info={}):
        # todo: must return logits and state, split the forward function into two functions
        logits, state = None, None
        tokenizer = llm_tokenization_table[self.llm]
        pipe = pipeline("conversational", tokenizer)
        messages = pipe(prompt)
        if mode == 'Q':
            logits, state = self.forward_Q(series, messages)
            llm_output = ""
        elif mode == 'str':
            llm_output = self.forward_text(messages, temp=temp, max_length=max_length, top_p=top_p)
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

    def explain_action(self, conversation, mode='str'):
        prompt = conversation.append_question("Please explain why the agent chose the last action within 100 words:")
        temp, max_length, top_p = 0.2, 300, 0.3
        series=torch.tensor([])
        _, _, response = self.forward(series, prompt, temp, max_length, top_p)
        return "Explanation of the last action: " + response

    def explain_obs(self, conversation, mode='str'):
        prompt = conversation.append_question("Please analyze the current state within 100 words:")
        temp, max_length, top_p = 0.2, 300, 0.3
        series=torch.tensor([])
        _, _, response = self.forward(series, prompt, temp, max_length, top_p)
        return "Analysis of the current state: " + response
    
    def q_pred(self, series, conversation, mode='Q'):
        prompt = conversation.append_question(f"Please predict the q value for the {self.num_actions} possible actions in the next timestep:")
        q_list, _, _ = self.forward(series, prompt, mode=mode)
        return q_list


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
        if not attr_name.startswith('__') and attr_name not in excluded_attributes and not callable(getattr(model, attr_name)):
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
        d_ff = 32,
        patch_len = 16,  # TODO: Adaptive value?
        stride = 8,  # TODO: Adaptive value?
        llm_layers = 6,
        d_model = 16,
        dropout = 0.1,
        n_heads = 8,
        enc_in = 7,
        prompt_domain = 0,
        content = "",
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
        if obs.ndim == 3:
            obs = obs.reshape(obs.shape[0], -1)
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
        if state is None:
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

class RecurrentPreprocess(nn.Module):
    def __init__(
            self,
            layer_num: int,
            state_shape: Union[int, Sequence[int]],
            device: Union[str, int, torch.device] = "cpu",
            hidden_layer_size: int = 128,
            dropout: float = 0.0,
            last_step_only: bool = True,
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
        self.fc1 = nn.Linear(int(np.prod(state_shape)), hidden_layer_size)
        self.use_last_step = last_step_only

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
        if state is None:
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
            obs = obs[:, -1]
        # please ensure the first dim is batch size: [bsz, len, ...]
        return obs, {
            "hidden": hidden.transpose(0, 1).detach(),
            "cell": cell.transpose(0, 1).detach()
        }

def define_single_network(input_shape: int, output_shape: int,
                          use_rnn=False, use_dueling=False, cat_num: int = 1, linear=False,
                          device="cuda" if torch.cuda.is_available() else "cpu",
                          ):
    if use_dueling and use_rnn:
        raise NotImplementedError("rnn and dueling are not implemented together")

    if use_dueling:
        if linear:
            dueling_params = ({"hidden_sizes": (), "activation": None},
                              {"hidden_sizes": (), "activation": None})
        else:
            dueling_params = ({"hidden_sizes": (256, 256), "activation": nn.ReLU},
                              {"hidden_sizes": (256, 256), "activation": nn.ReLU})
    else:
        dueling_params = None
    if not use_rnn:
        net = Net(state_shape=input_shape, action_shape=output_shape,
                  hidden_sizes=(256, 256, 256, 256) if not linear else (), activation=nn.ReLU if not linear else None,
                  device=device, dueling_param=dueling_params, cat_num=cat_num).to(device)
    else:
        net = Recurrent(layer_num=3,
                        state_shape=input_shape,
                        action_shape=output_shape,
                        device=device,
                        hidden_layer_size=256,
                        ).to(device)

    return net


class QRDQN(nn.Module):
    """Reference: Distributional Reinforcement Learning with Quantile \
    Regression.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(self, state_shape, action_shape, hidden_sizes=(256, 256, 256, 256), activation=nn.ReLU,
                 num_quantiles=200, cat_num: int = 1, device="cpu"):
        super(QRDQN, self).__init__()
        self.input_shape = state_shape
        self.action_shape = action_shape
        self.cat_num = cat_num
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.num_quantiles = num_quantiles
        self.device = device
        model_list = []
        for i in range(len(hidden_sizes)):
            if i == 0:
                model_list.append(nn.Linear(state_shape * self.cat_num, hidden_sizes[i]))
            else:
                model_list.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            model_list.append(self.activation())
        if hidden_sizes:
            model_list.append(nn.Linear(hidden_sizes[-1], action_shape * num_quantiles))
        else:
            model_list.append(nn.Linear(state_shape * self.cat_num, action_shape * num_quantiles))
        self.model = nn.Sequential(*model_list)

    def forward(
            self,
            obs: Union[np.ndarray, torch.Tensor],
            state: Optional[Any] = None,
            info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        r"""Mapping: x -> Z(x, \*)."""
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        if obs.ndim == 3:
            obs = obs.reshape(obs.shape[0], -1)
        obs = self.model(obs)
        obs = obs.view(-1, self.action_shape, self.num_quantiles)
        return obs, state

