import numpy as np
from tianshou.utils.net.common import MLP
from typing import (
    Any,
    Dict,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)
import torch
from torch import nn

ModuleType = Type[nn.Module]
ArgsType = Union[Tuple[Any, ...], Dict[Any, Any], Sequence[Tuple[Any, ...]],
Sequence[Dict[Any, Any]]]

# todo: move the following llm code to a separate file



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


class Critic(nn.Module):

    def __init__(
            self,
            state_net: nn.Module,
            action_net: nn.Module,
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
            act: np.ndarray | torch.Tensor,
            state: Optional[Dict[str, torch.Tensor]] = None,
            info: dict[str, Any] | None = {},
    ):
        """Mapping: (s, a) -> logits -> Q(s, a)."""
        obs, state = self.obs_net(obs, state)  # state here won't be useful anyway since tianshou does not RNN-critic
        act, _ = self.act_net(act)
        obs = torch.cat([obs, act], dim=1)
        value = self.last(obs)
        return value


def define_single_network(input_shape: int, output_shape: int, hidden_size=256, num_layer=4,
                          use_rnn=False, use_dueling=False, cat_num: int = 1, linear=False,
                          device="cuda" if torch.cuda.is_available() else "cpu",
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
                             device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Since Tianshou's critic network does not support RNN style network, we use a simple MLP network here.
    """
    obs_net = Net(state_shape=state_shape, action_shape=state_net_hidden_size,
                  hidden_sizes=(state_net_hidden_size,) * state_net_n_layer if not linear else (),
                  activation=nn.ReLU if not linear else None,
                  device=device, dueling_param=None, cat_num=1).to(device)
    act_net = Net(state_shape=action_shape, action_shape=action_net_hidden_size,
                  hidden_sizes=action_net_n_layer * [action_net_hidden_size],
                  activation=nn.ReLU,
                  device=device, cat_num=1).to(device)
    critic = Critic(obs_net, act_net, cat_size=state_net_hidden_size + action_net_hidden_size,
                    fuse_hidden_sizes=[state_net_hidden_size + action_net_hidden_size] * fuse_net_n_layer,
                    device=device).to(device)

    return critic
