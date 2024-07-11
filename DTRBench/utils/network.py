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
from transformers import LogitsProcessorList, TopKLogitsWarper, TopPLogitsWarper, TemperatureLogitsWarper, StoppingCriteria, StoppingCriteriaList, MaxLengthCriteria
import torch
from torch import nn
import torch.nn.functional as F
from tianshou.data.batch import Batch

ModuleType = Type[nn.Module]
ArgsType = Union[Tuple[Any, ...], Dict[Any, Any], Sequence[Tuple[Any, ...]],
Sequence[Dict[Any, Any]]]

llm_context_window = {
    "internlm2_5-7b-chat": 32768,
    "Phi-3-small-128k-instruct": 131072,
    "Yi-1.5-9b-Chat": 4096,
    "Qwen2-1.5B-Instruct": 32768,
}

obs_exp_prompt = ("The Simglucose environment is a simulation environment designed to mimic the physiological dynamics "
                  "of glucose metabolism in humans, often used in research of glucose control. "
                  "The primary goal in the Simglucose environment is to maintain a patient's blood glucose levels "
                  "(the observation) within a target range through the administration of insulin (the action). "
                  "The reason for a high value observation (high Blood Glucose Level (BG): the current blood glucose "
                  "concentration in mg/dL) is typically in last/last several timestep, more insulin (action) was "
                  "injected, a raising or high level of action is witnessed, and vice versa. The following would be "
                  "conversation history between user and you as an assistant, where the user gives blood glucose "
                  "observation and the agent's action (Insulin Bolus Dose) at every timestep, and next timestep's "
                  "observation, then the assistant give explaination to the observation at every next timestep.") # expertised system prompt of background knowledge for observation explanation
Q_prompt = ("The Simglucose environment is a simulation environment designed to mimic the physiological dynamics of "
            "glucose metabolism in humans, often used in research of glucose control. The primary goal in the "
            "Simglucose environment is to maintain a patient's blood glucose levels (the observation) within a target "
            "range through the administration of insulin (the action). 5 number of actions (Insulin Bolus Dose) "
            "represents 5 degrees of insulin injection to restrain high blood glucose level. In Q-learning, "
            "the Q-value represents the expected future rewards for taking a given action in a given state, "
            "with high Q-values indicating more favorable actions and low Q-values indicating less favorable actions. "
            "So for a q-learning agent, if the blood glucose level is observed to be high, the q value of the high "
            "value action should be high, and q value of the low value action should be low, and vice versa for low "
            "blood glucose level. The following would be conversation history between user and you as an assistant, "
            "where the user gives blood glucose observation and the agent's action (Insulin Bolus Dose) at every timestep, "
            "then the assistant give explaination to both the observation and action at every timestep.")       # expertised system prompt for series information description and Q value prediction
act_exp_prompt = ("The Simglucose environment is a simulation environment designed to mimic the physiological dynamics "
                  "of glucose metabolism in humans, often used in research of glucose control. The primary goal in the "
                  "Simglucose environment is to maintain a patient's blood glucose levels (the observation) within a "
                  "target range through the administration of insulin (the action). The reason for a high value action "
                  "(high Insulin Bolus Dose measured in units (U) of insulin) is typically in current timestep or the "
                  "past several timesteps, a relatively high value of Blood Glucose Level (BG): the current blood "
                  "glucose concentration in mg/dL is observed (low observation), thus the patient needs more insulin "
                  "to prevent the blood glucose from getting too high, and vice versa. The following would be conversation "
                  "history between user and you as an assistant, where the user gives blood glucose observation and "
                  "the agent's action (Insulin Bolus Dose) at every timestep, then the assistant give explaination to the "
                  "action at every timestep.") # expertised system prompt of background knowledge for action explanation
summary_prompt = ("The Simglucose environment is a simulation environment designed to mimic the physiological dynamics "
                  "of glucose metabolism in humans, often used in research of glucose control. The primary goal in the "
                  "Simglucose environment is to maintain a patient's blood glucose levels (the observation) within a "
                  "target range through the administration of insulin (the action). The reason for a high value action "
                  "(high Insulin Bolus Dose measured in units (U) of insulin) is typically in current timestep or the "
                  "past several timesteps, a relatively high value of Blood Glucose Level (BG): the current blood "
                  "glucose concentration in mg/dL is observed (low observation), thus the patient needs more insulin "
                  "to prevent the blood glucose from getting too high, and vice versa. The reason for a high value "
                  "observation (high Blood Glucose Level (BG): the current blood glucose concentration in mg/dL) is "
                  "typically in last/last several timestep, more insulin (action) was injected, a raising or high level "
                  "of action is witnessed, and vice versa.") # expertised system prompt of background knowledge for regulation summary


class LLMNet(GlucoseLLM.Model):
    def __init__(
            self,
            configs: argparse.Namespace,
            state_shape: Union[int, Sequence[int]],
            action_shape: Union[int, Sequence[int]] = 0,
            device: Union[str, int, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
            llm: str = "Qwen2-1.5B-Instruct",
            llm_dim: int = 1536,
            need_llm: bool = False,
    ) -> None:
        if isinstance(action_shape, int):
            self.num_actions = action_shape
        elif isinstance(action_shape, Sequence):
            self.num_actions = 1
            for dim in action_shape:
                self.num_actions *= dim
        configs.pred_len = self.num_actions
        configs.seq_len = state_shape
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
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.llm_model.device)
        outputs = self.llm_model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=1
        )
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = generated_text[len(prompt):].strip()
        cutoff_index = generated_text.find("<|im_end|>")
        if cutoff_index != -1:
            generated_text = generated_text[:cutoff_index].strip()
        return generated_text

    def forward(self, series, messages, max_length=100, mode='Q'):
        logits, state = None, None
        # prompt = messages.to_str()
        prompt = self.tokenizer.apply_chat_template(messages.conversation, tokenize=False, add_generation_prompt=True)
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

    def explain_obs(self, conversation, summary, mode='str'):
        prompt = conversation.clip(llm_context_window[self.llm]-300, self.tokenizer)
        prompt.insert_component("system", obs_exp_prompt, 0)
        if summary is None:
            prompt.append_content("Please analyze the current state within 100 words:", -1)
        else:
            prompt.append_content("Extracted rules and regulations: "+summary+"Please analyze the current state within 100 words:", -1)
        series=torch.tensor([]).to(self.device)
        _, _, response = self.forward(series, prompt, max_length=256, mode=mode)
        return response
    
    def q_pred(self, series, conversation, summary, mode='Q'):
        prompt = conversation.clip(llm_context_window[self.llm]-300, self.tokenizer)
        prompt.insert_component("system", Q_prompt, 0)
        if summary is None:
            prompt.append_content(f"Please predict the q value for the {self.num_actions} possible actions in the next timestep:", -1)
        else:
            prompt.append_content("Extracted rules and regulations: "+summary+f"Please predict the q value for the {self.num_actions} possible actions in the next timestep:", -1)
        series = torch.tensor(series, dtype=torch.float32).unsqueeze(-1).to(self.device)
        q_list, _, _ = self.forward(series, prompt, max_length=256, mode=mode)
        return q_list
    
    def explain_act(self, conversation, summary, mode='str'):
        prompt = conversation.clip(llm_context_window[self.llm]-300, self.tokenizer)
        prompt.insert_component("system", act_exp_prompt, 0)
        if summary is None:
            prompt.append_content("Please explain why the agent chose the last action within 100 words:", -1)
        else:
            prompt.append_content("Extracted rules and regulations: "+summary+"Please explain why the agent chose the last action within 100 words:", -1)
        series=torch.tensor([]).to(self.device)
        _, _, response = self.forward(series, prompt, max_length=256, mode=mode)
        return response

    def summarize(self, conversation, mode='str'):
        prompt = conversation.clip(llm_context_window[self.llm]-300, self.tokenizer)
        prompt.insert_component("system", summary_prompt, 0)
        prompt.append_content("Please summarize the rules and regulations you can observe or extract from the history data and background information. Separate each rule with a serial number.", -1)
        series=torch.tensor([]).to(self.device)
        _, _, response = self.forward(series, prompt, max_length=256, mode=mode)
        return response


def define_llm_network(input_shape: int, output_shape: int,
                          device="cuda" if torch.cuda.is_available() else "cpu", llm="Qwen2-1.5B-Instruct", llm_dim=1536,
                          ):
    configs = argparse.Namespace(
        d_ff = 32,
        patch_len = 9,  # TODO: Adaptive value?
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

