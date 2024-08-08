from typing import (
    Any,
    Dict,
    Sequence,
    Tuple,
    Type,
    Union,
)
from pathlib import Path
import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from GlucoseLLM.model.net import timeLLM
from GlucoseLLM.prompt_utils import Conversation
from GlucoseLLM.prompts import (SYSTEM_PROMPT, ACTOR_INSTRUCTION_PROMPT, SUMMARY_INSTRUCTION_PROMPT,
                                LLM_INFERENCE_INSTRUCTION_PROMPT, get_Q_instruction, get_patient_info_prompt)
ModuleType = Type[nn.Module]
ArgsType = Union[Tuple[Any, ...], Dict[Any, Any], Sequence[Tuple[Any, ...]],
Sequence[Dict[Any, Any]]]

llm_context_window = {
    "internlm2_5-7b-chat": 32768,
    "Phi-3-small-128k-instruct": 131072,
    "Yi-1.5-9B-Chat": 4096,
    "Qwen2-1.5B-Instruct": 32768,
    "Qwen2-0.5B-Instruct": 32768,
}


class LLMDQN(timeLLM):
    def __init__(self, state_shape: Union[int, Sequence[int]], action_shape: Union[int, Sequence[int]], llm,
                 seq_len, d_model, d_ff, patch_len, stride, token_dim, n_heads, enc_in, keep_old=False, dropout: float = 0.1,
                 device: Union[str, int, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", model_dir=None):
        super().__init__(llm=llm, n_vars=state_shape, seq_len=seq_len, d_model=d_model, d_ff=d_ff, patch_len=patch_len,
                         stride=stride,
                         token_dim=token_dim, n_heads=n_heads, enc_in=enc_in,
                         keep_old=keep_old, dropout=dropout, model_dir=model_dir)
        self.output_shape = int(np.prod(action_shape))
        self.device = device

    def q_pred(self, ts, conversations, model='current'):
        prompts = []
        for conversation in conversations:
            prompt = conversation.clip(llm_context_window[self.llm] - 300, self.tokenizer)
            prompt.insert_component("system", self.Q_prompt, 0)
            prompts.append(prompt)
        ts = torch.stack(ts, dim=0).to(self.device)
        q_list, _, _ = self.forward(ts, prompts, model=model)
        return q_list

    def summarize(self, ts):
        bs = ts.shape[0]
        prompts = []
        for _ in range(bs):
            conversation = Conversation()
            conversation.insert_component("system", SYSTEM_PROMPT, 0)
            conversation.insert_component("user", SUMMARY_INSTRUCTION_PROMPT, -1)
            prompts.append(conversation.get())
        ts = torch.from_numpy(ts).to(self.device)
        # todo: add obs2text here
        _, _, response = self.generate_text(ts, prompts)
        return response


def define_llm_dqn(input_shape: int, output_shape: int,
                   device="cuda" if torch.cuda.is_available() else "cpu", llm="Qwen2-1.5B-Instruct", token_dim=1536,
                   ):
    net = LLMDQN(state_shape=input_shape, action_shape=output_shape, llm=llm, seq_len=12, d_model=16,
                 d_ff=32, patch_len=6, stride=3, token_dim=token_dim, n_heads=8,
                 enc_in=7, keep_old=False, dropout=0.,
                 model_dir=Path(__file__).resolve().parent.absolute() / "model_hub").to(device)
    return net


class LLMPPO(timeLLM):
    pass


def define_llm_ppo():
    pass


