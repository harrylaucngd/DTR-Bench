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
                 seq_len, d_ff, patch_len, stride, token_dim, n_heads, enc_in, keep_old=False, dropout: float = 0.1,
                 device: Union[str, int, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", model_dir=None):
        super().__init__(llm, seq_len, d_ff, patch_len, stride, token_dim, n_heads, enc_in, keep_old, dropout, model_dir)
        self.input_shape = state_shape
        self.output_shape = int(np.prod(action_shape))
        self.device = device

    def q_pred(self, series, conversations, model='current'):
        prompts = []
        for conversation in conversations:
            prompt = conversation.clip(llm_context_window[self.llm] - 300, self.tokenizer)
            prompt.insert_component("system", self.Q_prompt, 0)
            prompts.append(prompt)
        series = torch.stack(series, dim=0).to(self.device)
        q_list, _, _ = self.forward(series, prompts, model=model)
        return q_list

    def summarize(self, conversations):
        prompts = []
        for conversation in conversations:
            prompt = conversation.clip(llm_context_window[self.llm] - 300, self.tokenizer)
            prompt.insert_component("system", self.summary_prompt, 0)
            prompts.append(prompt)
        series = torch.tensor([]).to(self.device)
        _, _, response = self.generate_text(series, prompts, max_length=256)
        return response


def define_llm_dqn(input_shape: int, output_shape: int,
                   device="cuda" if torch.cuda.is_available() else "cpu", llm="Qwen2-1.5B-Instruct", token_dim=1536,
                   ):
    net = LLMDQN(state_shape=input_shape, action_shape=output_shape, llm=llm, seq_len=48,
                 d_ff=32, patch_len=9, stride=8, token_dim=token_dim, n_heads=4,
                 enc_in=7, keep_old=False, dropout=0.,
                 model_dir=Path(__file__).resolve().parent.absolute() / "model_hub").to(device)
    return net


class LLMPPO(timeLLM):
    pass


def define_llm_ppo():
    pass


class LLM(torch.nn.Module):
    def __init__(self, llm="Qwen2-1.5B-Instruct", context_window=32768,
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 system_prompt=False):
        super().__init__()
        self.llm = llm
        self.max_length = context_window
        self.device = device
        self.system_prompt = system_prompt

        self.tokenizer = AutoTokenizer.from_pretrained(f"model_hub/{self.llm}", trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(f"model_hub/{self.llm}", trust_remote_code=True).to(
            self.device)

    def forward(self, input_text):
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.append({"role": "user", "content": input_text})
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False,
                                                    add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=self.max_length,
                do_sample=False,
                temperature=1
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        cutoff_index = generated_text.rfind("assistant\n")
        if cutoff_index != -1:  # answer cutoff
            generated_text = generated_text[cutoff_index + len("assistant\n"):]
        return generated_text


def define_llm(llm="Qwen2-1.5B-Instruct", context_window=32768,
               device="cuda" if torch.cuda.is_available() else "cpu",
               system_prompt=False,
               ):
    net = LLM(llm=llm, context_window=context_window, device=device, system_prompt=system_prompt).to(device)
    return net
