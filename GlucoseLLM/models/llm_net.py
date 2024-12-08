import argparse
import warnings
from typing import Any, Dict, Sequence, Tuple, Type, Union, List

import numpy as np
import torch
from tianshou.utils.net.common import MLP
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM

from GlucoseLLM.models.timeLLM import timeLLM
from GlucoseLLM.prompt import Q_PROMPT, SYS_PROMPT, SUMMARY_PROMPT, ACT_PROMPT

ModuleType = Type[nn.Module]
ArgsType = Union[Tuple[Any, ...], Dict[Any, Any], Sequence[Tuple[Any, ...]], Sequence[Dict[Any, Any]]]

SIGMA_MIN = -20
SIGMA_MAX = 2

llm_context_window = {
    "internlm2_5-7b-chat": 32768,
    "Phi-3-small-128k-instruct": 131072,
    "Yi-1.5-9B-Chat": 4096,
    "Qwen2-1.5B-Instruct": 32768,
    "Qwen2-0.5B-Instruct": 32768,
}


class LLMPPO(timeLLM):
    def __init__(
        self,
        configs: argparse.Namespace,
        action_shape: Union[int, Sequence[int]] = 0,
        hidden_sizes: Sequence[int] = 128,
        max_action: float = 1.0,
        device: Union[str, int, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
        unbounded: bool = False,
        conditioned_sigma: bool = False,
        need_llm: bool = False,
        # prompt options
        summary_prompt=False,
        act_prompt=False,
    ) -> None:
        configs.pred_len = action_shape
        configs.seq_len = 96
        super().__init__(configs)
        self.llm = configs.llm
        self.device = device
        self.summary_prompt = summary_prompt
        self.act_prompt = act_prompt
        if unbounded and not np.isclose(max_action, 1.0):
            warnings.warn("Note that max_action input will be discarded when unbounded is True.")
            max_action = 1.0
        self.output_dim = int(np.prod(action_shape))
        input_dim = action_shape
        self.mu = MLP(input_dim, self.output_dim, hidden_sizes, device=self.device)
        self._c_sigma = conditioned_sigma
        if conditioned_sigma:
            self.sigma = MLP(
                input_dim,
                self.output_dim,
                hidden_sizes,
                device=self.device,
            )
        else:
            self.sigma_param = nn.Parameter(torch.zeros(self.output_dim, 1))
        self.max_action = max_action
        self._unbounded = unbounded

    def forward_act(self, series, prompts):
        # Inference the whole network
        dec_out = self.forecast(series, prompts)
        return dec_out[:, -self.action_size :, :].squeeze(-1), []

    def forward_text(self, prompts, max_length=256):
        inputs = self.tokenizer(prompts, return_tensors="pt").to(self.llm_model.device)
        outputs = self.llm_model.generate(**inputs, max_new_tokens=max_length, do_sample=False, temperature=1)
        generated_texts = []
        for output in outputs:
            generated_text = self.tokenizer.decode(output, skip_special_tokens=True)
            cutoff_index = generated_text.rfind("assistant\n")
            if cutoff_index != -1:  # answer cutoff
                generated_text = generated_text[cutoff_index + len("assistant\n") :]
            generated_texts.append(generated_text)
        return generated_texts

    def forward(self, ts, messages, max_length=256, mode="act"):
        logits, state = None, None
        # prompt = messages.to_str()
        prompts = []
        if isinstance(messages, Conversation):
            prompts = self.tokenizer.apply_chat_template(messages.conversation, tokenize=False, add_generation_prompt=True)
        else:
            for message in messages:
                prompt = self.tokenizer.apply_chat_template(message.conversation, tokenize=False, add_generation_prompt=True)
                prompts.append(prompt)
        if mode == "act":
            logits, state = self.forward_act(ts, prompts)
            llm_output = ""
            mu = self.mu(logits)
            if not self._unbounded:
                mu = self.max_action * torch.tanh(mu)
            if self._c_sigma:
                sigma = torch.clamp(self.sigma(logits), min=SIGMA_MIN, max=SIGMA_MAX).exp()
            else:
                shape = [1] * len(mu.shape)
                shape[1] = -1
                sigma = (self.sigma_param.view(shape) + torch.zeros_like(mu)).exp()
            return (mu, sigma), state, llm_output
        elif mode == "str":
            llm_output = self.forward_text(prompts, max_length=max_length)
            return logits, state, llm_output
        else:
            raise ValueError("Unsupported mode! Use 'act' for full network inference or 'str' for llm_model inference.")

    def freeze_llm_model(self):
        """Ensure all llm_model parameters are frozen."""
        for param in self.llm_model.parameters():
            param.requires_grad = False

    def unfreeze_llm_model(self):
        """Unfreeze all llm_model parameters, allowing them to be updated during training."""
        for param in self.llm_model.parameters():
            param.requires_grad = True

    def act_pred(self, series, conversations, mode="act"):
        prompts = []
        for conversation in conversations:
            prompt = conversation.clip(llm_context_window[self.llm] - 300, self.tokenizer)
            prompt.insert_component("system", self.act_prompt, 0)
            prompts.append(prompt)
        series = torch.stack(series, dim=0).to(self.device)
        (mu, sigma), _, _ = self.forward(series, prompts, max_length=256, mode=mode)
        return (mu, sigma)

    def summarize(self, conversations, mode="str"):
        prompts = []
        for conversation in conversations:
            prompt = conversation.clip(llm_context_window[self.llm] - 300, self.tokenizer)
            prompt.insert_component("system", self.summary_prompt, 0)
            prompts.append(prompt)
        series = torch.tensor([]).to(self.device)
        _, _, response = self.forward(series, prompts, max_length=256, mode=mode)
        return response


def define_llm_ppo(
    input_shape: int,
    output_shape: int,
    unbounded=True,
    device="cuda" if torch.cuda.is_available() else "cpu",
    llm="Qwen2-1.5B-Instruct",
    token_dim=1536,
    act_prompt=False,
    summary_prompt=False,
):
    configs = argparse.Namespace(
        d_ff=32,
        patch_len=9,  # TODO: TBD
        stride=8,  # TODO: TBD
        llm_layers=6,
        d_model=16,
        dropout=0.1,
        n_heads=8,
        enc_in=7,
        prompt_domain=0,
        content="",
        token_dim=token_dim,
        llm=llm,
    )
    net = LLMPPO(
        configs=configs,
        action_shape=output_shape,
        unbounded=unbounded,
        device=device,
        need_llm=True,
        # prompt options
        summary_prompt=summary_prompt,
        act_prompt=act_prompt,
    ).to(device)
    return net


class LLM(torch.nn.Module):
    """
    LLM inference only
    """

    def __init__(self, llm="Qwen2.5-1.5B-Instruct", context_window=32768, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.llm = llm
        self.max_length = context_window
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(f"model_hub/{self.llm}", trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(f"model_hub/{self.llm}", trust_remote_code=True).to(self.device)

    def forward(self, input_text: str, system_prompt=SYS_PROMPT) -> str:
        messages = [{"role": "system", "content": system_prompt}] if system_prompt is not None else []
        messages.append({"role": "user", "content": input_text})
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(inputs.input_ids, max_length=self.max_length, do_sample=False)

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Clip by prompt
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt) :].strip()  # Remove the prompt from the generated text

        return generated_text


def define_llm(
    llm="Qwen2-1.5B-Instruct",
    context_window=32768,
    device="cuda" if torch.cuda.is_available() else "cpu",
    system_prompt=False,
):
    net = LLM(llm=llm, context_window=context_window, device=device, system_prompt=system_prompt).to(device)
    return net
