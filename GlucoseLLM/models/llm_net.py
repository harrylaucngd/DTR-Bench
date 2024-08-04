import argparse
from GlucoseLLM.models import GlucoseLLM
from GlucoseLLM.prompt_pipeline import Conversation
from typing import (
    Any,
    Dict,
    Sequence,
    Tuple,
    Type,
    Union,
)
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM

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


class LLMDQN(GlucoseLLM.Model):
    def __init__(
            self,
            configs: argparse.Namespace,
            state_shape: Union[int, Sequence[int]],
            action_shape: Union[int, Sequence[int]] = 0,
            device: Union[str, int, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
            need_llm: bool = False,
            # prompt options
            summary_prompt=False, Q_prompt=False,
    ) -> None:
        if isinstance(action_shape, int):
            self.num_actions = action_shape
        elif isinstance(action_shape, Sequence):
            self.num_actions = 1
            for dim in action_shape:
                self.num_actions *= dim
        configs.pred_len = self.num_actions
        configs.seq_len = 96
        super().__init__(configs, need_llm=need_llm)
        self.configs = configs
        self.llm = self.configs.llm
        self.input_shape = state_shape
        self.output_shape = action_shape
        self.device = device
        self.summary_prompt = summary_prompt
        self.Q_prompt = Q_prompt

    def forward_Q(self, series, prompts):
        # Inference the whole network
        dec_out = self.forecast(series, prompts)
        return dec_out[:, -self.pred_len:, :].squeeze(-1), []

    def forward_text(self, prompts, max_length=256):
        inputs = self.tokenizer(prompts, return_tensors="pt").to(self.llm_model.device)
        outputs = self.llm_model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=False,
            temperature=1
        )
        generated_texts = []
        for output in outputs:
            generated_text = self.tokenizer.decode(output, skip_special_tokens=True)
            cutoff_index = generated_text.rfind("assistant\n")
            if cutoff_index != -1:  # answer cutoff
                generated_text = generated_text[cutoff_index + len("assistant\n"):]
            generated_texts.append(generated_text)
        return generated_texts

    def forward(self, series, messages, max_length=256, mode='Q'):
        logits, state = None, None
        # prompt = messages.to_str()
        prompts = []
        if isinstance(messages, Conversation):
            prompts = self.tokenizer.apply_chat_template(messages.conversation, tokenize=False,
                                                         add_generation_prompt=True)
        else:
            for message in messages:
                prompt = self.tokenizer.apply_chat_template(message.conversation, tokenize=False,
                                                            add_generation_prompt=True)
                prompts.append(prompt)
        if mode == 'Q':
            logits, state = self.forward_Q(series, prompts)
            llm_output = ""
        elif mode == 'str':
            llm_output = self.forward_text(prompts, max_length=max_length)
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

    def q_pred(self, series, conversations, mode='Q'):
        prompts = []
        for conversation in conversations:
            prompt = conversation.clip(llm_context_window[self.llm] - 300, self.tokenizer)
            prompt.insert_component("system", self.Q_prompt, 0)
            prompts.append(prompt)
        series = torch.stack(series, dim=0).to(self.device)
        q_list, _, _ = self.forward(series, prompts, max_length=256, mode=mode)
        return q_list

    def summarize(self, conversations, mode='str'):
        prompts = []
        for conversation in conversations:
            prompt = conversation.clip(llm_context_window[self.llm] - 300, self.tokenizer)
            prompt.insert_component("system", self.summary_prompt, 0)
            prompts.append(prompt)
        series = torch.tensor([]).to(self.device)
        _, _, response = self.forward(series, prompts, max_length=256, mode=mode)
        return response


def define_llm_dqn(input_shape: int, output_shape: int,
                   device="cuda" if torch.cuda.is_available() else "cpu", llm="Qwen2-1.5B-Instruct", token_dim=1536,
                   Q_prompt=False, summary_prompt=False,
                   ):
    configs = argparse.Namespace(
        d_ff=32,
        patch_len=9,  # TODO: TBD
        stride=8,  # TODO: TBD
        llm_layers=6,
        d_model=16,
        dropout=0.,
        n_heads=8,
        enc_in=7,
        token_dim=token_dim,
        llm=llm,
    )
    net = LLMDQN(configs=configs, state_shape=input_shape, action_shape=output_shape,
                 device=device,
                 # prompt options
                 summary_prompt=summary_prompt, Q_prompt=Q_prompt).to(device)
    return net


class LLMPPO(GlucoseLLM.Model):
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
