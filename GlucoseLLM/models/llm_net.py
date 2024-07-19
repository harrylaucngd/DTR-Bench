import os
import numpy as np
import argparse
from GlucoseLLM.models import GlucoseLLM
from GlucoseLLM.prompt_pipeline import Conversation
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
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM

ModuleType = Type[nn.Module]
ArgsType = Union[Tuple[Any, ...], Dict[Any, Any], Sequence[Tuple[Any, ...]],
Sequence[Dict[Any, Any]]]

llm_context_window = {
    "internlm2_5-7b-chat": 32768,
    "Phi-3-small-128k-instruct": 131072,
    "Yi-1.5-9b-Chat": 4096,
    "Qwen2-1.5B-Instruct": 32768,
}


class LLMNet(GlucoseLLM.Model):
    # todo: overwrite by merge. Pls check the code
    def __init__(
            self,
            configs: argparse.Namespace,
            state_shape: Union[int, Sequence[int]],
            action_shape: Union[int, Sequence[int]] = 0,
            device: Union[str, int, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
            need_llm: bool = False,
            # prompt options
            summary_prompt=False, obs_exp_prompt=False, Q_prompt=False, act_exp_prompt=False,
    ) -> None:
        if isinstance(action_shape, int):
            self.num_actions = action_shape
        elif isinstance(action_shape, Sequence):
            self.num_actions = 1
            for dim in action_shape:
                self.num_actions *= dim
        configs.pred_len = self.num_actions
        configs.seq_len = state_shape  # TODO: need padding
        super().__init__(configs, need_llm=need_llm)
        self.configs = configs
        self.llm = self.configs.llm
        self.input_shape = state_shape
        self.output_shape = action_shape
        self.device = device

        self.summary_prompt = summary_prompt
        self.obs_exp_prompt = obs_exp_prompt
        self.Q_prompt = Q_prompt
        self.act_exp_prompt = act_exp_prompt

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
        cutoff_index = generated_text.rfind("assistant\n")
        if cutoff_index != -1:  # answer cutoff
            generated_text = generated_text[cutoff_index + len("assistant\n"):]
        return generated_text

    def forward(self, series, messages, max_length=100, mode='Q'):
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

    def explain_obs(self, conversation, summary, mode='str'):
        prompt = conversation.clip(llm_context_window[self.llm] - 300, self.tokenizer)
        prompt.insert_component("system", self.obs_exp_prompt, 0)
        if summary is None:
            prompt.append_content("Please analyze the current state within 100 words:", -1)
        else:
            prompt.append_content(
                "Extracted rules and regulations: " + summary + "Please analyze the current state within 100 words:",
                -1)
        series = torch.tensor([]).to(self.device)
        _, _, response = self.forward(series, prompt, max_length=256, mode=mode)
        return response

    def q_pred(self, series, conversations, summaries, mode='Q'):
        prompts = []
        for conversation, summary in zip(conversations, summaries):
            prompt = conversation.clip(llm_context_window[self.llm] - 300, self.tokenizer)
            prompt.insert_component("system", self.Q_prompt, 0)
            if summary is None:
                prompt.append_content(
                    f"Please predict the q value for the {self.num_actions} possible actions in the next timestep:", -1)
            else:
                prompt.append_content(
                    "Extracted rules and regulations: " + summary + f"Please predict the q value for the {self.num_actions} possible actions in the next timestep:",
                    -1)
            prompts.append(prompt)
        series = torch.tensor(series, dtype=torch.float32).to(self.device)
        q_list, _, _ = self.forward(series, prompts, max_length=256, mode=mode)
        return q_list

    def explain_act(self, conversation, summary, mode='str'):
        prompt = conversation.clip(llm_context_window[self.llm] - 300, self.tokenizer)
        prompt.insert_component("system", self.act_exp_prompt, 0)
        if summary is None:
            prompt.append_content("Please explain why the agent chose the last action within 100 words:", -1)
        else:
            prompt.append_content(
                "Extracted rules and regulations: " + summary + "Please explain why the agent chose the last action within 100 words:",
                -1)
        series = torch.tensor([]).to(self.device)

        _, _, response = self.forward(series, prompt, max_length=256, mode=mode)
        return response

    def summarize(self, conversation, mode='str'):
        prompt = conversation.clip(llm_context_window[self.llm] - 300, self.tokenizer)
        prompt.insert_component("system", self.summary_prompt, 0)
        prompt.append_content(
            "Please summarize the rules and regulations you can observe or extract from the history data and background information. Separate each rule with a serial number.",
            -1)
        series = torch.tensor([]).to(self.device)
        _, _, response = self.forward(series, prompt, max_length=256, mode=mode)
        return response


def define_llm_network(input_shape: int, output_shape: int,
                       device="cuda" if torch.cuda.is_available() else "cpu", llm="Qwen2-1.5B-Instruct", token_dim=1536,
                       obs_exp_prompt=False, Q_prompt=False, act_exp_prompt=False, summary_prompt=False,
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
    net = LLMNet(configs=configs, state_shape=input_shape, action_shape=output_shape,
                 device=device, need_llm=True,
                 # prompt options
                 summary_prompt=summary_prompt, obs_exp_prompt=obs_exp_prompt,
                 Q_prompt=Q_prompt, act_exp_prompt=act_exp_prompt).to(device)
    return net


class LLM(torch.nn.Module):
    def __init__(self, llm="Qwen2-1.5B-Instruct", context_window=32768, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.llm = llm
        self.max_length = context_window
        self.device = device
        
        self.tokenizer = AutoTokenizer.from_pretrained(f"model_hub/{self.llm}")
        self.model = AutoModelForCausalLM.from_pretrained(f"model_hub/{self.llm}").to(self.device)
    
    def forward(self, input_text):
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.max_length,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=1
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        cutoff_index = generated_text.rfind("assistant\n")
        if cutoff_index != -1:  # answer cutoff
            generated_text = generated_text[cutoff_index + len("assistant\n"):]
        return generated_text


def define_llm(llm="Qwen2-1.5B-Instruct", context_window=32768,
               device="cuda" if torch.cuda.is_available() else "cpu",
    ):
    net = LLM(llm=llm, context_window=context_window, device=device).to(device)
    return net
