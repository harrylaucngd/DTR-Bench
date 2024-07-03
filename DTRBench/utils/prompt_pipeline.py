import torch
from transformers import LlamaTokenizer, GPT2Tokenizer, AutoTokenizer


class Conversation:
    def __init__(self):
        # Initialize an empty conversation list
        self.conversation = []

    def add_component(self, role, content):
        # Add a new component to the conversation
        if role in ["system", "user", "assistant"]:
            self.conversation.append({"role": role, "content": content})
            self.syntax_check()
        else:
            raise ValueError("Role must be 'system', 'user', or 'assistant'.")
        
    def insert_component(self, role, content, loc):
        # Insert a new component at the specified location
        if role in ["system", "user", "assistant"]:
            if loc < 0:
                loc = len(self.conversation) + loc + 1
            if loc > len(self.conversation):
                loc = len(self.conversation)
            self.conversation.insert(loc, {"role": role, "content": content})
            self.syntax_check()
        else:
            raise ValueError("Role must be 'system', 'user', or 'assistant'.")

    def syntax_check(self):
        # Check for neighboring roles that are the same in the conversation
        for i in range(1, len(self.conversation)):
            if self.conversation[i]["role"] == self.conversation[i-1]["role"]:
                raise ValueError(f"Syntax error: Consecutive '{self.conversation[i]['role']}' roles found.")
    
    def count_tokens(self, text, tokenizer):
        # Use LLM tokenizer to detect token overflow
        tokens = tokenizer.encode(text)
        return len(tokens)
    
    def clip(self, context_length, tokenizer):
        # Clip the conversation to fit within the context length
        conv_str = '\n'.join(f'{component["role"]}: {component["content"]}' for component in self.conversation)
        tokens = self.count_tokens(conv_str, tokenizer)
        
        if tokens <= context_length:
            return self
        
        for i in range(len(self.conversation)):
            conv_str = '\n'.join(f'{component["role"]}: {component["content"]}' for component in self.conversation[i+1:])
            tokens = self.count_tokens(conv_str, tokenizer)
            if tokens <= context_length:
                clipped_conversation = Conversation()
                clipped_conversation.conversation = self.conversation[i+1:]
                return clipped_conversation

        return Conversation()
    

def obs_prompt_reprogramming(obs, act, obs_exp):
    history_prompt = Conversation()
    for i, (o, a, exp) in enumerate(zip(obs, act, obs_exp)):
        history_prompt.add_component("user", f"In timestep {i}, the blood glucose observation is {o}(mg/dL), the agent takes action {a}, the next blood glucose observation is {obs[i+1]}(mg/dL). Please analyze the current state within 100 words: ")
        history_prompt.add_component("assistant", f"{exp}")
    history_prompt.add_component("user", f"In current timestep, the blood glucose observation is {obs[-1]}(mg/dL). ")
    return history_prompt


def q_prompt_reprogramming(obs, act, obs_exp, act_exp):
    series = torch.tensor([])
    history_prompt = Conversation()
    for (o, a) in zip(obs, act):
        series = torch.cat((series, torch.tensor([o])), dim=0)
        series = torch.cat((series, torch.tensor([a])), dim=0)
    series = torch.cat((series, torch.tensor([obs[-1]])), dim=0)
    for i, (o, a, o_exp, a_exp) in enumerate(zip(obs, act, obs_exp, act_exp)):
        history_prompt.add_component("user", f"In timestep {i}, the blood glucose observation is {o}(mg/dL). Please analyze the current state within 100 words: ")
        history_prompt.add_component("assistant", f"{o_exp}")
        history_prompt.add_component("user", f"In timestep {i}, then the agent takes action {a}. Please explain why the agent chose the last action within 100 words: ")
        history_prompt.add_component("assistant", f"{a_exp}")
    history_prompt.add_component("user", f"In current timestep, the blood glucose observation is {obs[-1]}(mg/dL). Please analyze the current state within 100 words: ")
    history_prompt.add_component("assistant", f"{obs_exp[-1]}")
    series = torch.unsqueeze(series, 0)
    return series, history_prompt


def act_prompt_reprogramming(obs, act, act_exp):
    history_prompt = Conversation()
    for i, (o, a, exp) in enumerate(zip(obs, act, act_exp)):
        history_prompt.add_component("user", f"In timestep {i}, the blood glucose observation is {o}(mg/dL), the agent takes action {a}. Please explain why the agent chose the last action within 100 words: ")
        history_prompt.add_component("assistant", f"{exp}")
    history_prompt.add_component("user", f"In current timestep, the blood glucose observation is {obs[-1]}(mg/dL), the agent takes action {act[-1]}. ")
    return history_prompt