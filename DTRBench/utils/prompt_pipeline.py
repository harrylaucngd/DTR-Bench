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

    def append_content(self, additional_content, pos):
        # Append additional content to the content of the element at position pos
        if pos < 0:
            pos = len(self.conversation) + pos
        if 0 <= pos < len(self.conversation):
            self.conversation[pos]["content"] += additional_content
            self.syntax_check()
        else:
            raise IndexError("Position out of range.")

    def syntax_check(self):
        # Check for neighboring roles that are the same in the conversation
        i = 1
        while i < len(self.conversation):
            if self.conversation[i]["role"] == self.conversation[i-1]["role"]:
                # Append content of the current role to the previous role
                self.conversation[i-1]["content"] += self.conversation[i]["content"]
                # Remove the current role
                self.conversation.pop(i)
            else:
                i += 1
    
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

    def to_str(self):
        str = '\n'.join(f'{component["role"]}: {component["content"]}' for component in self.conversation)
        return str
    

def obs_prompt_reprogramming(obs, act, obs_exp):
    history_prompt = Conversation()
    for i, (o, a, exp) in enumerate(zip(obs, act, obs_exp)):
        history_prompt.add_component("user", f"In timestep {i}, the blood glucose observation is {o} (mg/dL), the agent takes action (Insulin Bolus Dose) {a}, the next blood glucose observation is {obs[i+1]} (mg/dL).")
        if exp != "":
            history_prompt.add_component("assistant", f"{exp}")
    history_prompt.add_component("user", f"In current timestep, the blood glucose observation is {obs[-1]}(mg/dL). ")
    return history_prompt


def q_prompt_reprogramming(obs, act, obs_exp, act_exp):
    series = torch.tensor([])
    history_prompt = Conversation()
    obs_tensor = torch.tensor(obs)
    act_tensor = torch.tensor(act)
    zero_tensor = torch.tensor([0])
    act_tensor = torch.cat((zero_tensor, act_tensor))
    series = torch.empty(2 * len(obs_tensor), dtype=obs_tensor.dtype)
    series[0::2] = obs_tensor
    series[1::2] = act_tensor
    series = series.unsqueeze(1)
    for i, (o, a, o_exp, a_exp) in enumerate(zip(obs, act, obs_exp, act_exp)):
        history_prompt.add_component("user", f"In timestep {i}, the blood glucose observation is {o} (mg/dL).")
        if o_exp != "":
            history_prompt.add_component("assistant", f"{o_exp}")
        history_prompt.add_component("user", f"In timestep {i}, then the agent takes action (Insulin Bolus Dose) {a}.")
        if a_exp != "":
            history_prompt.add_component("assistant", f"{a_exp}")
    history_prompt.add_component("user", f"In current timestep, the blood glucose observation is {obs[-1]} (mg/dL).")
    if obs_exp[-1] != "":
        history_prompt.add_component("assistant", f"{obs_exp[-1]}")
    return series, history_prompt


def act_prompt_reprogramming(obs, act, act_exp):
    history_prompt = Conversation()
    for i, (o, a, exp) in enumerate(zip(obs, act, act_exp)):
        history_prompt.add_component("user", f"In timestep {i}, the blood glucose observation is {o} (mg/dL), the agent takes action (Insulin Bolus Dose) {a}.")
        if exp != "":
            history_prompt.add_component("assistant", f"{exp}")
    history_prompt.add_component("user", f"In current timestep, the blood glucose observation is {obs[-1]} (mg/dL), the agent takes action (Insulin Bolus Dose) {act[-1]}.")
    return history_prompt

def summary_reprogramming(obs, act, summary):
    history_prompt = Conversation()
    obs_lst = ",".join(map(str, obs))
    act_lst = ",".join(map(str, act))
    history_prompt.add_component("user", f"In the past timesteps, the blood glucose observation (mg/dL) are {obs_lst}, the action (Insulin Bolus Dose) are {act_lst}. The current version of regular patterns of blood glucose control can be summarized as: {summary}.")
    return history_prompt