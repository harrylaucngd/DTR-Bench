import re
import numpy as np
import torch
from datetime import timedelta


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


def summary_reprogramming(batch):
    obs = batch.obs
    batch_size = len(obs)
    length = obs.shape[1]

    def adjust_time(datetime_input, min):
        adjusted_time = datetime_input + timedelta(minutes=min)
        return adjusted_time.strftime("%Y-%m-%d %H:%M:%S")

    conversations = []
    for i in range(batch_size):
        time = batch.info["time"][i]
        glucose = obs[:, :, 0][i]
        insulin = obs[:, :, 1][i]

        description = []
        for j in range(length):
            if glucose[j] == -1:
                continue
            if j == 0:
                description.append(f"Time:{adjust_time(time, -(length-1)*5)},insulin:{insulin[0]}. ")
            if j < length - 1:
                description.append(f"Time:{adjust_time(time, -(length-j-1)*5)},glucose:{glucose[j]},insulin:{insulin[j]+1}. ")
            else:
                description.append("Please extract as much information as possible while keeping the answer short. ")
        conversation = Conversation()
        conversation.add_component("user", " ".join(description))
        conversations.append(conversation)
    return conversations


def q_prompt_reprogramming(obs, act, summaries):
    series, history_prompt = [], []
    for o, a, summ in zip(obs, act, summaries):
        obs_tensor = torch.tensor(o)
        act_tensor = torch.tensor(a)
        ser = torch.empty(2 * len(obs_tensor), dtype=obs_tensor.dtype)
        ser[0::2] = obs_tensor
        ser[1::2] = act_tensor
        ser = ser.unsqueeze(1)
        series.append(ser)
        prompt = Conversation()
        if summ == "":
            prompt.add_component("user", "Please generate the expected discounted reward (i.e., Q(s, a))"
                " for each insulin bins in the order of the following insulin dosage bins: [0, 0-0.05,"
                " 0.05-0.1, 0.1-0.15, 0.15-0.2, 0.2-0.25, 0.25-0.3, 0.3-0.35, 0.35-0.4, 0.4-0.45, 0.45-0.5]. ")
        else:
            prompt.add_component("user", f"Extracted information from history is provided: {summ} Please generate"
                " the expected discounted reward (i.e., Q(s, a)) for each insulin bins in the order of the following"
                " insulin dosage bins: [0, 0-0.05, 0.05-0.1, 0.1-0.15, 0.15-0.2, 0.2-0.25, 0.25-0.3, 0.3-0.35, 0.35-0.4,"
                " 0.4-0.45, 0.45-0.5]. ")
        history_prompt.append(prompt)
    return series, history_prompt


def act_prompt_reprogramming(obs, act, summaries):
    series, history_prompt = [], []
    for o, a, summ in zip(obs, act, summaries):
        obs_tensor = torch.tensor(o)
        act_tensor = torch.tensor(a)
        ser = torch.empty(2 * len(obs_tensor), dtype=obs_tensor.dtype)
        ser[0::2] = obs_tensor
        ser[1::2] = act_tensor
        ser = ser.unsqueeze(1)
        series.append(ser)
        prompt = Conversation()
        if summ == "":
            prompt.add_component("user", "Please generate the expected discounted reward (i.e., Q(s, a))"
                " for each insulin bins in the order of the following insulin dosage bins: [0, 0-0.05,"
                " 0.05-0.1, 0.1-0.15, 0.15-0.2, 0.2-0.25, 0.25-0.3, 0.3-0.35, 0.35-0.4, 0.4-0.45, 0.45-0.5]. ")
        else:
            prompt.add_component("user", f"Extracted information from history is provided: {summ} Please generate"
                " the expected discounted reward (i.e., Q(s, a)) for each insulin bins in the order of the following"
                " insulin dosage bins: [0, 0-0.05, 0.05-0.1, 0.1-0.15, 0.15-0.2, 0.2-0.25, 0.25-0.3, 0.3-0.35, 0.35-0.4,"
                " 0.4-0.45, 0.45-0.5]. ")
        history_prompt.append(prompt)
    return series, history_prompt


def obs2text(batch):
    obs = batch.obs
    length = obs.shape[1]
    time = batch.info["time"][0]
    glucose = obs[:, :, 0][0]
    insulin = obs[:, :, 1][0]

    def adjust_time(datetime_input, min):
        adjusted_time = datetime_input + timedelta(minutes=min)
        return adjusted_time.strftime("%Y-%m-%d %H:%M:%S")

    descriptions = []
    for i in range(length):
        if glucose[i] == -1:
            continue
        if i == 0:
            descriptions.append(f"Time:{adjust_time(time, -(length-1)*5)},insulin:{insulin[0]}. ")
        if i < length - 1:
            descriptions.append(f"Time:{adjust_time(time, -(length-i-1)*5)},glucose:{glucose[i]},insulin:{insulin[i]+1}. ")
        else:
            descriptions.append("Please determine the current insulin dosage, giving a number in 0-0.5,"
                                " without anything else. ")
            descriptions.append(f"Current time: {adjust_time(time, 0)},glucose:{glucose[i]}, insulin:")
    return " ".join(descriptions)


def text2act(logits, action_space):
    numbers = re.findall(r'-?\d+\.?\d*', logits)  # todo: make sure it select the correct number with 0.
    numbers = [float(num) for num in numbers]

    if len(numbers) == 0:
        return action_space.sample()

    # always select the first number
    return np.clip(numbers[0], 0, 0.5)