import re
import numpy as np
import torch
from datetime import timedelta
from tianshou.data import Batch

class Conversation:
    def __init__(self):
        # Initialize an empty conversation list
        self._conversation = []

    def get(self):
        return self._conversation

    def add_component(self, role, content):
        # Add a new component to the conversation
        if role in ["system", "user", "assistant"]:
            self._conversation.append({"role": role, "content": content})
            self.syntax_check()
        else:
            raise ValueError("Role must be 'system', 'user', or 'assistant'.")
        
    def insert_component(self, role, content, loc):
        # Insert a new component at the specified location
        if role in ["system", "user", "assistant"]:
            if loc < 0:
                loc = len(self._conversation) + loc + 1
            if loc > len(self._conversation):
                loc = len(self._conversation)
            self._conversation.insert(loc, {"role": role, "content": content})
            self.syntax_check()
        else:
            raise ValueError("Role must be 'system', 'user', or 'assistant'.")

    def append_content(self, additional_content, pos):
        # Append additional content to the content of the element at position pos
        if pos < 0:
            pos = len(self._conversation) + pos
        if 0 <= pos < len(self._conversation):
            self._conversation[pos]["content"] += additional_content
            self.syntax_check()
        else:
            raise IndexError("Position out of range.")

    def syntax_check(self):
        # Check for neighboring roles that are the same in the conversation
        i = 1
        while i < len(self._conversation):
            if self._conversation[i]["role"] == self._conversation[i - 1]["role"]:
                # Append content of the current role to the previous role
                self._conversation[i - 1]["content"] += self._conversation[i]["content"]
                # Remove the current role
                self._conversation.pop(i)
            else:
                i += 1
    
    def count_tokens(self, text, tokenizer):
        # Use LLM tokenizer to detect token overflow
        tokens = tokenizer.encode(text)
        return len(tokens)

    def to_str(self):
        str = '\n'.join(f'{component["role"]}: {component["content"]}' for component in self._conversation)
        return str


def text2act(logits, action_space):
    numbers = re.findall(r'-?\d+\.?\d*', logits)  # todo: make sure it select the correct number with 0.
    numbers = [float(num) for num in numbers]

    if len(numbers) == 0:
        return action_space.sample()

    # always select the first number
    return np.clip(numbers[0], 0, 0.5)