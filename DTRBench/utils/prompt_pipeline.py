import torch
from transformers import LlamaTokenizer, GPT2Tokenizer, AutoTokenizer

model_hf = {
    "llama-2-13b": "meta-llama/Llama-2-13b-hf",
    "llama-13b": "huggyllama/llama-13b",
    "llama-3-8b": "meta-llama/Llama-3-8b",
    "llama-2-7b": "meta-llama/Llama-2-7b-hf",
    "llama-7b": "huggyllama/llama-7b",
    "gpt2": "openaicommunity/gpt2"
}


class Conversation:
    def __init__(self):
        # Initializes an empty conversation list
        self.conversation = []

    def add_component(self, role, content):
        # Adds a new component to the conversation
        if role in ["system", "user", "assistant"]:
            self.conversation.append({"role": role, "content": content})
            self.syntax_check()
        else:
            raise ValueError("Role must be 'system', 'user', or 'assistant'.")

    def syntax_check(self):
        # Checks for neighboring roles that are the same in the conversation
        for i in range(1, len(self.conversation)):
            if self.conversation[i]["role"] == self.conversation[i-1]["role"]:
                raise ValueError(f"Syntax error: Consecutive '{self.conversation[i]['role']}' roles found.")
    
    def count_tokens(self, text, llm):
        # todo：use LLM tokenizer
        if ("gpt" in llm) or ("llama" in llm):
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    f'model_hub/{llm}',
                    cache_dir=f'model_hub/{llm}',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = LlamaTokenizer.from_pretrained(
                        f'{model_hf[llm]}',
                        cache_dir=f'model_hub/{llm}',
                        trust_remote_code=True,
                        local_files_only=False
                    )
        else:
            raise ValueError("Unsupported LLM Class!")
        tokens = tokenizer.encode(text)
        return len(tokens)
    
    def to_str(self, context_length, llm):
        # Provides a string representation of the conversation with a cutoff to below context length
        conv_str = '\n'.join(f'{component["role"]}: {component["content"]}' for component in self.conversation)
        tokens = self.count_tokens(conv_str, llm)
        
        if tokens <= context_length:
            return conv_str
        
        for i in range(len(self.conversation)):
            conv_str = '\n'.join(f'{component["role"]}: {component["content"]}' for component in self.conversation[i+1:])
            tokens = self.count_tokens(conv_str, llm)
            if tokens <= context_length:
                return conv_str

        return ""
    

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