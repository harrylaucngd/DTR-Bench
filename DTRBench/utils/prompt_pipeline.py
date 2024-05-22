import torch


class Conversation:
    def __init__(self):
        # Initializes an empty conversation list
        self.conversation = []

    # todo: add syntax check
    # todo: add method to_batch
    def add_component(self, role, content):
        # Adds a new component to the conversation
        if role in ["system", "user", "assistant"]:
            self.conversation.append({"role": role, "content": content})
        else:
            raise ValueError("Role must be 'system', 'user', or 'assistant'.")
        
    def append_description(self, description):
        # Concatenates a description in front of the first component's content in the conversation
        if self.conversation:
            self.conversation[0]["content"] = description + " " + self.conversation[0]["content"]
        else:
            raise ValueError("The current conversation is empty.")
    
    def append_question(self, question):
        # Appends a question to the last component's content in the conversation
        if self.conversation:
            self.conversation[-1]["content"] += " " + question
            return self.conversation
        else:
            raise ValueError("The current conversation is empty.")

    def __str__(self):
        # Provides a string representation of the conversation
        return '\n'.join(f'{component["role"]}: {component["content"]}' for component in self.conversation)


def act_prompt_reprogramming(obs, act, act_exp):
    history_prompt = Conversation()
    if (act_exp == []) or all(exp == "" for exp in act_exp):    # first round or self.need_act_explain = False
        for i, (o, a) in enumerate(zip(obs, act)):
            history_prompt.add_component("user", f"In timestep {i}, The blood glucose observation is {o}(unit), the agent takes action {a}. ")
    else:
        for i, (o, a, exp) in enumerate(zip(obs, act, act_exp)):
            history_prompt.add_component("user", f"In timestep {i}, The blood glucose observation is {o}(unit), the agent takes action {a}. Please explain why the agent chose the last action within 100 words: ")
            history_prompt.add_component("assistant", f"{exp}")
    history_prompt.add_component("user", f"In current timestep, The blood glucose observation is {obs[-1]}(unit), the agent takes action {act[-1]}. ")
    return history_prompt


def obs_prompt_reprogramming(obs, act, obs_exp):
    history_prompt = Conversation()
    if (obs_exp == []) or all(exp == "" for exp in obs_exp):    # first round or self.need_obs_explain = False
        for i, (o, a) in enumerate(zip(obs, act)):
            history_prompt.add_component("user", f"In timestep {i}, The blood glucose observation is {o}(unit), the agent takes action {a}, The next blood glucose observation is {obs[i+1]}(unit). ")
    else:
        for i, (o, a, exp) in enumerate(zip(obs, act, obs_exp)):
            history_prompt.add_component("user", f"In timestep {i}, The blood glucose observation is {o}(unit), the agent takes action {a}, The next blood glucose observation is {obs[i+1]}(unit). Please analyze the current state within 100 words: ")
            history_prompt.add_component("assistant", f"{exp}")
    history_prompt.add_component("user", f"In current timestep, The blood glucose observation is {obs[-1]}(unit). ")
    return history_prompt


def q_prompt_reprogramming(obs, act, act_explain, obs_explain):
    series = torch.tensor([])
    history_prompt = Conversation()
    for (o, a) in zip(obs, act):
        series.append(o)
        series.append(a)
    series.append(act[-1])
    history_prompt.add_component("user", f"The explanation for the last action: {act_explain}. The explanation for the current observation: {obs_explain}. ")
    return series, history_prompt