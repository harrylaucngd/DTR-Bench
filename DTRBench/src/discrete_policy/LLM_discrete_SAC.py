import os
import openai
import httpx
from tianshou.data import Batch
from tianshou.policy import BasePolicy
from functools import wraps
from typing import Optional, Union
import torch
from modelscope import Model, snapshot_download
from modelscope import AutoModelForCausalLM, AutoTokenizer
from modelscope.models.nlp.llama2 import Llama2Tokenizer, Llama2Config
from transformers import BitsAndBytesConfig
import numpy as np


class LLMSpeaker:
    def __init__(self, model_type, api_key=None):
        self.model_type = model_type
        if self.model_type not in ["GPT-3.5", "GPT-4", "LLAMA2-70B", "Mistral-7B", "Mistral-8*7B", "Fake_GPT",
                                   "LLAMA2-13B", "Qwen-14B", "pseudo"]:
            raise NotImplementedError("Invalid LLM type")
        self.api_key = api_key
        if self.model_type in ["Fake_GPT"]:
            self.init_Fake_GPT()
        if self.model_type in ["GPT-3.5", "GPT-4"]:
            self.initGPT()
        if self.model_type in ["LLAMA2-7B"]:
            self.init_LLAMA2(7)
        if self.model_type == "LLAMA2-13B":
            self.init_LLAMA2(13)
        if self.model_type in ["LLAMA2-70B"]:
            self.init_LLAMA2(70)
        if self.model_type in ["Qwen1.5-7B"]:
            self.init_Qwen1dot5(7)
        if self.model_type in ["Qwen1.5-14B"]:
            self.init_Qwen1dot5(14)
        if self.model_type in ["Qwen1.5-72B"]:
            self.init_Qwen1dot5(72)
        if self.model_type in ["Mistral-7B"]:
            self.init_Mistral_7B()
        if self.model_type in ["Mistral-8*7B"]:
            self.init_Mistral_8_7B()
        if self.model_type in ["pseudo"]:
            self.init_pseudo()

    def init_Qwen1dot5(self, size, custom_cache_dir=None):
        if size not in [7, 14, 72]:
            raise ValueError("Invalid Qwen size, should be 7, 14, or 72.")
        custom_cache_dir = f'../qwen1.5-{size}b-chat' if custom_cache_dir is None else custom_cache_dir
        model = AutoModelForCausalLM.from_pretrained(f"qwen/Qwen1.5-{size}B-Chat", device_map="auto",
                                                     cache_dir=custom_cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(f"qwen/Qwen1.5-{size}B-Chat", cache_dir=custom_cache_dir)
        # todo: int8 quantization?
        self.tokenizer = tokenizer
        self.model = model

    def init_LLAMA2(self, size, custom_cache_dir=None):

        if size not in [7, 13, 72]:
            raise ValueError("Invalid LLAMA2 size, should be 7, 13, or 72.")
        custom_cache_dir = f"modelscope/Llama-2-{size}b-chat-ms" if custom_cache_dir is None else custom_cache_dir
        os.environ["MODELSCOPE_CACHE_DIR"] = custom_cache_dir

        model_dir = snapshot_download(model_id=f"modelscope/Llama-2-{size}b-chat-ms",
                                      cache_dir=custom_cache_dir, revision='v1.0.2', ignore_file_pattern=[r'.+\.bin$'])
        tokenizer = Llama2Tokenizer.from_pretrained(model_dir)
        model_config = Llama2Config.from_pretrained(model_dir)
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            load_in_4bit=False,
            # bnb_4bit_compute_dtype=torch.float32,
            bnb_4bit_quant_type='nf4', do_sample=True)
        model = Model.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            config=model_config,
            device_map='auto',
            quantization_config=quantization_config)
        self.tokenizer = tokenizer
        self.model = model

    def init_Mistral_7B(self):
        return NotImplementedError("Mistral-7B is not supported yet")

    def init_Mistral_8_7B(self):
        return NotImplementedError("Mistral-8*7B is not supported yet")

    def init_Fake_GPT(self):
        self.client = openai.OpenAI(
            base_url="https://oneapi.xty.app/v1",
            api_key=self.api_key,
            http_client=httpx.Client(
                base_url="https://oneapi.xty.app/v1",
                follow_redirects=True,
            ))

    def initGPT(self):
        with open('GPT_RL/apikey.txt', 'r', encoding='utf-8') as file:
            api_string = file.read()
        openai.api_key = api_string
        openai.api_base = "https://api.openai.com/v1"
        print("GPT-3.5 API key loaded")

    def init_pseudo(self):
        self.tokenizer = None

        def model_fn(*args, **kwargs):
            return "I am a pseudo model. I am not a real model. I am just a placeholder."

        self.model = model_fn

    def talk(self, prompt, temp, top_p):
        if self.model_type in ["pseudo"]:
            return self.model()

        if self.model_type in ["Fake_GPT"]:
            completion = self.client.chat.completions.create(
                model="gpt-3.5-turbo-16k",
                messages=[{"role": "user", "content": prompt}],
                temperature=temp,
                top_p=top_p)
            response = completion.choices[0].message.content.strip()
            # token_count = completion.usage.total_tokens
            return response  # , token_count

        if self.model_type in ["GPT-3.5"]:
            chat_completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k",
                messages=[{"role": "user", "content": prompt}],
                temperature=temp,
                top_p=top_p
            )
            response = chat_completion.choices[0].message.content.strip()
            token_count = chat_completion.usage.total_tokens
            return response
        if self.model_type in ["LLAMA2-70B", "LLAMA2-13B", "LLAMA2-7B"]:
            inputs = {'text': prompt, 'max_length': 4096, "do_sample": True, "temperature": temp, "top_p": top_p}
            output = self.model.chat(inputs, self.tokenizer)
            return output['response']
        if self.model_type in ["Mistral-7B"]:
            return NotImplementedError("Mistral-7B is not supported yet")
        if self.model_type in ["Mistral-8*7B"]:
            return NotImplementedError("Mistral-8*7B is not supported yet")

        if self.model_type in ["Qwen-14B"]:
            return NotImplementedError("Qwen-14B is not supported yet")


class printer:
    def __init__(self, func):
        self.func = func

    def __get__(self, instance, owner):
        @wraps(self.func)
        def wrapper(*args, **kwargs):
            result = self.func(instance, *args, **kwargs)  # Note how instance is passed here
            verbose = instance.config.verbose
            if verbose == 2:
                print(f"{self.func.__name__} input: {args}, {kwargs}")
                print(f"{self.func.__name__} output: {result}")
            elif verbose == 1:
                print(f"{self.func.__name__} output: {result}")
            elif verbose == 0:
                pass
            else:
                raise ValueError("verbose should be 0, 1, or 2")
            return result

        return wrapper

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


class LLM_discrete_SAC_Policy(BasePolicy):
    def __init__(self, speaker, config):
        super().__init__()
        self.config = config
        self.speaker = speaker
        self.client = None

        os.makedirs(self.config.log_dir, exist_ok=True)

    def forward(self, batch: Batch, state: Optional[Batch] = None, **kwargs) -> Batch:
        """
        :param batch: a tianshou batch for ALL SAMPLES of a patient
        :param state: a tianshou.data.Batch of interaction history. Including obs_explain, action_explain, and llm_act
        """
        # check batch size
        if len(batch) != 1:
            raise ValueError("Batch size must be 1, since LLM processes one patient at a time")
        assert np.unique(batch.info['patientID']).shape[0] == 1

        batch = batch[0]  # remove the first dimension
        batch.to_numpy()

        # prepare history, mask out the padding
        batch.obs = batch.obs[batch.mask == 1]
        batch.info = batch.info[batch.mask == 1]
        if state is None:
            assert len(batch.obs) == 1  # only one step
        patient_id = int(batch.info['patientID'][0])

        obs_prompt = self.get_obs_prompt(batch, state, mode="table") + "\n"
        prompt = self.prompt_fn_few_shot(obs_prompt) if self.config.few_shot else self.prompt_fn_zero_shot(
            obs_prompt)

        # get explanation
        if self.config.explain_obs:
            explain_of_obs = self.explain_obs(prompt)
            state.obs_explain = state.obs_explain + [explain_of_obs] if hasattr(state, 'obs_explain') else [
                explain_of_obs]

        # generate response again to include explanation
        obs_prompt = self.get_obs_prompt(batch, state, mode="table") + "\n"
        prompt = self.prompt_fn_few_shot(obs_prompt) if self.config.few_shot else self.prompt_fn_zero_shot(
            obs_prompt)
        response = self.decide(prompt)
        state.llm_act = state.llm_act + [response] if hasattr(state, 'llm_act') else [response]

        if self.config.explain_action:
            explain_of_action = self.explain_action(prompt)
            state.action_explain = state.action_explain + [explain_of_action] if hasattr(state, 'action_explain') else [
                explain_of_action]

        # return Batch(act=self.config.action_mapping_backward[response], state=state)
        return Batch(act=response, state=state)

    def learn(self, batch: Batch, **kwargs):
        raise NotImplementedError("ChatGPTPolicy does not support learn method")

    def map_action(self, action_number):
        return self.config.action_mapping_forward.get(action_number, "Invalid action")

    def get_obs_prompt(self, batch, state=None, mode="pair", name=None):
        """
        Generate state information in a specified format.

        :param batch: A tianshou batch containing observations, demographic info, etc.
        :param state: A batch containing state information such as obs_explain, llm_act, action_explain.
        :param mode: The format of the output. Supported modes are "pair", "table", "diff".
        :param name: Optional name parameter for future use or specific customizations.
        :return: A string formatted according to the specified mode.
        """
        assert mode in ["pair", "table", "diff"], "Invalid mode specified"

        demog_names = [item.decode('utf-8') for sublist in batch.info.demog_name for item in sublist]
        demog_values = [str(bool(value)) for value in np.squeeze(batch.info.demog).tolist()[:2]]
        demog_values += [format(value, '.5g') for value in np.squeeze(batch.info.demog).tolist()[2:]]

        t_series_names = [item.decode('utf-8') for item in batch.info.vital_name[0]]

        prompt = ""
        # 1. provide demographic
        if mode == "table":
            # Table mode header
            prompt += "Demographics names: " + ", ".join(demog_names) + "\n"
            prompt += "Demographics values: " + ", ".join(demog_values) + "\n"
            prompt += "Time-series names: " + ", ".join(t_series_names) + "\n"
        else:
            prompt += "Demographics: " + ", ".join([f"{name}:{value}" for name, value
                                                    in zip(demog_names, demog_values)]) + "\n"
            previous_values = {}
        for i in range(len(batch.obs)):
            hour = (int(np.squeeze(batch.info.step)) - len(batch.obs) + 1 + i) * 4
            t_series_values = [format(value, '.5g') for value in np.squeeze(batch.info.vital[i]).tolist()]

            if mode == "pair":
                time_series_pairs = ", ".join(
                    [f"{name}:{value}" for name, value in zip(t_series_names, t_series_values)])
                prompt += f"Hour {hour}: {time_series_pairs}\n"

            elif mode == "table":
                prompt += f"Hour {hour}: " + ", ".join(t_series_values) + "\n"

            elif mode == "diff":
                # For diff, only include values that have changed or are new
                current_values = {name: value for name, value in zip(t_series_names, t_series_values)}
                diff_pairs = ", ".join([f"{name}:{value}" for name, value in current_values.items() if
                                        name not in previous_values or previous_values[name] != value])
                if diff_pairs or i == 0:  # Always include first hour
                    prompt += f"Hour {hour}: {diff_pairs}\n"
                previous_values = current_values

            # Add explanations if available and not the current hour or if in table mode
            if state is not None and (i < len(batch.obs) - 1 or mode == "table"):
                prompt += f"obs_explain: {state.obs_explain[i]}\n" if hasattr(state, 'obs_explain') and i < len(
                    state.obs_explain) else ""
                if i < len(batch.obs) - 1:  # Exclude current hour for llm_act and action_explain
                    prompt += f"llm_act: {state.llm_act[i]}\n" if hasattr(state, 'llm_act') and i < len(
                        state.llm_act) else ""
                    prompt += f"action_explain: {state.action_explain[i]}\n" if hasattr(state,
                                                                                        'action_explain') and i < len(
                        state.action_explain) else ""

        return prompt

    def prompt_fn_zero_shot(self, state_prompt):
        prompt = (self.config.target_intro_prompt + self.config.feature_intro_prompt + self.config.action_intro_prompt
                  + state_prompt)
        return prompt

    def prompt_fn_few_shot(self, state_prompt):
        prompt = (self.config.target_intro_prompt + self.config.feature_intro_prompt + self.config.action_intro_prompt +
                  self.config.few_shot_prompt + state_prompt)
        return prompt

    @printer
    def explain_obs(self, prompt):
        prompt = prompt + ("\nPlease analyze the current state within 100 words.")
        temp, top_p = 0.2, 0.3
        response = self.speaker.talk(prompt, temp, top_p)
        print(response)
        return "\nAnalysis of the current state: \n" + response

    @printer
    def explain_action(self, prompt):
        prompt = prompt + "\nPlease explain why the doctor chose the last action within 50 words:"
        temp, top_p = 0.2, 0.3
        response = self.speaker.talk(prompt, temp, top_p)
        print(response)
        return "\nExplanation of the last action: \n" + response

    @printer
    def decide(self, prompt):
        prompt = prompt + "\nYour Action: (Choose 1-3 treatment combination codes and nothing else!)"
        temp, top_p = 0.2, 0.1
        return self.speaker.talk(prompt, temp, top_p)

    def retrival(self, prompt):
        # self.config.action_guideline_prompt
        return NotImplementedError("Retrival is not supported yet")
