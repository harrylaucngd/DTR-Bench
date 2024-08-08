from typing import (
    Any,
    Dict,
    Sequence,
    Tuple,
    Type,
    Union,
)
from pathlib import Path
import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from GlucoseLLM.model.net import timeLLM
from GlucoseLLM.prompt_utils import Conversation
from GlucoseLLM.prompts import (SYSTEM_PROMPT, ACTOR_INSTRUCTION_PROMPT, SUMMARY_INSTRUCTION_PROMPT,
                                LLM_INFERENCE_INSTRUCTION_PROMPT, get_Q_instruction, get_patient_info_prompt)
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



class LLMPPO(timeLLM):
    pass


def define_llm_ppo():
    pass


