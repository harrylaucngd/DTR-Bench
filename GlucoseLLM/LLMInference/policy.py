import asyncio
from typing import Any, Dict, List, Optional, Union
import gymnasium
from GlucoseLLM.LLMInference.client import VLLMClient, ChatGPTClient
from GlucoseLLM.prompt import SYS_PROMPT, HIDDEN_VIARABLES, OPERATION_GUIDE, get_text_obs, DIRECT_ACT_PROMPT, COT_ACT_PROMPT, text2act
import ipdb
from tianshou.data import Batch


class BaseTextPolicy:
    def __init__(self, client: Union[VLLMClient, ChatGPTClient], action_space: Optional[Any] = None):
        """Initializes the Policy with a provided LLM client."""
        self.client = client
        self.action_space = action_space

    async def forward(self, batch: Batch) -> str:

        if len(batch.obs) != 1:
            raise ValueError("Batch size must be 1 for this policy.")
        obs = get_text_obs(batch)

        message = [
            {"role": "system", "content": SYS_PROMPT},
            {
                "role": "user",
                "content": f"###Observations\n{obs[0]}\n\n### Request\n{DIRECT_ACT_PROMPT}\n\n###Answer\n",
            },
        ]
        # Call the client's send_request method to get the response
        response = await self.client.send_request(message=message)
        act = text2act(response, self.action_space)
        return Batch(act=act, obs=obs[0], response=response)
    
    async def __call__(self, *args, **kwargs):
        return await self.forward(*args, **kwargs)
    
class HiddenVariableTextPolicy(BaseTextPolicy):
    async def forward(self, batch: Batch) -> str:
        if len(batch.obs) != 1:
            raise ValueError("Batch size must be 1 for this policy.")
        obs = get_text_obs(batch)

        message = [
            {"role": "system", "content": SYS_PROMPT+HIDDEN_VIARABLES},
            {
                "role": "user",
                "content": f"###Observations\n{obs[0]}\n\n### Request\n{DIRECT_ACT_PROMPT}\n\n###Answer\n",
            },
        ]
        # Call the client's send_request method to get the response
        response = await self.client.send_request(message=message)
        act = text2act(response, self.action_space)
        ipdb.set_trace()
        return Batch(act=act, obs=obs[0], response=response)

class FullSysTextPolicy(BaseTextPolicy):
    async def forward(self, batch: Batch) -> str:
        if len(batch.obs) != 1:
            raise ValueError("Batch size must be 1 for this policy.")
        obs = get_text_obs(batch)

        message = [
            {"role": "system", "content": SYS_PROMPT+HIDDEN_VIARABLES+OPERATION_GUIDE},
            {
                "role": "user",
                "content": f"### Observations\n{obs[0]}\n\n### Request\n{DIRECT_ACT_PROMPT}\n\n###Answer\n",
            },
        ]
        # Call the client's send_request method to get the response
        response = await self.client.send_request(message=message)
        act = text2act(response, self.action_space)
        return Batch(act=act, obs=obs[0], response=response)

class CoTTextPolicy(BaseTextPolicy):
    async def forward(self, batch: Batch) -> str:
        if len(batch.obs) != 1:
            raise ValueError("Batch size must be 1 for this policy.")
        obs = get_text_obs(batch)

        message = [
            {"role": "system", "content": SYS_PROMPT+HIDDEN_VIARABLES+OPERATION_GUIDE},
            {
                "role": "user",
                "content": f"### Observations\n{obs[0]}\n\n### Request\n{COT_ACT_PROMPT}\n\n###Answer\n",
            },
        ]
        # Call the client's send_request method to get the response
        response = await self.client.send_request(message=message)
        act = text2act(response, self.action_space)
        return Batch(act=act, obs=obs[0], response=response)