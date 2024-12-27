import asyncio
from typing import Any, Dict, List, Optional, Union
import gymnasium
from GlucoseLLM.LLMInference.client import VLLMClient, ChatGPTClient
from GlucoseLLM.prompt import SYS_PROMPT, get_text_obs, ACT_PROMPT, text2act


class BaseTextPolicy:
    def __init__(self, client: Union[VLLMClient, ChatGPTClient], action_space: Optional[Any] = None):
        """Initializes the Policy with a provided LLM client."""
        self.client = client
        self.action_space = action_space

    async def forward(self, obs) -> str:
        """Generates an action based on the observation using the LLM client."""
        # Convert the observation into a message format expected by the client
        if len(obs) != 1:
            raise ValueError("Only one observation should be supported.")

        obs = get_text_obs(obs)

        message = [
            {"role": "system", "content": SYS_PROMPT},
            {
                "role": "user",
                "content": f"###Observations\n{obs[0]}\n\n### Request\n{ACT_PROMPT}\n\n###Answer\n",
            },
        ]
        # Call the client's send_request method to get the response
        response = await self.client.send_request(message=message)

        act = text2act(response, self.action_space)
        return act

    async def __call__(self, *args, **kwargs):
        return await self.forward(*args, **kwargs)
