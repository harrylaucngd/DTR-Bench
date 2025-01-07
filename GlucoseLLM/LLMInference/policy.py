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

        prompt = f"###Observations\n{obs[0]}\n\n### Request\n{DIRECT_ACT_PROMPT}\n\n###Answer\n",
        message = [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user","content": prompt,},
        ]
        # Call the client's send_request method to get the response
        response = await self.client.send_request(message=message)
        act, valid_action = text2act(response, self.action_space)
        return Batch(act=act, obs=obs[0], response=response, valid_action=valid_action)
    
    async def __call__(self, batch: Batch) -> str:
        """Adds retry logic while preserving the conversation history."""
        max_retry = 3
        retry_count = 0
        
        # Construct initial conversation

        while retry_count < max_retry:
            # Forward pass to get the initial or retried response
            result_batch = await self.forward(batch)

            # Check the validity of the action
            if result_batch.valid_action or self.client.temperature == 0:
                return result_batch
            retry_count += 1

        # If max retries exceeded, return the last result
        return result_batch
    
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
        act, valid_action = text2act(response, self.action_space)
        return Batch(act=act, obs=obs[0], response=response, valid_action=valid_action)

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
        act, valid_action = text2act(response, self.action_space)
        return Batch(act=act, obs=obs[0], response=response, valid_action=valid_action)

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
        act, valid_action = text2act(response, self.action_space)
        print("message:", message[-1]["content"])
        print("response:", response)

        return Batch(act=act, obs=obs[0], response=response, valid_action=valid_action)


class MajorityVotingTextPolicy(BaseTextPolicy):
    def __init__(self, client: VLLMClient | ChatGPTClient, action_space: Any | None = None, num_vote = 3):
        super().__init__(client, action_space)
        assert num_vote > 1, "num_vote must be greater than 1"
        assert self.client.temperature > 0, "temperature must be greater than 0"
        self.num_vote = num_vote
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
        responses = []
        for i in range(self.num_vote):
            response = await self.client.send_request(message=message)
            responses.append(f"""### Answer {i+1}
{response}
""")
        responses = "\n".join(responses)
        final_message = [
            {"role": "system", "content": SYS_PROMPT+HIDDEN_VIARABLES+OPERATION_GUIDE},
            {
                "role": "user",
                "content": f"""### Observations\n{obs[0]}

Here are {self.num_vote} thoughts to the treat the same patient at the current time step:
{responses}

Please analyse them critically and decide which one is the best one. Finally, you must choose a dosage value enclosed in answer tags (i.e., <ans> and </ans>]), for example, <ans>0</ans>, without any non-numerical word.
### Your Turn""",
            },
        ]
        final_response = await self.client.send_request(message=final_message)
        act, valid_action = text2act(final_response, self.action_space)
        return Batch(act=act, obs=obs[0], response=response, valid_action=valid_action)