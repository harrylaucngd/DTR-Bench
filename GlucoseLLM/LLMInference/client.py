from typing import Optional, List
from openai import AsyncOpenAI, AsyncAzureOpenAI
import ipdb
from pydantic import BaseModel, Field
import json


class VLLMClient:
    def __init__(
        self,
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature=1,
        top_p=1.0,
        top_k=-1,
        max_tokens=4096,
        truncate_prompt_tokens=None,
    ):
        """Initializes the LLM client with OpenAI API key and model."""
        print("api_key: ", api_key)
        print("base_url: ", base_url)

        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=100000000000, max_retries=3)
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.truncate_prompt_tokens = truncate_prompt_tokens

    async def send_request(
        self,
        message: List[dict],
        guided_choice: Optional[list] = None,
        guided_json: BaseModel = None,
        stop=None,
    ) -> str:
        json_str = json.dumps(message)
        json.loads(json_str)  # This will raise an error if JSON is invalid

        guided_json_ = guided_json.model_json_schema() if guided_json is not None else None

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=message,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            extra_body={
                "top_k": self.top_k,
                "guided_choice": guided_choice,
                "stop": stop,
                "guided_json": guided_json_,
                "truncate_prompt_tokens": self.truncate_prompt_tokens,
            },
        )
        ans = response.choices[0].message.content.strip()

        return ans


class ChatGPTClient:

    def __init__(
        self,
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        api_version: Optional[str] = None,
        temperature=1,
        top_p=1.0,
        top_k=-1,
        max_tokens=4096,
    ):
        """Initializes the LLM client with OpenAI API key and model."""
        print("api_key: ", api_key)
        print("base_url: ", base_url)

        self.client = AsyncAzureOpenAI(azure_endpoint=base_url, api_version=api_version, api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_tokens = max_tokens

    # 这个函数处理单个请求，返回单个结果
    async def send_request(
        self,
        message: List[dict],
        guided_choice: Optional[list] = None,
        guided_json: Optional[dict] = None,
        stop=None,
    ) -> str:
        if guided_json != None:
            completion = await self.client.beta.chat.completions.parse(
                model=self.model,
                messages=message,  # 注意这里是一个列表，包含所有的对话消息,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                response_format=guided_json,
            )
            return json.dumps(completion.choices[0].message.parsed.model_dump(mode="json"))

        else:
            completion = await self.client.chat.completions.create(
                model=self.model,
                messages=message,  # 注意这里是一个列表，包含所有的对话消息,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
            )
            return completion.choices[0].message.content.strip()
