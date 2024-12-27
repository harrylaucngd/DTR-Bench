import os
import signal
import subprocess
import sys
import time
from typing import Optional, List
import ipdb
import requests
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


def start_vllm_server(command: str) -> subprocess.Popen:
    print("Starting vllm server...")
    process = subprocess.Popen(
        ["bash", "-c", command],
        preexec_fn=os.setsid,  # To allow killing the entire process group
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        bufsize=1,
        text=True,
    )
    return process


def wait_for_server(port: int, timeout: int = 120):
    print("Waiting for vllm server to become ready...")
    start_time = time.time()
    while True:
        if time.time() - start_time > timeout:
            raise TimeoutError("vllm server did not start within the specified timeout.")
        try:
            response = requests.get(f"http://localhost:{port}/health")
            if response.status_code == 200:
                print("vllm server is up and running.")
                break
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(2)  # Wait before retrying


def shutdown_server(process: subprocess.Popen):
    """
    Gracefully shuts down the vllm server subprocess.

    Args:
        process (subprocess.Popen): The subprocess running the server.
    """
    print("Shutting down vllm server...")
    try:
        if os.name == "nt":
            process.terminate()
        else:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        process.wait(timeout=30)
        print("vllm server has been terminated.")
    except Exception as e:
        print(f"Error shutting down the server: {e}")


def signal_handler(sig, frame, server_process: subprocess.Popen):
    print("Interrupt received. Shutting down the server...")
    shutdown_server(server_process)
    sys.exit(0)
