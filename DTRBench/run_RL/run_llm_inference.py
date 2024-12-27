import signal
import asyncio
import argparse
from typing import List
import ipdb
import wandb
from tqdm.asyncio import tqdm_asyncio  # For asyncio-compatible progress bar
import pandas as pd
import os
# Importing existing modules
from GlucoseLLM.LLMInference.client import VLLMClient, start_vllm_server, wait_for_server, shutdown_server, signal_handler
from GlucoseLLM.LLMInference.policy import BaseTextPolicy
from GlucoseLLM.LLMInference.env import make_env, run_episode

# Configuration Defaults (Hardcoded)
DEFAULT_CONDA_SH_PATH = "/mnt/bn/gilesluo000/miniconda3/etc/profile.d/conda.sh"
DEFAULT_CONDA_ENV = "textgrad"
DEFAULT_MODEL_PATH = "/mnt/bn/gilesluo000/pretrained_models/Qwen2.5-Coder-7B-Instruct"
DEFAULT_PORT = 8001
DEFAULT_VLLM_SERVER_TIMEOUT = 360  # seconds
DEFAULT_SEEDS = [1, 100, 1000, 10000]
DEFAULT_REPEATS = 5
DEFAULT_CONCURRENCY = 32
DEFAULT_PROJECT = "llm_inference_rl"  # Replace with your default project name
DEFAULT_RUN_NAME = "TestRun"  # Replace with your default run name

# Server Command Template (Hardcoded Conda settings)
SERVER_COMMAND_TEMPLATE = """
source {conda_sh_path}
conda activate {conda_env}
export no_proxy=localhost
export VLLM_RPC_TIMEOUT={vllm_rpc_timeout}
export CUDA_VISIBLE_DEVICES={cuda_visible_devices}
vllm serve {model_path} --port {port} --dtype bfloat16 --tensor-parallel-size 4
"""


def parse_arguments():
    parser = argparse.ArgumentParser(description="Automate starting vllm server, running tests, and shutting down the server.")

    # WandB Configuration
    parser.add_argument("--project", type=str, default=DEFAULT_PROJECT, help=f"WandB project name (default: {DEFAULT_PROJECT})")
    parser.add_argument("--run_name", type=str, default=DEFAULT_RUN_NAME, help=f"WandB run name (default: {DEFAULT_RUN_NAME})")

    # Server Configuration
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help=f"Path to the model (default: {DEFAULT_MODEL_PATH})")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port for the server (default: {DEFAULT_PORT})")
    parser.add_argument("--cuda_visible_devices", type=str, default="4,5,6,7", help="CUDA_VISIBLE_DEVICES for the server (default: '4,5,6,7')")

    # Testing Configuration
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS, help=f"List of seeds for testing (default: {DEFAULT_SEEDS})")
    parser.add_argument("--num_repeats", type=int, default=DEFAULT_REPEATS, help=f"Number of repeats per test (default: {DEFAULT_REPEATS})")
    parser.add_argument(
        "--max_concurrency", type=int, default=DEFAULT_CONCURRENCY, help=f"Maximum number of concurrent tests (default: {DEFAULT_CONCURRENCY})"
    )
    parser.add_argument("--output_file", type=str, default="test_results.json", help="File to save test results (default: 'test_results.json')")
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_VLLM_SERVER_TIMEOUT,
        help=f"Timeout for server to start in seconds (default: {DEFAULT_VLLM_SERVER_TIMEOUT})",
    )

    return parser.parse_args()


async def run_tests(
    model_name,
    temperature,
    max_tokens,
    seeds: List[int],
    num_repeats: int,
    max_concurrency: int,
    port: int,
    output_file: str,
    wandb_run: wandb.run,
):
    # Initialize the VLLMClient
    client = VLLMClient(
        model=model_name,
        api_key="EMPTY",
        base_url=f"http://localhost:{port}/v1",
        temperature=temperature,
        top_p=0.95,
        top_k=-1,
        max_tokens=max_tokens,
        truncate_prompt_tokens=None,
    )

    # Initialize the Policy
    policy = BaseTextPolicy(client=client)

    print("Running parallel tests...")

    results = []
    tasks = []
    patient_list = [
        "adolescent#001",
        "adolescent#002",
        "adolescent#003",
        "adolescent#004",
        "adult#001",
        "adult#002",
        "adult#003",
        "adult#004",
        "child#001",
        "child#002",
        "child#003",
        "child#004",
    ]

    semaphore = asyncio.Semaphore(max_concurrency)

    async def run_single_test(patient_name: str, seed: int, i) -> float:
        episode_dict = {}
        async with semaphore:
            env = make_env(task="SimGlucoseEnv-single-patient", seed=seed, patient_name=patient_name)
            policy.action_space = env.action_space
            result_dict = await run_episode(policy, env)
        episode_dict["patient_name"] = patient_name
        episode_dict["seed"] = seed
        episode_dict["return"] = sum([step_dict["reward"] for step_dict in result_dict])
        episode_dict["len"] = len(result_dict)
        episode_dict["trajectory"] = result_dict
        episode_dict["model_name"] = model_name
        episode_dict["temperature"] = temperature
        episode_dict["max_tokens"] = max_tokens
        episode_dict["i"] = i
        return episode_dict

    # Create tasks for all combinations of patient names and seeds
    for patient_name in patient_list:
        for seed in seeds:
            for i in range(num_repeats):
                tasks.append(run_single_test(patient_name, seed, i))

    # Use tqdm to display progress
    for coro in tqdm_asyncio.as_completed(tasks, desc="Testing", total=len(tasks)):
        result = await coro
        results.append(result)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    # Save the results to a file
    pd.DataFrame(results).to_json(output_file, orient="records", lines=True)


def main():
    args = parse_arguments()

    # Initialize WandB
    wandb_run = wandb.init(project=args.project, name=args.run_name, config=vars(args))

    server_command = SERVER_COMMAND_TEMPLATE.format(
        conda_sh_path=DEFAULT_CONDA_SH_PATH,
        conda_env=DEFAULT_CONDA_ENV,
        vllm_rpc_timeout=100000000,
        cuda_visible_devices=args.cuda_visible_devices,
        model_path=args.model_path,
        port=args.port,
    )
    server_process = start_vllm_server(server_command)
    
    # Register the signal handler for graceful shutdown
    signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame, server_process))
    signal.signal(signal.SIGTERM, lambda sig, frame: signal_handler(sig, frame, server_process))


    # Wait for the server to be ready
    wait_for_server(args.port, timeout=args.timeout)

    # Run the tests within the asyncio event loop
    asyncio.run(
        run_tests(
            model_name=args.model_path,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            seeds=args.seeds,
            num_repeats=args.num_repeats,
            max_concurrency=args.max_concurrency,
            port=args.port,
            output_file=args.output_file,
            wandb_run=wandb_run,
        )
    )


if __name__ == "__main__":
    main()
