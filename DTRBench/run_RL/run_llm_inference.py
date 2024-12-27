import subprocess
import os
import time
import requests
import sys
import signal
import asyncio
import argparse
from typing import List, Optional

# Importing existing modules
from GlucoseLLM.LLMInference.client import VLLMClient
from GlucoseLLM.LLMInference.policy import BaseTextPolicy
from GlucoseLLM.LLMInference.env import parallel_testing

# Configuration Defaults
DEFAULT_CONDA_SH_PATH = "/mnt/bn/gilesluo000/miniconda3/etc/profile.d/conda.sh"
DEFAULT_CONDA_ENV = "textgrad"
DEFAULT_MODEL_PATH = "/mnt/bn/gilesluo000/pretrained_models/Qwen2.5-Coder-7B-Instruct"
DEFAULT_PORT = 8001
DEFAULT_VLLM_SERVER_TIMEOUT = 120  # seconds
DEFAULT_TEST_TIMEOUT = 3600  # seconds (adjust as needed)
DEFAULT_SEEDS = [1, 2]
DEFAULT_REPEATS = 5
DEFAULT_CONCURRENCY = 4

# Server Command Template (to be formatted with variables)
SERVER_COMMAND_TEMPLATE = """
source {conda_sh_path}
conda activate {conda_env}
export VLLM_RPC_TIMEOUT={vllm_rpc_timeout}
export CUDA_VISIBLE_DEVICES={cuda_visible_devices}
vllm serve {model_path} --port {port} --dtype bfloat16 --tensor-parallel-size 4
"""


def parse_arguments():
    """
    Parses command-line arguments using argparse.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Automate starting vllm server, running tests, and shutting down the server.")

    # Server Configuration
    parser.add_argument(
        "--conda_sh_path", type=str, default=DEFAULT_CONDA_SH_PATH, help=f"Path to conda.sh script (default: {DEFAULT_CONDA_SH_PATH})"
    )
    parser.add_argument(
        "--conda_env", type=str, default=DEFAULT_CONDA_ENV, help=f"Conda environment name to activate (default: {DEFAULT_CONDA_ENV})"
    )
    parser.add_argument(
        "--model_path", type=str, default=DEFAULT_MODEL_PATH, help=f"Path to the pretrained model (default: {DEFAULT_MODEL_PATH})"
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port number for the vllm server (default: {DEFAULT_PORT})")
    parser.add_argument(
        "--vllm_rpc_timeout", type=int, default=100000000, help="Value for VLLM_RPC_TIMEOUT environment variable (default: 100000000)"
    )
    parser.add_argument("--cuda_visible_devices", type=str, default="4,5,6,7", help="CUDA_VISIBLE_DEVICES to set (default: '4,5,6,7')")

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


def start_vllm_server(command: str) -> subprocess.Popen:
    """
    Starts the vllm server as a subprocess.

    Args:
        command (str): The bash command to start the server.

    Returns:
        subprocess.Popen: The subprocess running the server.
    """
    print("Starting vllm server...")
    process = subprocess.Popen(
        ["bash", "-c", command],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid,  # To allow killing the entire process group
        text=True,  # Decode stdout and stderr as strings
    )
    return process


def wait_for_server(port: int, timeout: int = 120):
    """
    Waits until the vllm server is ready by checking the /health endpoint.

    Args:
        port (int): The port where the server is expected to run.
        timeout (int): Maximum time to wait in seconds.

    Raises:
        TimeoutError: If the server does not start within the timeout.
    """
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


async def run_tests(seeds: List[int], num_repeats: int, max_concurrency: int, port: int, output_file: str):
    """
    Orchestrates the entire testing process:
    - Initializes the LLM client and policy.
    - Runs parallel tests.
    - Prints and saves the results.

    Args:
        seeds (List[int]): List of seeds for testing.
        num_repeats (int): Number of repeats per test.
        max_concurrency (int): Maximum number of concurrent tests.
        port (int): Port where the vllm server is running.
        output_file (str): File to save test results.
    """
    # Initialize the VLLMClient
    client = VLLMClient(
        model="gpt-4",  # Adjust if necessary
        api_key=None,  # Assuming no API key is needed for local server
        base_url=f"http://localhost:{port}",
        temperature=1,
        top_p=1.0,
        top_k=-1,
        max_tokens=4096,
        truncate_prompt_tokens=None,
    )

    # Initialize the Policy
    policy = BaseTextPolicy(client=client)

    print("Running parallel tests...")
    rewards: List[float] = await parallel_testing(
        policy=policy, num_seeds=len(seeds), num_repeats=num_repeats, max_concurrency=max_concurrency, seeds=seeds  # Pass the list of seeds
    )

    # Process and display results
    average_reward = sum(rewards) / len(rewards) if rewards else 0
    print(f"Completed {len(rewards)} tests.")
    print(f"Average Reward: {average_reward}")

    # Save the results to a file
    with open(output_file, "w") as f:
        import json

        json.dump({"total_tests": len(rewards), "average_reward": average_reward, "rewards": rewards}, f, indent=4)
    print(f"Test results have been saved to '{output_file}'.")


def main():
    # Parse command-line arguments
    args = parse_arguments()

    # Format the server command with provided arguments
    server_command = SERVER_COMMAND_TEMPLATE.format(
        conda_sh_path=args.conda_sh_path,
        conda_env=args.conda_env,
        vllm_rpc_timeout=args.vllm_rpc_timeout,
        cuda_visible_devices=args.cuda_visible_devices,
        model_path=args.model_path,
        port=args.port,
    )

    # Start the vllm server
    server_process = start_vllm_server(server_command)

    try:
        # Wait for the server to be ready
        wait_for_server(args.port, timeout=args.timeout)

        # Run the tests within the asyncio event loop
        asyncio.run(
            run_tests(
                seeds=args.seeds,
                num_repeats=args.num_repeats,
                max_concurrency=args.max_concurrency,
                port=args.port,
                output_file=args.output_file,
            )
        )

    except Exception as e:
        print(f"An error occurred during testing: {e}")
    finally:
        # Ensure the server is shut down gracefully
        shutdown_server(server_process)


if __name__ == "__main__":
    main()
