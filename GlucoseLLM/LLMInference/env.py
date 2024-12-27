from GlucoseLLM.LLMInference.policy import BaseTextPolicy
from typing import Any, List
import gymnasium as gym
import asyncio


async def run_episode(policy: BaseTextPolicy, env: gym.Env) -> float:
    """Runs a single epoch and returns the total reward."""
    observation = env.reset()
    total_reward = 0

    while True:
        action = await policy(observation)
        next_observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
        observation = next_observation

    return total_reward


def make_env(task, seed, **env_args):
    try:
        import envpool
    except:
        # warnings.warn("envpool not installed, switch to for loop")
        envpool = None
    if envpool is not None:
        env = envpool.make_gymnasium(task, num_envs=1, seed=seed, **env_args)
    else:
        env = gym.make(task, **env_args)
        env.unwrapped.seed(seed)
    return env


async def parallel_testing(policy: BaseTextPolicy, num_seeds: int, num_repeats: int, max_concurrency: int) -> List[float]:
    """Runs parallel testing of the policy on the environment."""
    patient_list = (
        [
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
        ],
    )
    semaphore = asyncio.Semaphore(max_concurrency)
    rewards = []

    async def run_single_test(patient_name, seed: int):
        total_reward = 0
        for _ in range(num_repeats):
            async with semaphore:
                env = make_env(task="SimGlucoseEnv-single-patient", seed=seed, patient_name=patient_name)
                reward = await run_episode(policy, env)
                total_reward += reward
        return total_reward / num_repeats

    tasks = []
    for patient_name in patient_list:
        for seed in range(num_seeds):
            tasks.append(run_single_test(patient_name, seed))

    results = await asyncio.gather(*tasks)
    rewards.extend(results)

    return rewards
