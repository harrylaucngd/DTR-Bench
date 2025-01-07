from GlucoseLLM.LLMInference.policy import BaseTextPolicy
from typing import Any, List
import gymnasium as gym
import asyncio
import DTRGym
from tianshou.data import Batch


async def run_episode(policy: BaseTextPolicy, env: gym.Env) -> float:
    observation, info = env.reset()
    records = []
    while True:
        act_batch = await policy(Batch.stack([Batch(obs=observation, info=info)]))
        action = act_batch.act
        next_observation, reward, terminated, truncated, info = env.step(action)
        info["reward"] = reward
        info["drug"] = str(float(action)) 
        info["obs"] = act_batch.obs
        info["response"] = act_batch.response
        info["valid_action"] = act_batch.valid_action.tolist()
        records.append(info)
        if terminated or truncated:
            break
        observation = next_observation

    return records


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
