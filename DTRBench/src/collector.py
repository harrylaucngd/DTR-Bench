import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast, Optional

import gymnasium as gym
import numpy as np
import torch

from tianshou.data import (
    Batch,
    ReplayBuffer,
    SequenceSummaryStats,
    to_numpy,
)
from tianshou.data.collector import Collector, CollectStats, CollectStatsBase
from tianshou.data.batch import alloc_by_keys_diff
from tianshou.data.types import RolloutBatchProtocol
from tianshou.env import BaseVectorEnv, DummyVectorEnv
from tianshou.policy import BasePolicy
import copy


def get_env_result(data: Batch) -> dict[str, Any]:
    bg = data.obs[:, 0] * 100
    bg_normal = np.logical_and(70 < bg, 140 > bg).mean()
    bg_hypo = (bg < 70).mean()
    bg_hyper = (bg > 180).mean()

    action = data.info["action"]

    return {"bg_normal": bg_normal,
            "bg_hypo": bg_hypo,
            "bg_hyper": bg_hyper,
            "drug_mean": action.mean(),
            "drug_max": action.max(),
            "mortality": np.array(data.terminated == True).any()}


@dataclass(kw_only=True)
class CollectStatsGlucose(CollectStatsBase):
    """A data structure for storing the statistics of rollouts."""

    collect_time: float = 0.0
    """The time for collecting transitions."""
    collect_speed: float = 0.0
    """The speed of collecting (env_step per second)."""
    returns: np.ndarray
    """The collected episode returns."""
    returns_stat: SequenceSummaryStats | None  # can be None if no episode ends during collect step
    """Stats of the collected returns."""
    lens: np.ndarray
    """The collected episode lengths."""
    lens_stat: SequenceSummaryStats | None  # can be None if no episode ends during collect step
    """Stats of the collected episode lengths."""
    bg_normal: SequenceSummaryStats
    """The percentage of normal blood glucose levels."""
    bg_hypo: SequenceSummaryStats
    """The percentage of hypoglycemic blood glucose levels."""
    bg_hyper: SequenceSummaryStats
    """The percentage of hyperglycemic blood glucose levels."""
    drug_mean: SequenceSummaryStats
    """The mean drug dose."""
    drug_max: SequenceSummaryStats
    """The maximum drug dose."""
    mortality: float = 0.0


class GlucoseCollector(Collector):
    """ Same Collector but including more visualizations to loggers for the Glucose environment. """

    def __init__(self, policy: BasePolicy, env: gym.Env | BaseVectorEnv, buffer: ReplayBuffer | None = None,
                 preprocess_fn: Callable[..., RolloutBatchProtocol] | None = None,
                 exploration_noise: bool = False) -> None:

        super().__init__(policy, env, buffer, preprocess_fn, exploration_noise)

    def collect(
            self,
            n_step: int | None = None,
            n_episode: int | None = None,
            random: bool = False,
            render: float | None = None,
            no_grad: bool = True,
            gym_reset_kwargs: dict[str, Any] | None = None,
    ) -> CollectStatsGlucose:
        """
        Apart from what is provided in tianshou, return collected obs, actions, reward as time series
        """
        assert not self.env.is_async, "Please use AsyncCollector if using async venv."
        if n_step is not None:
            assert n_episode is None, (
                f"Only one of n_step or n_episode is allowed in Collector."
                f"collect, got n_step={n_step}, n_episode={n_episode}."
            )
            assert n_step > 0
            if n_step % self.env_num != 0:
                warnings.warn(
                    f"n_step={n_step} is not a multiple of #env ({self.env_num}), "
                    "which may cause extra transitions collected into the buffer.",
                )
            ready_env_ids = np.arange(self.env_num)
        elif n_episode is not None:
            assert n_episode > 0
            ready_env_ids = np.arange(min(self.env_num, n_episode))
            self.data = self.data[: min(self.env_num, n_episode)]
        else:
            raise TypeError(
                "Please specify at least one (either n_step or n_episode) "
                "in AsyncCollector.collect().",
            )

        start_time = time.time()

        step_count = 0
        episode_count = 0
        episode_returns: list[float] = []
        episode_lens: list[int] = []
        episode_start_indices: list[int] = []

        all_episode_data: list[Batch] = []
        episode_data: list[Batch] = []

        while True:
            assert len(self.data) == len(ready_env_ids)
            # restore the state: if the last state is None, it won't store
            last_state = self.data.policy.pop("hidden_state", None)

            # get the next action
            if random:
                try:
                    act_sample = [self._action_space[i].sample() for i in ready_env_ids]
                except TypeError:  # envpool's action space is not for per-env
                    act_sample = [self._action_space.sample() for _ in ready_env_ids]
                act_sample = self.policy.map_action_inverse(act_sample)  # type: ignore
                self.data.update(act=act_sample)
            else:
                if no_grad:
                    with torch.no_grad():  # faster than retain_grad version
                        # self.data.obs will be used by agent to get result
                        result = self.policy(self.data, last_state)
                else:
                    result = self.policy(self.data, last_state)
                # update state / act / policy into self.data
                policy = result.get("policy", Batch())
                assert isinstance(policy, Batch)
                state = result.get("state", None)
                if state is not None:
                    policy.hidden_state = state  # save state into buffer
                act = to_numpy(result.act)
                if self.exploration_noise:
                    act = self.policy.exploration_noise(act, self.data)
                self.data.update(policy=policy, act=act)

            # get bounded and remapped actions first (not saved into buffer)
            action_remap = self.policy.map_action(self.data.act)
            # step in env

            obs_next, rew, terminated, truncated, info = self.env.step(
                action_remap,
                ready_env_ids,
            )
            done = np.logical_or(terminated, truncated)

            if len(self.data.obs.shape) == 3:
                data = copy.deepcopy(self.data)
                data.obs = data.obs[:, -1, :]
                episode_data.append(data)  # WARNING: not checked when num_env > 1
            else:
                episode_data.append(self.data)

            self.data.update(
                obs_next=obs_next,
                rew=rew,
                terminated=terminated,
                truncated=truncated,
                done=done,
                info=info,
            )

            if self.preprocess_fn:
                self.data.update(
                    self.preprocess_fn(
                        obs_next=self.data.obs_next,
                        rew=self.data.rew,
                        done=self.data.done,
                        info=self.data.info,
                        policy=self.data.policy,
                        env_id=ready_env_ids,
                        act=self.data.act,
                    ),
                )

            if render:
                self.env.render()
                if render > 0 and not np.isclose(render, 0):
                    time.sleep(render)

            # add data into the buffer
            ptr, ep_rew, ep_len, ep_idx = self.buffer.add(self.data, buffer_ids=ready_env_ids)

            # collect statistics
            step_count += len(ready_env_ids)

            if np.any(done):
                env_ind_local = np.where(done)[0]
                env_ind_global = ready_env_ids[env_ind_local]
                episode_count += len(env_ind_local)
                episode_lens.extend(ep_len[env_ind_local])
                episode_returns.extend(ep_rew[env_ind_local])
                episode_start_indices.extend(ep_idx[env_ind_local])
                # now we copy obs_next to obs, but since there might be
                # finished episodes, we have to reset finished envs first.
                self._reset_env_with_ids(env_ind_local, env_ind_global, gym_reset_kwargs)
                for i in env_ind_local:
                    self._reset_state(i)

                all_episode_data.append(Batch(**get_env_result(Batch.cat(episode_data))))
                episode_data = []

                # remove surplus env id from ready_env_ids
                # to avoid bias in selecting environments
                if n_episode:
                    surplus_env_num = len(ready_env_ids) - (n_episode - episode_count)
                    if surplus_env_num > 0:
                        mask = np.ones_like(ready_env_ids, dtype=bool)
                        mask[env_ind_local[:surplus_env_num]] = False
                        ready_env_ids = ready_env_ids[mask]
                        self.data = self.data[mask]

            self.data.obs = self.data.obs_next

            if (n_step and step_count >= n_step) or (n_episode and episode_count >= n_episode):
                if episode_data:
                    all_episode_data.append(Batch(**get_env_result(Batch.cat(episode_data))))
                break

        # generate statistics
        self.collect_step += step_count
        self.collect_episode += episode_count
        collect_time = max(time.time() - start_time, 1e-9)
        self.collect_time += collect_time

        if n_episode:
            data = Batch(
                obs={},
                act={},
                rew={},
                terminated={},
                truncated={},
                done={},
                obs_next={},
                info={},
                policy={},
            )
            self.data = cast(RolloutBatchProtocol, data)
            self.reset_env()

        return CollectStatsGlucose(
            n_collected_episodes=episode_count,
            n_collected_steps=step_count,
            collect_time=collect_time,
            collect_speed=step_count / collect_time,
            returns=np.array(episode_returns),
            returns_stat=SequenceSummaryStats.from_sequence(episode_returns)
            if len(episode_returns) > 0
            else None,
            lens=np.array(episode_lens, int),
            lens_stat=SequenceSummaryStats.from_sequence(episode_lens)
            if len(episode_lens) > 0
            else None,
            bg_normal=SequenceSummaryStats.from_sequence(np.array([data.bg_normal for data in all_episode_data])),
            bg_hypo=SequenceSummaryStats.from_sequence(np.array([data.bg_hypo for data in all_episode_data])),
            bg_hyper=SequenceSummaryStats.from_sequence(np.array([data.bg_hyper for data in all_episode_data])),
            drug_mean=SequenceSummaryStats.from_sequence(np.array([data.drug_mean for data in all_episode_data])),
            drug_max=SequenceSummaryStats.from_sequence(np.array([data.drug_max for data in all_episode_data])),
            mortality=np.array([data.mortality for data in all_episode_data]).mean()
        )
