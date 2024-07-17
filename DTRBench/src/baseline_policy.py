import numpy as np
import os
from tianshou.data import Batch
import gymnasium as gym
from pathlib import Path
import pandas as pd
from tianshou.policy import BasePolicy
from typing import Any, Generic, Literal, Self, TypeVar, cast

from tianshou.data import Batch, ReplayBuffer, to_numpy, to_torch_as
from tianshou.data.batch import BatchProtocol
from tianshou.data.types import (
    BatchWithReturnsProtocol,
    ModelOutputBatchProtocol,
    ObsBatchProtocol,
    RolloutBatchProtocol,
)
from tianshou.policy.base import TTrainingStats


class RandomPolicy(BasePolicy):
    def __init__(self, *, min_act, max_act, action_space: gym.Space):
        super().__init__(action_space=action_space)
        self.min_act = min_act
        self.max_act = max_act
        assert self.min_act < self.max_act

    def forward(self, batch, state=None, **kwargs, ):
        batch_size = batch.obs.shape[0]

        act = np.random.rand(batch_size, *self.act_shape)
        act = act * (self.max_act - self.min_act) + self.act_min
        result = Batch(act=act, state=None)
        return cast(ModelOutputBatchProtocol, result)

    def learn(self, batch: RolloutBatchProtocol, *args: Any, **kwargs: Any) -> TTrainingStats:
        raise NotImplementedError("RandomPolicy does not support learning.")


class ConstantPolicy(BasePolicy):
    def __init__(self, *, dose, action_space: gym.Space):
        super().__init__(action_space=action_space)
        self.dose = dose

    def forward(self, batch: Batch, state=None, **kwargs, ):
        batch_size = batch.obs.shape[0]
        act = np.tile(self.dose, (batch_size, 1))
        result = Batch(act=act, state=None)
        return cast(ModelOutputBatchProtocol, result)

    def learn(self, batch: RolloutBatchProtocol, *args: Any, **kwargs: Any) -> TTrainingStats:
        raise NotImplementedError("ConstantPolicy does not support learning.")


class PulsePolicy(BasePolicy):
    def __init__(self, *, dose, interval, action_space: gym.Space):
        super().__init__(action_space=action_space)
        self.dose = dose
        self.interval = interval

    def forward(self, batch: Batch, state=None, **kwargs, ):
        """
        Generate a pulse action with a fixed interval.
        """
        batch_size = batch.obs.shape[0]
        steps = batch.info["step"]
        act = np.zeros((batch_size, 1))
        for i in range(batch_size):
            if steps[i] % self.interval == 0:
                act[i] = self.dose
        result = Batch(act=act, state=None)
        return cast(ModelOutputBatchProtocol, result)

    def learn(self, batch: RolloutBatchProtocol, *args: Any, **kwargs: Any) -> TTrainingStats:
        raise NotImplementedError("PulsePolicy does not support learning.")


# Example usage
if __name__ == "__main__":
    # Define a continuous action space
    action_space_box = gym.spaces.Box(low=np.array([-1.0, -1.0]),
                                      high=np.array([1.0, 1.0]),
                                      dtype=np.float32)

    # Define a discrete action space
    action_space_discrete = gym.spaces.Discrete(5)

    # Test the policies with a continuous action space
    print("Testing with a continuous action space:")
    demo_run_policy(RandomPolicy(action_space_box), action_space_box)
    demo_run_policy(MinPolicy(action_space_box), action_space_box)
    demo_run_policy(MaxPolicy(action_space_box), action_space_box)

    # Test the policies with a discrete action space
    print("\nTesting with a discrete action space:")
    demo_run_policy(RandomPolicy(action_space_discrete), action_space_discrete)
    demo_run_policy(MinPolicy(action_space_discrete), action_space_discrete)
    demo_run_policy(MaxPolicy(action_space_discrete), action_space_discrete)
