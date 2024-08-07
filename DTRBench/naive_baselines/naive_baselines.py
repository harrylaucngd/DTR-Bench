import gymnasium as gym
from typing import Any, Generic, Literal, Self, TypeVar, cast

from tianshou.data import Batch, ReplayBuffer, to_numpy, to_torch_as
from tianshou.data.types import (
    BatchWithReturnsProtocol,
    ModelOutputBatchProtocol,
    ObsBatchProtocol,
    RolloutBatchProtocol,
)
import numpy as np
from tianshou.policy.base import BasePolicy
from tianshou.policy.base import TTrainingStats


class RandomPolicy(BasePolicy):
    def __init__(self, *, min_act, max_act, action_space: gym.Space):
        super().__init__(action_space=action_space)
        self.min_act = min_act
        self.max_act = max_act

        assert self.min_act < self.max_act
        if isinstance(self.action_space, gym.spaces.Box):
            self.act_shape = self.action_space.shape
            self.env_type = "continuous"
        elif isinstance(self.action_space, gym.spaces.Discrete):
            self.act_shape = (1,)
            self.env_type = "discrete"
        else:
            raise NotImplementedError(f"Action space {self.action_space} not supported.")

    def forward(self, batch, state=None, **kwargs, ):
        batch_size = batch.obs.shape[0]
        if self.env_type == "continuous":
            act = np.random.rand(batch_size, *self.act_shape)
            act = act * (self.max_act - self.min_act) + self.min_act
            result = Batch(act=act, state=None)
            return cast(ModelOutputBatchProtocol, result)

        elif self.env_type == "discrete":
            act = np.random.randint(low=self.min_act, high=self.max_act, size=(batch_size, 1))
            result = Batch(act=act, state=None)
            return cast(ModelOutputBatchProtocol, result)

        else:
            raise NotImplementedError(f"Action space {self.action_space} not supported.")

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
    pass
