from torch import Tensor
from dataclasses import dataclass


@dataclass
class Transition:
    """
    A transition data at a single timestep.
    - observation: (*observation_shape)
    - action: (action_size,)
    - reward: (1,)
    - terminated: (1,)
    - truncated: (1,)
    """

    observation: Tensor | None
    action: Tensor | None
    reward: Tensor | None
    terminated: Tensor | None
    truncated: Tensor | None


@dataclass
class TransitionSequenceBatch:
    """
    A batch of transitions.
    - observations: (batch_size, chunk_length, *observation_shape)
    - actions: (batch_size, chunk_length, action_size)
    - rewards: (batch_size, chunk_length, 1)
    - terminateds: (batch_size, chunk_length, 1)
    - truncateds: (batch_size, chunk_length, 1)
    """

    observations: Tensor | None
    actions: Tensor | None
    rewards: Tensor | None
    terminateds: Tensor | None
    truncateds: Tensor | None

    def __len__(self) -> int:
        return len(self.observations)

    def __str__(self) -> str:
        return (
            f"TransitionSequenceBatch(observations={self.observations.shape},\n"
            f"               actions={self.actions.shape},\n"
            f"               rewards={self.rewards.shape},\n"
            f"               terminateds={self.terminateds.shape},\n"
            f"               truncateds={self.truncateds.shape})"
        )
