import torch
import numpy as np

from omegaconf import DictConfig
from utils.transition import Transition, TransitionSequenceBatch


class ReplayBuffer:
    def __init__(
        self,
        observation_shape: tuple[int, ...],
        action_size: int,
        action_discrete: bool,
        device: str,
        config: DictConfig,
    ):
        self.device = device
        self.config = config

        # Length of the buffer
        self.capacity: int = config.capacity

        # Batch size for sampling
        self.batch_size: int = config.batch_size

        # Chunk length for sampling
        self.chunk_length: int = config.chunk_length

        # Elements in the buffer
        self.observations = np.empty((self.capacity, *observation_shape))
        self.actions = np.empty((self.capacity, action_size))  # One-hot encoded
        self.rewards = np.empty((self.capacity, 1))
        self.terminateds = np.empty((self.capacity, 1))
        self.truncateds = np.empty((self.capacity, 1))

        # Current position in the buffer
        self.current_index = 0

        self.is_full = False

    def __len__(self):
        return self.capacity if self.is_full else self.current_index

    def add(self, transition: Transition):
        """Add a transition to the buffer."""
        self.observations[self.current_index] = transition.observation
        self.actions[self.current_index] = transition.action
        self.rewards[self.current_index] = transition.reward
        self.terminateds[self.current_index] = transition.terminated
        self.truncateds[self.current_index] = transition.truncated

        self.current_index = (self.current_index + 1) % self.capacity

        if self.current_index == 0:
            self.is_full = True

    def sample(self) -> TransitionSequenceBatch:
        """
        Sample a batch of consecutive transitions from the buffer.

        Each transition sequence has the length of `chunk_length`.
        """

        start_positions = np.random.randint(
            low=0,
            high=self.capacity
            if self.is_full
            else self.current_index - self.chunk_length,
            size=self.batch_size,
        )
        indices = (
            np.array(
                [
                    np.arange(start, start + self.chunk_length)
                    for start in start_positions
                ]
            )
            % self.capacity
        )

        # Convert to tensors and return the batch
        return TransitionSequenceBatch(
            observations=torch.as_tensor(
                self.observations[indices],
                dtype=torch.float,
                device=self.device,
            ),
            actions=torch.as_tensor(
                self.actions[indices],
                dtype=torch.float,
                device=self.device,
            ),
            rewards=torch.as_tensor(
                self.rewards[indices],
                dtype=torch.float,
                device=self.device,
            ),
            terminateds=torch.as_tensor(
                self.terminateds[indices],
                dtype=torch.float,
                device=self.device,
            ),
            truncateds=torch.as_tensor(
                self.truncateds[indices],
                dtype=torch.float,
                device=self.device,
            ),
        )


# TODO
class IOEpisodeReplayBuffer:
    """
    A replay buffer that stores transitions using local npz files.

    Each episode is stored in a seperate file.

    When sampling a batch, it randomly selects an episode and samples a chunk of
    consecutive transitions from that episode.
    """

    def __init__(
        self,
        observation_shape: tuple[int, ...],
        action_size: int,
        device: str,
        config: DictConfig,
    ):
        pass

    def __len__(self):
        pass

    def add(self, transition: Transition):
        """Add a transition to the buffer."""
        pass

    def sample(self) -> TransitionSequenceBatch:
        """
        Sample a batch of consecutive transitions from the buffer.

        Each transition sequence has the length of `chunk_length`.
        """
        pass
