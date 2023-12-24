import torch
import numpy as np

from torch import nn, Tensor
from omegaconf import DictConfig


class PixelEncoder(nn.Module):
    """
    Encodes an image observation into a embedded observation (o) using CNNs.

    o ~ p(o|x)

    ### Input:
    - Image observation (x)

    ### Output:
    - Embedded observation (o)
    """

    def __init__(
        self,
        observation_shape: tuple[int, ...],
        embedded_observation_size: int,
        config: DictConfig,
    ):
        super().__init__()

        self.observation_shape = observation_shape
        self.embedded_observation_size = embedded_observation_size
        self.config = config

        self.convs = nn.Sequential(
            nn.Conv2d(
                in_channels=observation_shape[0],
                out_channels=config.depth,
                kernel_size=config.kernel_size,
                stride=config.stride,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=config.depth,
                out_channels=config.depth * 2,
                kernel_size=config.kernel_size,
                stride=config.stride,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=config.depth * 2,
                out_channels=config.depth * 4,
                kernel_size=config.kernel_size,
                stride=config.stride,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=config.depth * 4,
                out_channels=config.depth * 8,
                kernel_size=config.kernel_size,
                stride=config.stride,
            ),
            nn.ReLU(),
        )

        if self.cnn_output_size != embedded_observation_size:
            print(
                f"Reshaping CNN output from {self.cnn_output_size} to {embedded_observation_size}\n"
            )
            self.fc = nn.Linear(
                in_features=self.cnn_output_size,
                out_features=embedded_observation_size,
            )
        else:
            self.fc = nn.Identity()

    def forward(self, observation: Tensor) -> Tensor:
        # Reshape the input to stack the batch and chunk dimensions and encode all images at once
        # [Batch, Chunk, Channels, Height, Width] -> [Batch * Chunk, Channels, Height, Width]
        x = observation.reshape(-1, *self.observation_shape)

        # Pass the observation through the CNNs
        x: Tensor = self.convs(x)

        # Flatten the values after the first batch dimension
        # [Batch, Channels, Height, Width] -> [Batch, Channels * Height * Width]
        x = x.flatten(start_dim=1)

        # Adjust the shape of the output if necessary
        x = self.fc(x)

        # Reshape the output to make it back into a batch of chunks
        # [Batch * Chunk, Embedded Observation Size] -> [Batch, Chunk, Embedded Observation Size]
        x = x.reshape(*observation.shape[:-3], self.embedded_observation_size)
        return x

    @property
    # Returns the output size of the CNNs
    # For the time being, this uses a dummy input to get the output size
    def cnn_output_size(self) -> int:
        dummy_input = torch.empty(1, *self.observation_shape)
        with torch.no_grad():
            dummy_input = self.convs(dummy_input)
        output_size = np.prod(dummy_input.shape[1:])
        return output_size
