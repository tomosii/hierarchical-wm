import torch
import numpy as np

from torch import nn, Tensor
from omegaconf import DictConfig


class PixelDecoderHead(nn.Module):
    """
    Decodes latent states (deterministic and stochastic) into an image observation using CNNs.

    x ~ p(x|h, z)

    ### Input:
    - Deterministic state (h)
    - Stochastic state (z)

    ### Output:
    - Gaussian distribution (std fixed to 1) of image observation (x)
    """

    def __init__(
        self,
        observation_shape: tuple[int, ...],
        deterministic_state_size: int,
        stochastic_state_size: int,
        config: DictConfig,
    ):
        super().__init__()

        self.observation_shape = observation_shape
        self.deteministic_state_size = deterministic_state_size
        self.stochastic_state_size = stochastic_state_size
        self.config = config

        self.fc = nn.Linear(
            in_features=deterministic_state_size + stochastic_state_size,
            out_features=config.depth * 32,
        )

        self.convs = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=config.depth * 32,
                out_channels=config.depth * 4,
                kernel_size=config.kernel_size,
                stride=config.stride,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=config.depth * 4,
                out_channels=config.depth * 2,
                kernel_size=config.kernel_size,
                stride=config.stride,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=config.depth * 2,
                out_channels=config.depth,
                kernel_size=config.kernel_size + 1,
                stride=config.stride,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=config.depth,
                out_channels=observation_shape[0],
                kernel_size=config.kernel_size + 1,
                stride=config.stride,
            ),
        )

    def forward(
        self,
        deter_h: Tensor,
        stoch_z: Tensor,
    ) -> torch.distributions.Distribution:
        # Concatenate the inputs
        x = torch.cat([deter_h, stoch_z], dim=-1)

        # Pass the inputs through the linear layer
        x: Tensor = self.fc(x)

        # Reshape the output to match the input shape of the CNNs
        # [Batch * Chunk, Channels * Height * Width] -> [Batch * Chunk, Channels, Height, Width]
        x = x.reshape(-1, self.config.depth * 32, 1, 1)

        if self.config.output == "gaussian":
            # Pass the inputs through the transposed CNNs
            # Output mean of the Gaussian distribution
            mean = self.convs(x)

            # Create the Gaussian distribution
            # Variance is fixed to 1
            base_distribution = torch.distributions.Normal(mean, 1)

            # Need each pixel to be a separate distribution
            # Specify that the batch dimension is the first dimension
            distribution = torch.distributions.Independent(
                base_distribution, reinterpreted_batch_ndims=3
            )
            return distribution
        elif self.config.output == "pixel":
            x = self.convs(x)
            return x
        else:
            raise NotImplementedError


class RewardHead(nn.Module):
    """
    Predicts the reward from the deterministic and stochastic states.

    r ~ p(r|h, z)

    ### Input:
    - Deterministic state (h)
    - Stochastic state (z)

    ### Output:
    - Gaussian distribution (std fixed to 1) of reward (r)
    """

    def __init__(
        self,
        deterministic_state_size: int,
        stochastic_state_size: int,
        config: DictConfig,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                in_features=deterministic_state_size + stochastic_state_size,
                out_features=config.hidden_size,
            ),
            nn.ELU(),
            nn.Linear(
                in_features=config.hidden_size,
                out_features=config.hidden_size,
            ),
            nn.ELU(),
            nn.Linear(
                in_features=config.hidden_size,
                out_features=1,
            ),
        )

    def forward(
        self,
        deter_h: Tensor,
        stoch_z: Tensor,
    ) -> torch.distributions.Distribution:
        # Concatenate the inputs
        x = torch.cat([deter_h, stoch_z], dim=-1)

        # Pass the inputs through the linear layer
        mean: Tensor = self.network(x)

        # Create the Gaussian distribution
        # Variance is fixed to 1
        distribution = torch.distributions.Normal(mean, 1)

        return distribution
