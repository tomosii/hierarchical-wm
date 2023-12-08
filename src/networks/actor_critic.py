import torch
from torch import nn
from torch import Tensor

from omegaconf import DictConfig


class Actor(nn.Module):
    """
    Actor network for SAC.
    """

    def __init__(
        self,
        deterministic_state_size: int,
        stochastic_state_size: int,
        action_size: int,
        action_discrete: bool,
        config: DictConfig,
    ):
        super().__init__()

        self.action_discrete = action_discrete
        self.config = config

        self.network = nn.Sequential(
            nn.Linear(
                deterministic_state_size + stochastic_state_size,
                config.hidden_size,
            ),
            nn.ELU(),
            nn.Linear(
                config.hidden_size,
                config.hidden_size,
            ),
            nn.ELU(),
            nn.Linear(
                config.hidden_size,
                action_size,
            ),
            nn.Tanh(),
        )

    def forward(
        self,
        deter_h: Tensor,
        stoch_z: Tensor,
    ) -> torch.Tensor:
        # Concatenate the inputs
        x = torch.cat([deter_h, stoch_z], dim=-1)

        # Pass the inputs through the linear layer
        x = self.network(x)

        if self.action_discrete:
            # Create the categorical distribution
            onehot_distribution = torch.distributions.OneHotCategoricalStraightThrough(
                logits=x
            )
            # Use straight-through trick for discrete actions
            # sample = sample + prob - prob.detach()
            # (Gumbel-Softmax also works but needs hyperparameter tuning)
            action = onehot_distribution.rsample()
        else:
            # Create the Gaussian distribution
            # Variance is fixed to 1
            base_distribution = torch.distributions.Normal(x, 1)

            # Need each dimension to be independent
            distribution = torch.distributions.Independent(
                base_distribution, reinterpreted_batch_ndims=1
            )
            action = distribution.rsample()
        return action


class Critic(nn.Module):
    """
    Critic network for SAC.
    """

    def __init__(
        self,
        deterministic_state_size: int,
        stochastic_state_size: int,
        config: DictConfig,
    ):
        super().__init__()

        self.config = config

        self.network = nn.Sequential(
            nn.Linear(
                deterministic_state_size + stochastic_state_size,
                config.hidden_size,
            ),
            nn.ELU(),
            nn.Linear(
                config.hidden_size,
                config.hidden_size,
            ),
            nn.ELU(),
            nn.Linear(
                config.hidden_size,
                1,
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
        dist = torch.distributions.Normal(mean, 1)

        return dist
