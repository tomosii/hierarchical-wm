import torch

from torch import nn
from omegaconf import DictConfig
from networks.actor_critic import Actor, Critic


class Behavior(nn.Module):
    def __init__(
        self,
        observation_shape: tuple[int, ...],
        action_size: int,
        action_discrete: bool,
        embedded_observation_size: int,
        deterministic_state_size: int,
        stochastic_state_size: int,
        device: str,
        config: DictConfig,
    ):
        super().__init__()
        self.observation_shape = observation_shape
        self.action_size = action_size
        self.action_discrete = action_discrete
        self.embedded_observation_size = embedded_observation_size
        self.deterministic_state_size = deterministic_state_size
        self.stochastic_state_size = stochastic_state_size
        self.device = device
        self.config = config

        self.actor = Actor(
            deterministic_state_size=deterministic_state_size,
            stochastic_state_size=stochastic_state_size,
            action_size=action_size,
            action_discrete=action_discrete,
            config=config.actor,
        ).to(device)

        self.critic = Critic(
            deterministic_state_size=deterministic_state_size,
            stochastic_state_size=stochastic_state_size,
            config=config.critic,
        ).to(device)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.actor.learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=config.critic.learning_rate
        )

    def train(
        self,
        deterministic_hs: torch.Tensor,
        stochastic_zs: torch.Tensor,
        rewards: torch.Tensor,
    ):
        # Predict value of the state
        value_distributions: torch.distributions.Distribution = self.critic(
            deterministic_hs, stochastic_zs
        )
        values = value_distributions.mean
        # print("values.shape", values.shape)

        discounts = self.config.discount_factor * torch.ones_like(rewards)
        # print("discounts.shape", discounts.shape)

        # Compute the discounted lambda-returns
        lambda_ = self.config.return_lambda
        next_values = values[1:]
        last = next_values[-1]
        targets = rewards[:-1] + discounts[:-1] * next_values * (1 - lambda_)

        outputs = []
        for i in reversed(range(rewards.shape[0] - 1)):
            last = targets[i] + discounts[i] * lambda_ * last
            outputs.append(last)
        returns = torch.stack(list(reversed(outputs)))
        # print("returns.shape", returns.shape)

        actor_loss = -torch.mean(returns)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        value_distributions = self.critic(
            deterministic_hs[:-1].detach(), stochastic_zs[:-1].detach()
        )
        critic_loss = -torch.mean(value_distributions.log_prob(returns.detach()))

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        metrics = {
            "actor_loss": round(actor_loss.item(), 5),
            "critic_loss": round(critic_loss.item(), 5),
        }

        return metrics
