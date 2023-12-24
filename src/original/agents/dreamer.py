import torch
import random
import numpy as np

from omegaconf import DictConfig
from models.world_model import WorldModel
from utils.transition import Transition, TransitionSequenceBatch
from models.behavior import Behavior


class DreamerAgent:
    def __init__(
        self,
        observation_shape: tuple[int, ...],
        action_size: int,
        action_discrete: bool,
        device: str,
        config: DictConfig,
    ):
        self.observation_shape = observation_shape
        self.action_size = action_size
        self.action_discrete = action_discrete
        self.config = config
        self.device = device

        self.world_model = WorldModel(
            observation_shape=observation_shape,
            action_size=action_size,
            action_discrete=action_discrete,
            embedded_observation_size=config.embedded_observation_size,
            deterministic_state_size=config.deterministic_state_size,
            stochastic_state_size=config.stochastic_state_size,
            device=device,
            config=config.world_model,
        )

        self.behavior = Behavior(
            observation_shape=observation_shape,
            action_size=action_size,
            action_discrete=action_discrete,
            embedded_observation_size=config.embedded_observation_size,
            deterministic_state_size=config.deterministic_state_size,
            stochastic_state_size=config.stochastic_state_size,
            device=device,
            config=config.behavior,
        )

    def train(self, transitions: TransitionSequenceBatch) -> dict:
        metrics = {}

        # Update the world model
        stochastic_zs, deterministic_hs, mets = self.world_model.train(transitions)
        # print("stochastic_posterior_zs.shape", stochastic_zs.shape)
        # print("deterministic_hs.shape", deterministic_hs.shape)

        metrics.update(mets)

        # Imagine next states and rewards using the current policy
        (
            imagined_deter_hs,
            imagined_stoch_zs,
            imagined_rewards,
        ) = self.world_model.imagine(
            stochastic_zs=stochastic_zs,
            deterministic_hs=deterministic_hs,
            horizon=self.config.imagination_horizon,
            actor=self.behavior.actor,
        )

        # Update the policy
        mets = self.behavior.train(
            deterministic_hs=imagined_deter_hs,
            stochastic_zs=imagined_stoch_zs,
            rewards=imagined_rewards,
        )

        metrics.update(mets)

        return metrics

    def policy(
        self,
        observation: torch.Tensor,
        prev_deterministic_h: torch.Tensor = None,
        prev_stochastic_z: torch.Tensor = None,
        prev_action: torch.Tensor = None,
    ) -> tuple[torch.Tensor, np.ndarray, torch.Tensor, torch.Tensor]:
        # if self.action_discrete:
        #     action = random.randint(0, self.action_size - 1)
        # else:
        #     action = np.random.uniform(-1, 1, self.action_size)

        # Create initial state if not provided (at the beginning of the episode)
        if prev_deterministic_h is None:
            prev_deterministic_h = torch.zeros(
                1,
                self.config.deterministic_state_size,
            ).to(self.device)
            prev_stochastic_z = torch.zeros(
                1,
                self.config.stochastic_state_size,
            ).to(self.device)
            prev_action = torch.zeros(
                1,
                self.action_size,
            ).to(self.device)

        # Normalize observation
        obs = torch.as_tensor(observation).float().to(self.device)
        obs = obs / 255.0 - 0.5

        # Encode observations
        embedded_observation = self.world_model.encoder(obs.unsqueeze(0))

        # Predict deterministic state h_t from h_t-1, z_t-1, and a_t-1
        deterministic_h = self.world_model.recurrent_model(
            prev_stochastic_z,
            prev_action,
            prev_deterministic_h,
        )

        # Predict stochastic state z_t using both h_t and o_t
        # (called Posterior because it is after seeing observation)
        stochastic_z_distribution: torch.distributions.Distribution = (
            self.world_model.representation_model(
                embedded_observation,
                deterministic_h,
            )
        )
        stochastic_z = stochastic_z_distribution.sample()

        # Select action
        action: torch.Tensor = self.behavior.actor(
            deterministic_h,
            stochastic_z,
        )
        action = action.detach()

        if self.action_discrete:
            env_action = action.squeeze().argmax().cpu().numpy()
        else:
            env_action = action.squeeze().cpu().numpy()

        return action, env_action, deterministic_h, stochastic_z
