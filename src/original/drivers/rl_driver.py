import numpy as np
import torch
import wandb
import gymnasium as gym

from omegaconf import DictConfig
from tqdm import tqdm

from agents.dreamer import DreamerAgent
from utils.transition import Transition
from utils.replay import ReplayBuffer
from utils.image import save_image, save_gif_video
from datasets.mmnist import MovingMNISTDataset

import torchvision
import os
from torch.utils.data import DataLoader
import tqdm


class RLDriver:
    """
    A driver that runs N steps in an environment.
    """

    def __init__(
        self,
        env: gym.Env,
        agent: DreamerAgent,
        buffer: ReplayBuffer,
    ):
        self.env = env
        self.agent = agent
        self.buffer = buffer

    def run(self, max_steps: int = 1000, train_every: int = 5):
        """
        Run environment steps and train until reaching max_steps.
        """
        total_step = 0

        # Fill the replay buffer
        print("Filling the replay buffer...\n")
        with tqdm(total=self.buffer.capacity) as progress_bar:
            obs, info = self.env.reset()
            while not self.buffer.is_full:
                progress_bar.update(1)
                # Select a random action
                action = self.env.action_space.sample()
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                total_step += 1
                transition = Transition(
                    observation=next_obs,
                    action=action,
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                )
                self.buffer.add(transition)
                obs = next_obs
                if terminated or truncated:
                    obs, info = self.env.reset()
        print(f"Replay buffer has {len(self.buffer)} transitions.\n")

        obs, info = self.env.reset()
        env_step = 0
        episode_reward = 0
        episode_rewards = []

        print("Start training...\n")

        prev_deterministic_h = None
        prev_stochastic_z = None
        prev_action = None

        while total_step < max_steps:
            action, env_action, deterministic_h, stochastic_z = self.agent.policy(
                observation=obs,
                prev_deterministic_h=prev_deterministic_h,
                prev_stochastic_z=prev_stochastic_z,
                prev_action=prev_action,
            )

            # Take a step in the environment
            next_obs, reward, terminated, truncated, info = self.env.step(env_action)

            env_step += 1
            total_step += 1

            transition = Transition(
                observation=next_obs,
                action=env_action,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
            )

            # Add transition to the buffer
            self.buffer.add(transition)

            obs = next_obs
            prev_deterministic_h = deterministic_h
            prev_stochastic_z = stochastic_z
            prev_action = action

            episode_reward += reward

            if terminated or truncated:
                # print(
                #     f"Episode finished after {env_step} steps. Total reward: {episode_reward}"
                # )
                obs, info = self.env.reset()
                env_step = 0
                prev_deterministic_h = None
                prev_stochastic_z = None
                prev_action = None
                episode_reward = 0
                episode_rewards.append(episode_reward)

            if total_step % train_every == 0:
                # print(f"Training at step {total_step}.")

                # Get a batch from the buffer
                transitions = self.buffer.sample()

                # Train agent with the batch data
                metrics = self.agent.train(transitions)

            # Print metrics
            if total_step % 1000 == 0:
                print(f"Step {total_step}: {metrics}")
                print(f"Average reward: {np.mean(episode_rewards)}")
                wandb.log(
                    step=total_step,
                    data={
                        "reward": np.mean(episode_rewards),
                    },
                )
                wandb.log(
                    step=total_step,
                    data=metrics,
                )
                episode_rewards = []

        print("Training finished.")
