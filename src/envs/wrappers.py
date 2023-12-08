import gymnasium as gym
import numpy as np
import cv2

from gymnasium.wrappers import PixelObservationWrapper


class PixelEnv(gym.ObservationWrapper):
    """Use image as observation."""

    def __init__(self, env):
        env = PixelObservationWrapper(env)
        super().__init__(env)

    def observation(self, obs):
        image = obs["pixels"]
        return image


class ChannelFirstEnv(gym.ObservationWrapper):
    """Convert [H, W, C] to [C, H, W]."""

    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):
        return np.transpose(obs, (2, 0, 1))


class ResizeImageEnv(gym.ObservationWrapper):
    """Resize image observation."""

    def __init__(self, env, size):
        super().__init__(env)
        self.size = size

    def observation(self, obs):
        image = cv2.resize(obs, self.size)
        return image


class ActionRepeatEnv(gym.Wrapper):
    """Repeat action for n steps."""

    def __init__(self, env, repeat=4):
        super().__init__(env)
        self.repeat = repeat

    def step(self, action):
        total_reward = 0.0
        for _ in range(self.repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


# class BatchEnv
