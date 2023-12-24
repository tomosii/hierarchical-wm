import torch
import hydra
import gymnasium as gym
import wandb
import os

from torch.utils.data import DataLoader, Dataset, random_split

from omegaconf import DictConfig, OmegaConf
from agents.dreamer import DreamerAgent
from models.world_model import WorldModel
from utils.replay import ReplayBuffer
from drivers.rl_driver import RLDriver
from drivers.rssm_driver import RSSMDriver
from envs.wrappers import (
    ChannelFirstEnv,
    PixelEnv,
    ResizeImageEnv,
    ActionRepeatEnv,
    # BatchEnv,
)
from datasets.mmnist import MovingMNISTDataset

# from envs.dmc import DMCPixelEnv
from envs.space import get_env_spaces


# Use hydra to load configs
@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig):
    # Check device
    if config.device == "cuda":
        if not torch.cuda.is_available():
            print("CUDA is not available. Using CPU instead.")
            config.device = "cpu"
        else:
            print("Using CUDA.")
    elif config.device == "mps":
        if not torch.backends.mps.is_available():
            print("Apple's MPS is not available. Using CPU instead.")
            config.device = "cpu"
        else:
            print("Using Apple's MPS.")
    elif config.device == "cpu":
        print("Using CPU.")
    print()

    print(" Config ".center(20, "="))
    print(OmegaConf.to_yaml(config))

    # ====================================================

    wandb.init(
        project="hierarchical-wm",
        config=dict(config),
    )

    # === Create environment ===
    # env_name = config.environment.name
    # env = ActionRepeatEnv(
    #     gym.make(env_name, render_mode="rgb_array"),
    #     repeat=config.environment.action_repeat,
    # )
    # # env = DMCPixelEnv(domain="cartpole", task="swingup")
    # env = PixelEnv(env)
    # env = ResizeImageEnv(
    #     env, (config.environment.image_width, config.environment.image_height)
    # )
    # env = ChannelFirstEnv(env)

    # obs_shape, action_size, action_discrete = get_env_spaces(env)
    # print(f"< {env_name} >")
    # print(f"observation space: {obs_shape}")
    # print(
    #     f"action space: {action_size} ({'discrete' if action_discrete else 'continuous'})"
    # )

    env = None
    obs_shape = (1, 64, 64)
    # action_size = 1
    # action_discrete = True

    # === Create agent ===
    # agent = DreamerAgent(
    #     observation_shape=obs_shape,
    #     action_size=action_size,
    #     action_discrete=action_discrete,
    #     device=config.device,
    #     config=config.agent,
    # )

    # === Create replay buffer ===
    # buffer = ReplayBuffer(
    #     observation_shape=obs_shape,
    #     action_size=action_size,
    #     action_discrete=action_discrete,
    #     device=config.device,
    #     config=config.replay_buffer,
    # )

    # === Create world model ===
    world_model = WorldModel(
        observation_shape=obs_shape,
        embedded_observation_size=config.agent.embedded_observation_size,
        deterministic_state_size=config.agent.deterministic_state_size,
        stochastic_state_size=config.agent.stochastic_state_size,
        device=config.device,
        config=config.agent.world_model,
    )

    # === Create dataset ===
    data_root = os.path.join(os.getcwd(), "data")
    if config.dataset.name == "mmnist":
        dataset = MovingMNISTDataset(root=data_root)

    else:
        raise NotImplementedError
    train_size = int(len(dataset) * 0.8)
    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, len(dataset) - train_size],
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.dataset.batch_size,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.dataset.batch_size,
    )
    print(" Dataset ".center(20, "="))
    print(f"< {config.dataset.name} >")
    print(f"train dataset: {len(train_dataset)}")
    print(f"test dataset: {len(test_dataset)}")
    print()

    os.makedirs("outputs", exist_ok=True)

    # The driver that runs the training loop
    driver = RSSMDriver(
        world_model=world_model,
        train_data=train_dataloader,
        test_data=test_dataloader,
    )

    # Start training
    driver.run(
        epochs=config.epochs,
    )


if __name__ == "__main__":
    main()
