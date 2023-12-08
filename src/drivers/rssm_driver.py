import numpy as np
from tqdm import tqdm
import os
from torch.utils.data import DataLoader, Dataset
import tqdm

from models.world_model import WorldModel
from datasets.mmnist import MovingMNISTDataset


class RSSMDriver:
    def __init__(
        self,
        world_model: WorldModel,
        train_data: DataLoader,
        test_data: DataLoader,
    ):
        self.world_model = world_model
        self.train_data = train_data
        self.test_data = test_data

    def run(
        self,
        epochs: int,
    ):
        print(f"Start training for {epochs} epochs...\n")

        for epoch in range(epochs):
            total_losses = []
            recon_losses = []
            kl_losses = []

            with tqdm.tqdm(self.train_data) as progress_bar:
                for batch in progress_bar:
                    _, _, metrics = self.world_model.train(batch)
                    total_losses.append(metrics["wm_total_loss"])
                    recon_losses.append(metrics["reconstruction_loss"])
                    kl_losses.append(metrics["kl_divergence_loss"])

                    # print loss
                    progress_bar.set_postfix(
                        total_loss=np.mean(total_losses),
                        recon_loss=np.mean(recon_losses),
                        kl_loss=np.mean(kl_losses),
                    )

            print(f"Epoch {epoch}: {np.mean(total_losses)}")

            # Predict video
            self.world_model.imagine_open_loop(batch)

        print("Training finished!")
