import torch

from omegaconf import DictConfig
from networks.encoder import PixelEncoder
from networks.heads import PixelDecoderHead, RewardHead
from networks.rssm import RepresentationModel, TransitionModel, RecurrentModel
from utils.transition import TransitionSequenceBatch
from utils.image import save_image, save_gif_video, save_grid_image


class WorldModel(torch.nn.Module):
    def __init__(
        self,
        observation_shape: tuple[int, ...],
        embedded_observation_size: int,
        deterministic_state_size: int,
        stochastic_state_size: int,
        device: str,
        config: DictConfig,
    ):
        super().__init__()

        self.observation_shape = observation_shape
        self.embedded_observation_size = embedded_observation_size
        self.deterministic_state_size = deterministic_state_size
        self.stochastic_state_size = stochastic_state_size
        self.device = device
        self.config = config

        # Models
        self.encoder = PixelEncoder(
            observation_shape=observation_shape,
            embedded_observation_size=embedded_observation_size,
            config=config.encoder,
        ).to(device)

        self.representation_model = RepresentationModel(
            embedded_observation_size=embedded_observation_size,
            deterministic_state_size=deterministic_state_size,
            stochastic_state_size=stochastic_state_size,
            config=config.representation_model,
        ).to(device)

        self.transition_model = TransitionModel(
            deterministic_state_size=deterministic_state_size,
            stochastic_state_size=stochastic_state_size,
            config=config.transition_model,
        ).to(device)

        self.recurrent_model = RecurrentModel(
            deterministic_state_size=deterministic_state_size,
            stochastic_state_size=stochastic_state_size,
            config=config.recurrent_model,
        ).to(device)

        self.decoder = PixelDecoderHead(
            observation_shape=observation_shape,
            deterministic_state_size=deterministic_state_size,
            stochastic_state_size=stochastic_state_size,
            config=config.decoder,
        ).to(device)

        self.reward_head = RewardHead(
            deterministic_state_size=deterministic_state_size,
            stochastic_state_size=stochastic_state_size,
            config=config.reward_head,
        ).to(device)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            params=self.parameters(), lr=config.learning_rate
        )

    def train(
        self,
        observation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Update the world model using transition data.
        """
        observation = observation.to(self.device)

        # Normalize observation
        observation = observation / 255.0 - 0.5

        # Encode observations
        embedded_observation = self.encoder(observation)

        sequence_length = observation.shape[1]

        prior_z_distributions: list[torch.distributions.Distribution] = []
        posterior_z_distributions: list[torch.distributions.Distribution] = []
        posterior_z_samples: list[torch.Tensor] = []
        deter_hs = []

        # Initial input for recurrent model for each batch
        prev_deter_h = torch.zeros(
            observation.shape[0],
            self.deterministic_state_size,
        ).to(self.device)
        prev_stoch_z = torch.zeros(
            observation.shape[0],
            self.stochastic_state_size,
        ).to(self.device)

        # Iterate over timesteps of a chunk
        for t in range(sequence_length - 1):
            # Predict deterministic state h_t from h_t-1, z_t-1, and a_t-1
            deter_h = self.recurrent_model(
                prev_stoch_z,
                prev_deter_h,
            )

            # Predict stochastic state z_t (gaussian) from h_t without o_t
            # (called Prior because it is before seeing observation)
            prior_stoch_z_distribution = self.transition_model(deter_h)

            # Predict stochastic state z_t using both h_t and o_t
            # (called Posterior because it is after seeing observation)
            posterior_stoch_z_distribution: torch.distributions.Distribution = (
                self.representation_model(
                    embedded_observation[:, t + 1],
                    deter_h,
                )
            )

            # Get reparameterized samples of posterior z
            posterior_stoch_z_sample = posterior_stoch_z_distribution.rsample()

            # Append to list for calculating loss later
            prior_z_distributions.append(prior_stoch_z_distribution)
            posterior_z_distributions.append(posterior_stoch_z_distribution)
            posterior_z_samples.append(posterior_stoch_z_sample)
            deter_hs.append(deter_h)

            # Update previous states
            prev_deter_h = deter_h
            prev_stoch_z = posterior_stoch_z_sample

        # The state list has batch_size * sequence_length elements
        # We can regard them as a single batch of size (batch_size * sequence_length)
        # because we don't use recurrent model hereafter

        flattened_deter_hs = torch.cat(deter_hs, dim=0)
        flattened_posterior_z_samples = torch.cat(posterior_z_samples, dim=0)

        # Get reconstructed images
        reconstructed_images = self.decoder(
            flattened_deter_hs, flattened_posterior_z_samples
        )

        # Reconstruction loss (MSE version)
        reconstruction_loss = torch.nn.functional.mse_loss(
            reconstructed_images,
            # Flatten the observation to match the shape of reconstructed images
            observation[:, 1:].reshape(-1, *observation.shape[-3:]),
        )

        # Calculate KL divergence loss
        # How different is the prior distribution from the posterior distribution
        kl = torch.distributions.kl.kl_divergence(
            posterior_stoch_z_distribution,
            prior_stoch_z_distribution,
        )
        # print("kl.shape", kl.shape)
        # print("kl.mean()", kl.mean())
        kl_divergence_loss = kl.mean()

        total_loss = reconstruction_loss + kl_divergence_loss

        # Update the parameters
        self.optimizer.zero_grad()
        total_loss.backward()
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.parameters(), 100)
        self.optimizer.step()

        metrics = {
            "reconstruction_loss": round(reconstruction_loss.item(), 5),
            "kl_divergence_loss": round(kl_divergence_loss.item(), 5),
            "wm_total_loss": round(total_loss.item(), 5),
            # "reconstructed_images": reconstructed_images,
        }

        return (
            flattened_posterior_z_samples.detach(),
            flattened_deter_hs.detach(),
            metrics,
        )

    def imagine_open_loop(
        self,
        observation: torch.Tensor,
    ):
        observation = observation.to(self.device)
        # print("observation.shape", observation.shape)

        # Normalize observation
        observation = observation / 255.0 - 0.5

        # Encode observations
        embedded_observation = self.encoder(observation)

        # Initial input for recurrent model for each batch
        prev_deter_h = torch.zeros(
            observation.shape[0],
            self.deterministic_state_size,
        ).to(self.device)
        prev_stoch_z = torch.zeros(
            observation.shape[0],
            self.stochastic_state_size,
        ).to(self.device)

        sequence_length = observation.shape[1]

        # Get posteriors using the context frames
        for t in range(self.config.context_length):
            # Predict deterministic state h_t from h_t-1, z_t-1, and a_t-1
            deter_h = self.recurrent_model(
                prev_stoch_z,
                prev_deter_h,
            )

            # Predict stochastic state z_t using both h_t and o_t
            # (called Posterior because it is after seeing observation)
            posterior_stoch_z_distribution: torch.distributions.Distribution = (
                self.representation_model(
                    embedded_observation[:, t],
                    deter_h,
                )
            )

            # Update previous states
            prev_deter_h = deter_h
            prev_stoch_z = posterior_stoch_z_distribution.rsample()

        imagined_deter_hs = []
        imagined_stoch_zs = []
        imagination_length = sequence_length - self.config.context_length
        # print("imagination_length", imagination_length)

        # Predict future frames without observation
        # Initial state is the last posterior state from the context frames
        for t in range(imagination_length):
            deter_h = self.recurrent_model(
                prev_stoch_z,
                prev_deter_h,
            )

            # Predict stochastic state z_t (gaussian) from h_t without o_t
            # (called Prior because it is before seeing observation)
            prior_stoch_z_distribution: torch.distributions.Distribution = (
                self.transition_model(deter_h)
            )

            prior_stoch_z = prior_stoch_z_distribution.rsample()

            prev_deter_h = deter_h
            prev_stoch_z = prior_stoch_z

            imagined_deter_hs.append(deter_h)
            imagined_stoch_zs.append(prior_stoch_z)

        imagined_deter_hs = torch.stack(imagined_deter_hs)
        imagined_stoch_zs = torch.stack(imagined_stoch_zs)

        # Reconstruct frames from imagined states
        reconstructed_images: torch.Tensor = self.decoder(
            imagined_deter_hs,
            imagined_stoch_zs,
        )

        # print("reconstructed_images.shape", reconstructed_images.shape)

        # (batch * seq_length, *observation_shape)
        # -> (batch, seq_length, *observation_shape)
        reconstructed_sequence = reconstructed_images.reshape(
            observation.shape[0],
            imagination_length,
            *observation.shape[-3:],
        )

        # save_image(
        #     reconstructed_images[0],
        #     "open_loop_reconstructed.png",
        # )

        save_gif_video(
            reconstructed_sequence[0],
            "open_loop_reconstructed.gif",
        )

        save_grid_image(
            reconstructed_sequence[0],
            "open_loop_reconstructed_grid.png",
        )

        save_grid_image(
            observation[0, : self.config.context_length],
            "context_grid.png",
        )
