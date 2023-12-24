import numpy as np
import torch
from PIL import Image
import torchvision


def save_image(image: torch.Tensor, path: str, normalize: bool = True):
    """Save an image to a file."""
    # print(image.max())
    # print(image.min())
    # print(image.shape)

    image = image.detach().cpu().numpy()
    if normalize:
        image = (image + 0.5) * 255
        image = np.clip(image, 0, 255).astype(np.uint8)

    if image.shape[0] == 1:
        image = image.squeeze(0)
    elif image.shape[0] == 3:
        # channel first to channel last
        image = image.transpose(1, 2, 0)

    pil_image = Image.fromarray(image)
    pil_image.save(path)


def save_gif_video(images: torch.Tensor, path: str, normalize: bool = True, fps: int = 5):
    """Save a list of images as a gif video."""

    images = images.detach().cpu().numpy()
    if normalize:
        images = (images + 0.5) * 255
        images = np.clip(images, 0, 255).astype(np.uint8)

    if images.shape[-3] == 1:
        images = images.squeeze(-3)
    elif images.shape[-3] == 3:
        # channel first to channel last
        images = images.transpose(0, 3, 1, 2)

    # print("images.shape", images.shape)

    pil_images = [Image.fromarray(image) for image in images]
    pil_images[0].save(
        path,
        save_all=True,
        append_images=pil_images[1:],
        duration=1000 // fps,
        loop=0,
    )

def save_grid_image(images: torch.Tensor, path: str, normalize: bool = True):
    """Save sequence of images as a grid image."""
    # make grid
    grid = torchvision.utils.make_grid(images, nrow=len(images), padding=2, pad_value=1)

    # save image
    save_image(grid, path, normalize=normalize)