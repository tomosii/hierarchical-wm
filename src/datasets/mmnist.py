import torch
import torchvision
import os

from torch.utils.data import Dataset, DataLoader


class MovingMNISTDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

        self.data = torchvision.datasets.MovingMNIST(
            root=root,
            download=True,
        )

    def __getitem__(self, index):
        sequence = self.data[index]
        # print(batch.shape)
        return sequence

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    data_root = os.path.join(os.getcwd(), "data")
    print(data_root)

    dataset = MovingMNISTDataset(root=data_root, train=True)
    print(len(dataset))

    loader = DataLoader(
        dataset,
        batch_size=1,
    )

    # get first batch
    data = next(iter(loader))

    print(torch.max(data))

    # save image
    torchvision.utils.save_image(data[0], "test.png")
