import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import FashionMNIST


class FashionMnistData(Dataset):
    label = {
        0: "T-shirt/top",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle boot"
    }

    def __init__(self, root: str, train: bool, download: bool, device: torch.device):
        self.device = device
        self.mnist = FashionMNIST(
            root=root, train=train, download=download, transform=self._transform
        )

    def __len__(self):
        return len(self.mnist)

    def _transform(self, x: np.ndarray):
        return torch.tensor((x - np.mean(x)) / np.std(x), device=self.device)

    def __getitem__(self, item):
        return self.mnist[item]

    def to_dataloader(self, batch_size: int, shuffle: bool = False, drop_last: bool = False) -> DataLoader:
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
