from typing import Any

import numpy as np
import torch
from torch import Tensor as Ts
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
        self.mnist = FashionMNIST(
            root=root, train=train, download=download, transform=self._transform
        )
        self._gen_mapper()
        self.train = train
        self.device = device

    def _gen_mapper(self):
        self.target_list = self.mnist.targets.unique().cpu().tolist()
        self.target_map = {t: [] for t in self.target_list}

        for i, t in enumerate(self.mnist.targets):
            self.target_map[t.item()].append(i)

    def __len__(self) -> int:
        return len(self.mnist.data)

    def _to_tensor(self, value: Any) -> Ts:
        return torch.tensor(value, device=self.device)

    def _transform(self, x: np.ndarray):
        return torch.tensor((x - np.mean(x)) / np.std(x), device=self.device, dtype=torch.float32)

    def __getitem__(self, index: int):
        anchor, label = self.mnist[index]
        negative = label
        while label == negative:
            negative = np.random.choice(self.target_list, size=1).item()
        neg_idx_list = self.target_map[negative]
        pos_idx_list = self.target_map[label]

        pos = np.random.choice(pos_idx_list, size=1).item()
        neg = np.random.choice(neg_idx_list, size=1).item()
        image1, _ = self.mnist[pos]
        image2, _ = self.mnist[neg]
        return anchor, image1, image2,

    def to_dataloader(self, batch_size: int, shuffle: bool = False, drop_last: bool = False) -> DataLoader:
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
