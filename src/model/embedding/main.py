from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pytorch_model_summary
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from torch import Tensor as Ts
from torch.optim import Adam

from src.arguments import Argument
from src.common.callbacks import ModelCheckPoint, MlflowLogger
from src.common.model_utils import train_progressbar, accuracy
from src.common.summary import TrainSummary
from src.model.embedding.iterator import FashionMnistData
from src.model.embedding.model import Net


class EmbeddingModel:
    def __init__(self, arg: Argument, device: str = 'cpu'):
        print(f"device: {device} - cuda available: {torch.cuda.is_available()}")
        self.arg = arg
        self.device = torch.device(device)
        self.loss_func = nn.CrossEntropyLoss()
        self.model = Net(device=self.device)
        self.lr = 5e-5
        self.batch_size = 64
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)

        model_params = {"lr": self.lr, "loss_func": self.loss_func, "batch_size": self.batch_size}
        self.callbacks = [
            MlflowLogger("FashionMnist", model_params, run_name='contrastiveLearning'),
            ModelCheckPoint(
                "result/cl_e{epoch:02d}_acc{val_acc:0.3f}.zip", mf_logger=None,
                save_best=True, monitor='val_acc', mode='max'
            )
        ]

    def _forward(self, anchor: Ts, image1: Ts, image2: Ts) -> Tuple[Ts, Ts, Ts]:
        bs = anchor.size(0)
        anchor = anchor.repeat(2, 1, 1)
        images = torch.concat([image1, image2], dim=0)
        label = torch.tensor([0] * bs, dtype=torch.long)

        emb1 = self.model(anchor).unsqueeze(-1)
        emb2 = self.model(images).unsqueeze(-1)
        pred = torch.matmul(emb1.permute(0, 2, 1), emb2).reshape([bs * 2])
        pred = pred.reshape(-1, bs).T
        loss = self.loss_func(pred, label)
        return loss, pred, label

    def _back_propagation(self, loss: Ts) -> None:
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.model.zero_grad()

    @staticmethod
    def _placeholder(length: int, bs: int):
        pred = torch.empty([length * bs])
        true = torch.empty([length * bs])
        return pred, true

    def train(self):
        train_dataset = FashionMnistData(
            root='./dataset/FashionMnist', train=True, download=True, device=self.device
        ).to_dataloader(batch_size=self.batch_size, shuffle=True, drop_last=True)
        test_dataset = FashionMnistData(
            root='./dataset/FashionMnist', train=False, download=True, device=self.device
        ).to_dataloader(batch_size=self.batch_size, shuffle=False, drop_last=True)

        anchor, image1, image2 = next(iter(train_dataset))
        print(pytorch_model_summary.summary(self.model, image1, show_input=True))

        bs: int = train_dataset.batch_size
        total_step = len(train_dataset) + len(test_dataset) + 1
        step = 0
        for e in range(1, self.arg.epoch + 1):
            summary = TrainSummary(epoch=e + 1)
            y_pred, y_true = self._placeholder(len(train_dataset), bs)
            self.model.train()
            for step, (anchor, image1, image2) in enumerate(train_dataset, start=1):
                if step % max((total_step // 10), 1) == 0:
                    train_progressbar(total_step, step, bar_length=30, prefix=f'train {e:03d}/{self.arg.epoch} epoch')
                loss, pred, label = self._forward(anchor, image1, image2)
                self._back_propagation(loss=loss)
                summary.add_loss(loss=loss.item())
                y_pred[(step - 1) * bs: step * bs] = torch.where((pred[:, 0] > 0.5), 0, 1)
                y_true[(step - 1) * bs: step * bs] = label

            summary.train_acc = accuracy(y_pred, y_true)

            y_pred, y_true = self._placeholder(len(test_dataset), bs)
            self.model.eval()
            with torch.no_grad():
                for v_step, (anchor, image1, image2) in enumerate(test_dataset, start=0):
                    if (v_step + step) % max((total_step // 10), 1) == 0:
                        train_progressbar(total_step, v_step + step, bar_length=30,
                                          prefix=f'test  {e:03d}/{self.arg.epoch} epoch')
                    loss, pred, label = self._forward(anchor, image1, image2)
                    summary.add_val_loss(loss=loss.item())
                    # print(torch.where((pred[:, 0] > 0.5), 1, 0))
                    y_pred[v_step * bs: (v_step + 1) * bs] = torch.where((pred[:, 0] > 0.5), 0, 1)
                    y_true[v_step * bs: (v_step + 1) * bs] = label

            summary.val_acc = accuracy(y_pred, y_true)
            for func in self.callbacks:
                func(self.model, summary)

            print(summary)

    def inference(self):
        encoder = TSNE(n_components=2, metric='cosine', random_state=42, angle=0.8)
        test_dataset = FashionMnistData(
            root='./dataset/FashionMnist', train=False, download=True, device=self.device
        ).mnist
        self.model.eval()
        if self.arg.checkpoint is None:
            raise ValueError('parameter [checkpoint] is not provided')
        self.model.load_state_dict(torch.load(self.arg.checkpoint, map_location=self.device))

        length = len(test_dataset)
        category = []
        print("embedding")
        samples = np.random.choice(length, int(length * 0.1), replace=False)
        with torch.no_grad():
            embedding = torch.empty((len(samples), self.model.output_dim))
            for i, s in enumerate(samples):
                image, label = test_dataset[s]
                category.append(label)
                embedding[i] = self.model(image)

        print("TSNE transform...")
        encoded = encoder.fit_transform(embedding.cpu().numpy())

        plt.scatter(encoded[:, 0], encoded[:, 1], c=category)
        plt.legend()
        plt.show()
