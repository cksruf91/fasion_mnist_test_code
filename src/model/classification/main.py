import random
from typing import Tuple

import pytorch_model_summary
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch import Tensor as Ts

from src.arguments import Argument
from src.common.callbacks import MlflowLogger, ModelCheckPoint
from src.common.model_utils import train_progressbar, accuracy
from src.common.summary import TrainSummary
from src.model.classification.iterator import FashionMnistData
from src.model.classification.model import Net


class ClsModel:

    def __init__(self, arg: Argument, device: str = 'cpu'):
        super().__init__()
        print(f"device: {device} - cuda available: {torch.cuda.is_available()}")
        self.arg = arg
        self.device = torch.device(device)
        self.loss_func = torch.nn.CrossEntropyLoss().to(self.device)
        self.model = Net(device=self.device).double()
        self.lr = 1e-3
        self.batch_size = 16
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        model_params = {"lr": self.lr, "loss_func": self.loss_func, "batch_size": self.batch_size}
        self.callbacks = [
            MlflowLogger("FashionMnist", model_params, run_name='CnnClassification'),
            ModelCheckPoint(
                "result/cnn_cls_e{epoch:02d}_acc{val_acc:0.3f}.zip", mf_logger=None,
                save_best=True, monitor='val_acc', mode='max'
            )
        ]

    def inference(self):
        self.model.eval()
        if self.arg.checkpoint is None:
            raise ValueError('parameter [checkpoint] is not provided')
        self.model.load_state_dict(torch.load(self.arg.checkpoint, map_location=self.device))

        test_dataset = FashionMnistData(
            root='./dataset/FashionMnist', train=False, download=True, device=self.device
        )
        idx = random.choice(range(0, len(test_dataset)))

        sample, label = test_dataset[idx]
        print(self.model)
        predict = self.model(sample.unsqueeze(0))
        pred = torch.argmax(predict, dim=-1).item()
        print(f"predicted category : {test_dataset.label[pred]} - Category : {test_dataset.label[label]}")
        plt.imshow(sample)
        plt.show()

    def _forward(self, image: Ts, label: Ts) -> Tuple[Ts, Ts]:
        pred = self.model(image)
        loss = self.loss_func(pred, label)
        return pred, loss

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
        ).to_dataloader(batch_size=self.batch_size, shuffle=True, drop_last=True)

        model_input = next(iter(train_dataset))
        print(pytorch_model_summary.summary(self.model, model_input[0], show_input=True))

        total_step = len(train_dataset) + len(test_dataset) + 1
        bs: int = train_dataset.batch_size

        for e in range(1, self.arg.epoch + 1):
            step = 1
            summary = TrainSummary(epoch=e + 1)

            pred, true = self._placeholder(len(train_dataset), bs)
            self.model.train()
            for step, (image, label) in enumerate(train_dataset, start=step):
                if step % max((total_step // 10), 1) == 0:
                    train_progressbar(total_step, step, bar_length=30,
                                      prefix=f'train {e:03d}/{self.arg.epoch} epoch')
                output, loss = self._forward(image, label)
                self._back_propagation(loss=loss)
                summary.add_loss(loss=loss.item())
                pred[(step - 1) * bs: step * bs] = torch.argmax(output, dim=-1)
                true[(step - 1) * bs: step * bs] = label

            summary.train_acc = accuracy(pred, true)

            pred, true = self._placeholder(len(test_dataset), bs)
            self.model.eval()
            for v_step, (image, label) in enumerate(test_dataset):
                if (v_step + step) % max((total_step // 10), 1) == 0:
                    train_progressbar(total_step, v_step + step, bar_length=30,
                                      prefix=f'test  {e:03d}/{self.arg.epoch} epoch')
                output, loss = self._forward(image, label)
                summary.add_val_loss(loss=loss.item())
                pred[v_step * bs: (v_step + 1) * bs] = torch.argmax(output, dim=-1)
                true[v_step * bs: (v_step + 1) * bs] = label

            summary.val_acc = accuracy(pred, true)

            for func in self.callbacks:
                func(self.model, summary)

            print(summary)
