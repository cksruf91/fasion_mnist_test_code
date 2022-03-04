import torch
import torch.nn.functional as F
from torch import nn

from model.model_utils import TorchModelInterface


class Net(TorchModelInterface):

    def __init__(self, device):
        super(Net, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(3, 3), padding=(1, 1))
        self.dropout = nn.Dropout(0.1)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.linear = nn.Linear(14 * 14 * 3, 10)
        self.linear2 = nn.Linear(10, 10)

        self.device = device

        self.double()
        self.apply(self._weights_init)
        self.to(self.device)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28).double()
        x = self.conv2d(x)
        x = self.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        x = self.linear(x)
        x = self.linear2(x)
        return self.softmax(x)

    def _compute_loss(self, data, loss_func, optimizer=None, scheduler=None, train=True):
        image = data[0].to(self.device)
        label = data[1].to(self.device)

        output = self.forward(image)
        loss = loss_func(output, label)

        if train:
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            optimizer.step()

            if scheduler is not None:
                scheduler.step()  # Update learning rate schedule
            self.zero_grad()

        y_hat = torch.argmax(output, dim=-1)

        return loss, label, y_hat
