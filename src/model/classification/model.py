import torch
import torch.nn.functional as F
from torch import nn


class Net(nn.Module):

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

    def predict(self, image):
        return self.forward(image)
