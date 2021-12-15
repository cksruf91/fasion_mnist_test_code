import numpy as np
import pytorch_model_summary
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST

from functions.callbacks import MlflowLogger, ModelCheckPoint
from model.cnn_model import Net

if __name__ == '__main__':
    print(f"cuda : {torch.cuda.is_available()}")
    lr = 1e-3
    batch_size = 16

    train_dataset = FashionMNIST(
        root='./dataset/FashionMnist', train=True, download=True,
        transform=lambda x: (x - np.mean(x)) / np.std(x)
    )
    test_dataset = FashionMNIST(
        root='./dataset/FashionMnist', train=False, download=True,
        transform=lambda x: (x - np.mean(x)) / np.std(x)
    )

    train_iterator = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_iterator = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    model = Net()
    model.double()

    model_input = next(iter(train_iterator))
    print(
        pytorch_model_summary.summary(
            model, model_input[0], show_input=True
        )
    )

    cross_entropy_loss = torch.nn.CrossEntropyLoss().to(model.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model_params = {"lr": lr, "loss_func": cross_entropy_loss, "batch_size": batch_size}

    mlflow_logger = MlflowLogger("FashionMnist", model_params, run_name='version_0.3')
    model_checkpoint = ModelCheckPoint(
        "result/model_e{epoch:02d}_acc{val_acc:0.3f}.zip", mf_logger=mlflow_logger,
        save_best=True, monitor='val_acc', mode='max'
    )
    callbacks = [model_checkpoint, mlflow_logger]

    model.fit(10, train_iterator, test_iterator, loss_func=cross_entropy_loss, optimizer=optimizer, callback=callbacks)
