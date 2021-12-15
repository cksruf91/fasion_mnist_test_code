import random

import mlflow
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST

if __name__ == '__main__':
    logged_model = 'runs:/9e99e8b2faf24258abf303f991a3d741/torch_model'
    loaded_model = mlflow.pyfunc.load_model(logged_model)

    test_dataset = FashionMNIST(root='./dataset/FashionMnist', train=False, download=True,
                                transform=lambda x: (x - np.mean(x)) / np.std(x))
    test_iterator = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=True)
    idx = random.choice(range(0, len(test_dataset)))

    sample, label = test_dataset[idx]
    image = np.zeros([1, 1, 28, 28])
    image[0, 0, :, :] = sample

    print(loaded_model)
    predict = loaded_model.predict(image)
    print(f"predicted category : {np.argmax(predict, axis=-1)} - Category : {label}")
    plt.imshow(sample)
    plt.show()
