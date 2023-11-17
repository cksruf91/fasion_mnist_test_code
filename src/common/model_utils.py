import sys

import numpy as np
from torch import Tensor as Ts


def accuracy(output: Ts, label: Ts) -> float:
    """calculate accuracy
    Args:
        output (torch.tensor): model prediction
        label (torch.tensor): label
    Returns:
        float: accuracy
    """
    total = len(output)
    label_array = np.array(label.cpu())
    output_array = np.array(output.cpu())

    assert len(label_array) == len(output_array)
    match = np.sum(label_array == output_array)
    return match / total


def train_progressbar(total: int, i: int, bar_length: int = 50, prefix: str = '', suffix: str = '') -> None:
    """progressbar
    """
    dot_num = int((i + 1) / total * bar_length)
    dot = '■' * dot_num
    empty = ' ' * (bar_length - dot_num)
    sys.stdout.write(f'\r {prefix} [{dot}{empty}] {i / total * 100:3.2f}% Done {suffix}')
