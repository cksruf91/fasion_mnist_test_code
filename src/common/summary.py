import time
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class TrainSummary:
    epoch: int = field()
    _time: int = field(default_factory=time.time, repr=False)
    train_acc: float = field(default=0.)
    val_acc: float = field(default=0.)

    _t_loss: float = field(default=0., repr=False)
    _v_loss: float = field(default=0., repr=False)

    _t_step: int = field(default=0, repr=False)
    _v_step: int = field(default=0, repr=False)

    @property
    def time(self): return time.time() - self._time

    @property
    def train_loss(self): return self._t_loss / self._t_step

    @property
    def val_loss(self): return self._v_loss / self._v_step

    def add_loss(self, loss: float):
        self._t_step += 1
        self._t_loss += loss

    def add_val_loss(self, loss: float):
        self._v_step += 1
        self._v_loss += loss

    def __str__(self):
        return (f" time: {self.time: 0.3f}"
                f" loss: {self.train_loss:3.3f} acc: {self.train_acc:3.3f}"
                f" val_loss: {self.val_loss:3.3f} val_acc: {self.val_acc:3.3f}")

    def __repr__(self):
        return str(self)

    def to_dict(self) -> Dict:
        return {
            'time': self.time, 'train_loss': self.train_loss, 'val_loss': self.val_loss,
            'epoch': self.epoch, 'train_acc': self.train_acc, 'val_acc': self.val_acc
        }
