import argparse
from dataclasses import dataclass, field, fields


@dataclass
class Argument:
    model: str = field(default=None)
    type: str = field(default=None)
    checkpoint: str = field(default=None)
    epoch: int = field(default=None)

    _parser = argparse.ArgumentParser()
    _arg = None

    def __post_init__(self):
        self._parsing()

    def _parsing(self):
        self._parser.add_argument('-m', '--model', default='cls', choices=['cls', 'emb'],
                                  type=str, help='model type')
        self._parser.add_argument('-t', '--type', default='train', choices=['train', 'inference'],
                                  type=str, help='action')
        self._parser.add_argument('-c', '--checkpoint', default=None,
                                  type=str, help='inference checkpoint ignore when "--type=train"')
        self._parser.add_argument('-e', '--epoch', default=10,
                                  type=int, help='epoch for train')
        self._arg = self._parser.parse_args()

        for f in fields(self):
            setattr(self, f.name, getattr(self._arg, f.name))
