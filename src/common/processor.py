from abc import ABCMeta, abstractmethod


class ModelProcessor(metaclass=ABCMeta):

    def __init__(self, *args, **kwargs):
        ...

    @abstractmethod
    def train(self, *args, **kwargs): ...

    @abstractmethod
    def inference(self, *args, **kwargs): ...
