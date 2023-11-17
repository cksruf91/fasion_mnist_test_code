from src.arguments import Argument
from src.model.classification.main import ClsModel
from src.model.embedding.main import EmbeddingModel


class Main:

    def __init__(self):
        self.arg = Argument()
        print(self.arg)

    def run(self):
        if self.arg.model == 'cls':
            self._classification()
        elif self.arg.model == 'emb':
            self._embedding()

    def _classification(self):
        model = ClsModel(self.arg)
        if self.arg.type == 'train':
            model.train()
        elif self.arg.type == 'inference':
            model.inference()

    def _embedding(self):
        model = EmbeddingModel(self.arg)
        if self.arg.type == 'train':
            model.train()
        elif self.arg.type == 'inference':
            model.inference()


if __name__ == '__main__':
    Main().run()
