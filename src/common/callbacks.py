import os

import mlflow
import numpy as np
import torch
import torch.nn as nn
from mlflow.exceptions import MlflowException

from src.common.summary import TrainSummary


class TrainHistory:

    def __init__(self, file: str):
        self.file = file
        if os.path.isfile(self.file):
            with open(self.file, 'a') as f:
                f.write('\n')

    def __call__(self, model: nn.Module, history: TrainSummary):
        with open(self.file, 'a+') as f:
            f.write(str(history) + '\n')


class MlflowLogger:

    def __init__(self, experiment_name: str, model_params: dict, run_name: str = None):
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.model_params = model_params
        self._server_available = True
        self._set_env()
        self.run_id = self._get_run_id()

    def __call__(self, model: nn.Module, history: TrainSummary):
        history = history.to_dict()
        if self._server_available:
            with mlflow.start_run(run_id=self.run_id):
                mlflow.log_metrics(history, step=history['epoch'])

    def __eq__(self, other):
        return "MLFlow" == other

    def _get_run_id(self):
        if self._server_available:
            with mlflow.start_run(run_name=self.run_name) as mlflow_run:
                mlflow.log_params(self.model_params)
                return mlflow_run.info.run_id
        return 'null'

    def _set_server_satus(self, value: bool, message: str = None) -> None:
        if message is not None:
            print(message)
        self._server_available = value

    def _set_env(self) -> None:
        if os.getenv('MLFLOW_TRACKING_URI') is None:
            self._set_server_satus(value=False, message="Environment variable MLFLOW_TRACKING_URI is not exist")
            return None
        try:
            mlflow.set_experiment(self.experiment_name)
        except MlflowException:
            self._set_server_satus(
                value=False, message=f'mlflow ui connection fail, '
                                     f'skip mlflow logging tracking uri: {mlflow.get_tracking_uri()}'
            )

    def log_model(self, model: nn.Module, name: str):
        with mlflow.start_run(run_id=self.run_id):
            mlflow.pytorch.log_model(pytorch_model=model, artifact_path=name)


class ModelCheckPoint:

    def __init__(self, file: str, mf_logger: MlflowLogger = None, save_best: bool = True,
                 monitor: str = 'val_loss', mode: str = 'min'):
        self.file = file
        save_dir = os.path.dirname(self.file)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        self.mf_logger = mf_logger
        self.save_best = save_best
        self.monitor = monitor
        self.mode = mode
        init_values = {'min': np.inf, 'max': -np.inf}
        self.best_score = init_values[mode]

    def __call__(self, model: nn.Module, history: TrainSummary):
        history = history.to_dict()
        val_score = history[self.monitor]
        check_point = self.file.format(**history)

        if not self.save_best:
            self.save_model(model, check_point)
        elif self._best(val_score, self.best_score):
            self.best_score = val_score
            self.save_model(model, check_point)

    def _best(self, val, best):
        if self.mode == 'min':
            return val <= best
        else:
            return val >= best

    def save_model(self, model: nn.Module, file_name: str):
        if self.mf_logger is not None:
            self.mf_logger.log_model(model, "torch_model")
        else:
            torch.save(model.state_dict(), file_name)
