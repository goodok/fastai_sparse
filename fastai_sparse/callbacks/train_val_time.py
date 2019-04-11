"A `Callback` that track train and valid time duration."
from typing import Any
from time import time

from fastai.basic_train import LearnerCallback, Learner
from fastai.torch_core import add_metrics, MetricsList

__all__ = ['TimeLogger']


class TimeLogger(LearnerCallback):
    _order = -30  # Needs to run before the recorder
    "A `LearnerCallback` that track train and valid time duration."

    def __init__(self, learn: Learner):
        super().__init__(learn)

    def append_metrics_names(self, names):
        recorder = self.learn.recorder
        if not hasattr(recorder, '_added_met_names'):
            recorder._added_met_names = []
        recorder._added_met_names += names

    def on_train_begin(self, **kwargs: Any) -> None:
        self.append_metrics_names(['train_time', 'valid_time'])

    def on_epoch_begin(self, **kwargs: Any) -> None:
        t = time()
        self.train_start = t
        self.train_end = t
        self.valid_start = t
        self.valid_end = 0.

    def on_batch_begin(self, train, **kwargs):
        t = time()
        if train:
            self.train_end = t
            self.valid_start = t
        else:
            self.valid_end = t

    def on_batch_end(self, train, **kwargs):
        t = time()
        if train:
            self.train_end = t
            self.valid_start = t
        else:
            self.valid_end = t

    def on_epoch_end(self, last_metrics: MetricsList, **kwargs: Any) -> bool:
        train_time = self.train_end - self.train_start
        valid_time = self.valid_end - self.valid_start
        return add_metrics(last_metrics, [train_time, valid_time])
