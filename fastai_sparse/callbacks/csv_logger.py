"A `Callback` that saves tracked metrics into a persistent file."
# Contribution from devforfu: https://nbviewer.jupyter.org/gist/devforfu/ea0b3fcfe194dad323c3762492b05cae

import numpy as np
import pandas as pd
from time import time

from torch import Tensor
from fastprogress.fastprogress import format_time

from fastai.basic_train import Learner, LearnerCallback

from ..core import Any, MetricsList, ifnone

__all__ = ['CSVLogger', 'CSVLoggerIouByCategory']


# Fixes
# - flush  on_epoch_end
# - learn.add_time

class CSVLogger(LearnerCallback):
    "A `LearnerCallback` that saves history of metrics while training `learn` into CSV `filename`."

    def __init__(self, learn: Learner, filename: str = 'history'):
        super().__init__(learn)
        self.filename = filename
        self.path = self.learn.path / f'{filename}.csv'

    def read_logged_file(self):
        "Read the content of saved file"
        return pd.read_csv(self.path)

    def on_train_begin(self, **kwargs: Any) -> None:
        "Prepare file with metric names."
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open('w')
        self.file.write(','.join(self.learn.recorder.names) + '\n')
        self.file.flush()

    def on_epoch_end(self, epoch: int, smooth_loss: Tensor, last_metrics: MetricsList, **kwargs: Any) -> bool:
        "Add a line with `epoch` number, `smooth_loss` and `last_metrics`."
        last_metrics = ifnone(last_metrics, [])
        stats = [str(stat) if isinstance(stat, int) else f'{stat:.6f}'
                 for name, stat in zip(self.learn.recorder.names, [epoch, smooth_loss] + last_metrics)]
        if self.learn.add_time:
            stats.append(format_time(time() - self.learn.recorder.start_epoch))
        str_stats = ','.join(stats)
        self.file.write(str_stats + '\n')
        self.file.flush()

    def on_train_end(self, **kwargs: Any) -> None:
        "Close the file."
        self.file.close()


class CSVLoggerIouByCategory(LearnerCallback):
    "A `LearnerCallback` that saves history of IoU by categoried into CSV `filename`."

    def __init__(self, learn: Learner, cb_iou_by_categories, categories_names=None, filename: str = 'iou'):
        super().__init__(learn)
        self.filename = filename,
        self.path = self.learn.path / f'{filename}.csv'
        self.cb_iou_by_categories = cb_iou_by_categories
        self.categories_names = categories_names

        if self.categories_names is None:
            self.categories_names = [str(i) for i in range(cb_iou_by_categories.n_categories)]

    def read_logged_file(self):
        "Read the content of saved file"
        return pd.read_csv(self.path)

    def on_train_begin(self, **kwargs: Any) -> None:
        "Prepare file with metric names."
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open('w')
        columns = ['epoch', 'datatype', 'average'] + self.categories_names
        self.file.write(','.join(columns) + '\n')
        self.file.flush()

    def on_epoch_end(self, epoch: int, **kwargs: Any) -> bool:
        "Add a line with `epoch` number, `smooth_loss` and `last_metrics`."
        cb = self.cb_iou_by_categories
        for datatype in ['train', 'valid']:
            d = cb._d[datatype]
            stats = [str(epoch), datatype, f"{d['average']:.6f}"]

            iou = np.zeros(len(self.categories_names))
            iou_by_categ = d['mean_iou_by_category']
            non_zeros = d['non_zeros']
            iou[non_zeros] = iou_by_categ

            stats += [f'{value:.6f}' for value in iou]

            str_stats = ','.join(stats)
            self.file.write(str_stats + '\n')
            self.file.flush()

    def on_train_end(self, **kwargs: Any) -> None:
        "Close the file."
        self.file.close()
