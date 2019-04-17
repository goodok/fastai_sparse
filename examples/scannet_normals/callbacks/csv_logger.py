"A `Callback` that saves tracked metrics into a persistent file."
#Contribution from devforfu: https://nbviewer.jupyter.org/gist/devforfu/ea0b3fcfe194dad323c3762492b05cae

from fastprogress.fastprogress import format_time

from fastai.torch_core import *
from fastai.basic_data import DataBunch
from fastai.callback import *
from fastai.basic_train import Learner, LearnerCallback


from time import time

__all__ = ['CSVLoggerIouByClass']


class CSVLoggerIouByClass(LearnerCallback):
    "A `LearnerCallback` that saves history of IoU by classes into CSV `filename`."
    def __init__(self, learn:Learner, cb_iou_mean, class_names=None, filename:str='iou_by_class'): 
        super().__init__(learn)
        self.filename = filename,
        self.path = self.learn.path/f'{filename}.csv'
        self.cb_iou_mean = cb_iou_mean
        self.class_names = class_names

        if self.class_names is None:
            self.class_names = [str(i) for i in range(cb_iou_mean.n_categories)]

    def read_logged_file(self):  
        "Read the content of saved file"
        return pd.read_csv(self.path)

    def on_train_begin(self, **kwargs: Any) -> None:
        "Prepare file with metric names."
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open('w')
        columns = ['epoch', 'datatype', 'mean_iou'] + self.class_names
        self.file.write(','.join(columns) + '\n')
        self.file.flush()

    def on_epoch_end(self, epoch: int, **kwargs: Any) -> bool:
        "Add a line with `epoch` number, `smooth_loss` and `last_metrics`."
        cb = self.cb_iou_mean
        for datatype in ['train', 'valid']:
            d =  cb._d[datatype]
            stats = [str(epoch), datatype, f"{d['iou']:.6f}"]

            iou_per_class = d['iou_per_class']
            stats += [f'{value:.6f}' for value in iou_per_class]

            str_stats = ','.join(stats)
            self.file.write(str_stats + '\n')
            self.file.flush()

    def on_train_end(self, **kwargs: Any) -> None:  
        "Close the file."
        self.file.close()
