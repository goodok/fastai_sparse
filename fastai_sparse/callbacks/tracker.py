"A `Callback` that track train and valid time duration."

from warnings import warn

import torch

from fastai.callbacks import TrackerCallback
from fastai.basic_train import Learner

from ..core import Any

__all__ = ['SaveModelCallback']


class SaveModelCallback(TrackerCallback):
    "A `TrackerCallback` that saves the model when monitored quantity is best."

    def __init__(self, learn: Learner, monitor: str = 'val_loss', mode: str = 'auto', every: str = 'improvement', name: str = 'bestmodel', overwrite=True):
        super().__init__(learn, monitor=monitor, mode=mode)
        self.every, self.name, self.overwrite = every, name, overwrite
        if self.every not in ['improvement', 'epoch']:
            warn(f'SaveModel every {self.every} is invalid, falling back to "improvement".')
            self.every = 'improvement'

    def jump_to_epoch(self, epoch: int) -> None:
        try:
            self.learn.load(f'{self.name}_{epoch-1}', purge=False)
            print(f"Loaded {self.name}_{epoch-1}")
        except:
            print(f'Model {self.name}_{epoch-1} not found.')

    def on_epoch_end(self, epoch: int, **kwargs: Any) -> None:
        "Compare the value monitored to its best score and maybe save the model."
        if self.every == "epoch":
            if not self.overwrite:
                self.learn.save(f'{self.name}_{epoch}')
            else:
                self.learn.save(f'{self.name}')
                fn_epoch = self.learn.path / self.learn.model_dir / f'{self.name}_epoch.pth'
                torch.save(epoch, fn_epoch)

        else:  # every="improvement"
            current = self.get_monitor_value()
            if current is not None and self.operator(current, self.best):
                print(f'Better model found at epoch {epoch} with {self.monitor} value: {current}.')
                self.best = current
                self.learn.save(f'{self.name}')

    def on_train_end(self, **kwargs):
        "Load the best model."
        if self.every == "improvement" and (self.learn.path / f'{self.learn.model_dir}/{self.name}.pth').is_file():
            self.learn.load(f'{self.name}', purge=False)
