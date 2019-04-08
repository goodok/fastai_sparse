"`Learner` support for SparseConvNet"
import math
from matplotlib import pyplot as plt

import fastai.train as _dummy  # needed to connect lr_find and other methods to Learner
from fastai.basic_train import Learner as LearnerBase
from fastai.callback import annealing_exp
from fastai.callbacks.general_sched import TrainingPhase, GeneralScheduler

from .utils import print_trainable_parameters


__all__ = ['Learner', 'SparseModelConfig', '_dummy']


# TODO: move to examples (shapenet and scannet)
class SparseModelConfig():
    def __init__(self,
                 spatial_size=50 * 8 + 8,
                 dimension=3,
                 num_input_features=1,
                 block_reps=1,
                 m=32,
                 num_planes_coeffs=[1, 2, 3, 4, 5],
                 num_planes=None,
                 residual_blocks=False,
                 downsample=[2, 2],
                 bias=False,
                 mode=3,
                 num_classes=50,
                 ):
        """
        Parameters
        ----------
        dimension: int
        reps: int
            Conv block repetition factor
        m: int
            Unet number of features
        num_planes_coeffs: array of int
        num_planes=None:  array of int
            UNet number of features per level
        residual_blocks: bool
        mode: int
            mode == 0 if the input is guaranteed to have no duplicates
            mode == 1 to use the last item at each spatial location
            mode == 2 to keep the first item at each spatial location
            mode == 3 to sum feature vectors sharing one spatial location
            mode == 4 to average feature vectors at each spatial location
        num_input_features: int
        downsample: list
            [filter_size, filter_stride]
        bias: bool
        num_classes_total: int
        """

        self.spatial_size = spatial_size
        self.dimension = dimension
        self.block_reps = block_reps
        self.m = m
        self.num_planes_coeffs = num_planes_coeffs
        self.num_planes = num_planes
        self.residual_blocks = residual_blocks
        self.num_classes = num_classes
        self.num_input_features = num_input_features
        self.downsample = downsample
        self.bias = bias
        self.mode = mode

        if self.num_planes is None:
            self.num_planes = [self.m * i for i in num_planes_coeffs]

    def check_accordance(self, data_config, equal_keys=['num_classes'], sparse_item=None):
        for key in equal_keys:
            v1 = getattr(self, key)
            v2 = getattr(data_config, key)
            assert v1 == v2, f"Key '{key}' is not equal"

        if sparse_item is not None:
            assert sparse_item.data['features'].shape[1] == self.num_input_features

    def __repr__(self) -> str:
        lines = [f"{self.__class__.__name__};"]
        for key in ['spatial_size', 'dimension', 'block_reps', 'm',
                    'num_planes', 'residual_blocks', 'num_classes', 'num_input_features',
                    'mode', 'downsample', 'bias']:
            value = getattr(self, key)
            if value is not None:
                lines.append(f'   {key}: {value}')
        s = '\n'.join(lines)
        return s


class Learner(LearnerBase):

    def fit_annealing_exp(self, epochs, lr=0.1, lr_decay=4e-2, momentum=0.9, simulate=False):
        lr_end = lr * math.exp((1 - epochs) * lr_decay)

        n_iter_batch = len(self.data.train_dl)
        n_iter = n_iter_batch * epochs

        phase = TrainingPhase(length=n_iter, lrs=(lr, lr_end), moms=(momentum, momentum), lr_anneal=annealing_exp)

        if simulate:
            lr_list = [phase.lr_step.step() for i in range(n_iter)]
            plt.plot(lr_list)
            print('lr after first epoch:', lr_list[n_iter_batch], 'lr last:', lr_list[-1])
            return lr_list

        scheduler = GeneralScheduler(self, [phase])

        self.fit(epochs, callbacks=[scheduler])

    def find_callback(self, class_or_name):
        for cb in self.callbacks:
            if isinstance(class_or_name, str):
                is_matched = cb.__class__.__name__ == class_or_name
            else:
                is_matched = isinstance(cb, class_or_name)
            if is_matched:
                return cb

    def print_trainable_parameters(self, max_rows=200, max_colwidth=100):
        print_trainable_parameters(self.model, max_rows=max_rows, max_colwidth=max_colwidth)
