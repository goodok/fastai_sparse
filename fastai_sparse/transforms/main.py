# -*- coding: utf-8 -*-

import numpy as np
import functools
from typing import List, Optional
from dataclasses import dataclass, field
from copy import copy


from ..data_items import PointsItem, ItemBase
from ..core import listify, _get_default_args

__all__ = ['TRANSFER_KEYS', 'transfer_keys', 'Transform', 'RandTransform', 'Compose', 'log_transforms',
           'sample_points',
           ]


def rand_bool(p: float, size: Optional[List[int]] = None):
    "Draw 1 or shape=`size` random booleans (`True` occuring with probability `p`)."
    # numpy version of fastai.torch_core.rand_bool as transforms works with numpy random state.
    return np.random.uniform(0, 1, size) < p


# default keys that preserves while transformation
TRANSFER_KEYS = ['id', 'random_seed', 'categ', 'class_offset', 'num_classes']


def transfer_keys(src, dest, keys=[]):
    keys = TRANSFER_KEYS + keys
    for key in keys:
        if key in src:
            assert key not in dest, f"'{key}' is already present"
            dest[key] = src[key]


class Transform():
    "Utility class for adding probability and wrapping support to transform `func`."
    # Based on
    #   - https://github.com/renato145/fastai_scans/blob/master/fastai_scans/transform.py
    #   - https://github.com/fastai/fastai/blob/master/fastai/vision/image.py
    #   - https://github.com/fastai/fastai/blob/master/fastai/vision/transform.py

    _wrap = None
    order = 0

    def __init__(self, func, order=None):
        "Create a transform for `func` and assign it an priority `order`"

        if order is not None:
            self.order = order
        self.func = func
        # To remove the _ that begins every transform function.
        self.func.__name__ = func.__name__[1:]
        functools.update_wrapper(self, self.func)
        self.def_args = _get_default_args(func)
        self.params = copy(func.__annotations__)

    def __call__(self, *args, is_random=True, p=1., **kwargs):
        "Calc now if `args` passed; else create a transform called prob `p` if `random`."
        if args:
            return self.calc(*args, **kwargs)
        else:
            return RandTransform(self, kwargs=kwargs, is_random=is_random, p=p)

    def calc(self, x, *args, **kwargs):
        "Apply to image `x`, wrapping it if necessary."
        if self._wrap:
            return getattr(x, self._wrap)(self.func, *args, **kwargs)
        else:
            return self.func(x, *args, **kwargs)

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return f'{self.name} ({self.func.__name__})'


@dataclass
class RandTransform():
    "Wrap `Transform` to add randomized execution."
    tfm: Transform
    kwargs: dict
    p: float = 1.0
    resolved: dict = field(default_factory=dict)
    do_run: bool = True
    is_random: bool = True

    def __post_init__(self):
        functools.update_wrapper(self, self.tfm)

    def resolve(self, tfm_params=None):
        "Bind any random variables in the transform."
        if not self.is_random:
            self.resolved = {**self.tfm.def_args, **self.kwargs}
            return

        self.resolved = {}
        # solve patches tfm
        if tfm_params is not None:
            tfm_params = self.tfm.resolve(tfm_params)
            for k, v in tfm_params.items():
                self.resolved[k] = v

        # for each param passed to tfm...
        for k, v in self.kwargs.items():
            # ...if it's annotated, call that fn...
            if k in self.tfm.params:
                rand_func = self.tfm.params[k]
                self.resolved[k] = rand_func(*listify(v))
            # ...otherwise use the value directly
            else:
                self.resolved[k] = v
        # use defaults for any args not filled in yet
        for k, v in self.tfm.def_args.items():
            if k not in self.resolved:
                self.resolved[k] = v
        if self.p != 1.0:
            # to do not disturb random state
            self.do_run = rand_bool(self.p)
        else:
            self.do_run = True

    @property
    def order(self):
        return self.tfm.order

    def __call__(self, x, *args, **kwargs):
        "Randomly execute our tfm on `x`."
        if self.do_run:
            return self.tfm(x, *args, **{**self.resolved, **kwargs})
        else:
            return x


class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, item):
        if isinstance(item, ItemBase):
            # We suppose that the  Item object knows better how to work with transformations.
            item = item.apply_tfms(self.transforms)
        else:
            for t in self.transforms:
                item = t(item)
        return item

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

    def log(self):
        return log_transforms(self.transforms)


def log_transforms(tfms):
    import pandas as pd
    rows = []
    for tfm in tfms:
        row = {}
        row['class'] = tfm.__class__.__name__
        row['class2'] = tfm.tfm.__class__.__name__
        row['func'] = tfm.tfm.func.__name__
        if tfm.p != 1.0:
            row['p'] = tfm.p
        else:
            row['p'] = ''
        row['kwargs'] = tfm.kwargs
        rows.append(row)
    df = pd.DataFrame(rows)
    # df = utils.df_order_columns(['class'])
    return df


def _sample_points(x: PointsItem, num_points=50000, replace=True):

    d = x.data
    n = num_points

    num_points_was = len(d['points'])

    if n > 0:
        n = np.min([n, num_points_was])
        if replace:
            indices = np.random.randint(num_points_was, size=n)
        else:
            indices = np.random.choice(num_points_was, size=n, replace=False)
        for k in ['points', 'normals', 'colors', 'labels']:
            if k in d:
                if k == 'labels':
                    is_multilabels = isinstance(d[k], (list, tuple))
                    if is_multilabels:
                        sampled = []
                        for labels in d[k]:
                            assert len(labels) == num_points_was
                            sampled.append(labels[indices])
                        d[k] = sampled
                    else:
                        assert len(d[k]) == num_points_was
                        d[k] = d[k][indices]
                else:
                    assert len(d[k]) == num_points_was
                    d[k] = d[k][indices]
    return x


sample_points = Transform(_sample_points)
