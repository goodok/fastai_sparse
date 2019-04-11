# -*- coding: utf-8 -*-

import os
import numpy as np
from numpy import array
from typing import Any, Optional, Collection, Union, Callable, Dict, List
from types import SimpleNamespace
import collections
from collections import Iterable
from pathlib import Path
import inspect
from numbers import Number

import torch
from torch import Tensor
import torch.nn.functional as F

"""Some utils and datatypes from fastai library"""

__all__ = ['array', 'Any', 'Optional', 'Collection', 'Union', 'Callable', 'Dict', 'List', 'Iterable', 'Path', 'Number',
           'F', ]

# datatypes

ListOrItem = Union[Collection[Any], int, float, str]
OptListOrItem = Optional[ListOrItem]
PathOrStr = Union[Path, str]

# torch

TensorOrNumber = Union[Tensor, Number]
MetricsList = Collection[TensorOrNumber]
Tensors = Union[Tensor, Collection['Tensors']]


# utils

def is_listy(x: Any) -> bool:
    return isinstance(x, (tuple, list))


def listify(p: OptListOrItem = None, q: OptListOrItem = None):
    "Make `p` listy and the same length as `q`."
    if p is None:
        p = []
    elif isinstance(p, str):
        p = [p]
    elif not isinstance(p, Iterable):
        p = [p]
    # Rank 0 tensors in PyTorch are Iterable but don't have a length.
    else:
        try:
            _ = len(p)
        except:
            p = [p]
    n = q if type(q) == int else len(p) if q is None else len(q)
    if len(p) == 1:
        p = p * n
    assert len(p) == n, f'List len mismatch ({len(p)} vs {n})'
    return list(p)


def _get_default_args(func: Callable):
    return {k: v.default
            for k, v in inspect.signature(func).parameters.items()
            if v.default is not inspect.Parameter.empty}


def num_cpus() -> int:
    "Get number of cpus"
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count()


_default_cpus = min(16, num_cpus())
defaults = SimpleNamespace(cpus=_default_cpus, cmap='viridis', return_fig=False)


def try_int(o: Any) -> Any:
    "Try to convert `o` to int, default to `o` if not possible."
    # NB: single-item rank-1 array/tensor can be converted to int, but we don't want to do this
    if isinstance(o, (np.ndarray, )):
        return o if o.ndim else int(o)
    if isinstance(o, collections.Sized) or getattr(o, '__array_interface__', False):
        return o
    try:
        return int(o)
    except:
        return o


def show_some(items: Collection, n_max: int = 5, sep: str = ','):
    "Return the representation of the first  `n_max` elements in `items`."
    if items is None or len(items) == 0:
        return ''
    res = sep.join([f'{o}' for o in items[:n_max]])
    if len(items) > n_max:
        res += '...'
    return res


class PreProcessor():
    "Basic class for a processor that will be applied to items at the end of the data block API."

    def __init__(self, ds: Collection = None):
        self.ref_ds = ds

    def process_one(self, item: Any):
        return item

    def process(self, ds: Collection):
        ds.items = array([self.process_one(item) for item in ds.items])


PreProcessors = Union[PreProcessor, Collection[PreProcessor]]


def ifnone(a: Any, b: Any) -> Any:
    "`a` if `a` is not None, otherwise `b`."
    return b if a is None else a


def is_dict(x: Any) -> bool:
    return isinstance(x, dict)


# torch utils

defaults.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def to_device(b: Tensors, device: torch.device):
    "Recursively put `b` on `device`."
    device = ifnone(device, defaults.device)
    if is_listy(b):
        return [to_device(o, device) for o in b]
    if is_dict(b):
        return {k: to_device(v, device) for k, v in b.items()}
    return b.to(device, non_blocking=True)
