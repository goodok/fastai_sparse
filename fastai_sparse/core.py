# -*- coding: utf-8 -*-

from typing import Any, Optional, Collection, Union, Callable
from collections import Iterable
import inspect

"""Some utils and datatypes from fastai library"""

# datatypes

ListOrItem = Union[Collection[Any], int, float, str]
OptListOrItem = Optional[ListOrItem]


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
            a = len(p)
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
