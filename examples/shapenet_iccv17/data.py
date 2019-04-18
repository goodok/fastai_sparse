# -*- coding: utf-8 -*-

from functools import partial
from fastai_sparse.data import SparseDataBunch


merge_fn = partial(SparseDataBunch.merge_fn, keys_lists=['id', 'random_seed', 'num_points', 'categ', 'class_offset', 'num_classes'])
