# -*- coding: utf-8 -*-

from functools import partial
from fastai_sparse.data import SparseDataBunch

merge_fn = partial(SparseDataBunch.merge_fn, keys_lists=['id', 'labels_raw', 'filtred_mask', 'random_seed', 'num_points'])
