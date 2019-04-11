# -*- coding: utf-8 -*-

# import numpy as np

from fastai_sparse.data_items import ItemBase
from fastai_sparse.transforms import Transform, transfer_keys

from fastai_sparse.transforms import (rotate, flip_x, translate, scale, fit_to_sphere, merge_features, to_sparse_voxels)


import fastai_sparse.transforms.main as transform_base

transform_base.TRANSFER_KEYS = ['id', 'random_seed', 'categ', 'class_offset', 'num_classes']


def _shift_labels(x: ItemBase, offset=0):
    x.data['labels'] += offset
    return x


shift_labels = Transform(_shift_labels)
