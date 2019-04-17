# -*- coding: utf-8 -*-

import numpy as np
import math
import scipy

from fastai_sparse.data_items import MeshItem, SparseItem, PointsItem
from fastai_sparse.transforms import Transform, transfer_keys

from fastai_sparse.transforms import (to_points_cloud, sample_points,
                                     normalize_spatial, colors_normalize, colors_noise,
                                     rotate, rotate_XY, translate, rand_translate, scale, flip_x,
                                     noise_affine, 
                                     elastic, crop_points,
                                     merge_features, to_sparse_voxels,
                                     log_transforms)


from fastai_sparse import utils

import fastai_sparse.transforms.main as transform_base
transform_base.TRANSFER_KEYS = [
    'id', 'random_seed', 'num_classes', 'filtred_mask', 'labels_raw']


def _remap_labels(x, remapper):
    x.labels = remapper[x.labels]
    return x

remap_labels = Transform(_remap_labels)


def _specific_translate(x, full_scale=4096):
    # from https://github.com/facebookresearch/SparseConvNet/blob/master/examples/ScanNet/data.py

    if isinstance(x, MeshItem):
        points = x.vertices.astype(np.float32)
    else:
        points = x.data['points']

    m = points.min(0)   # min corner of cube
    M = points.max(0)   # max corner of cube
    # q = M - m

    offset = -m + \
        np.clip(full_scale - M + m - 0.001, 0, None) * np.random.rand(3) + \
        np.clip(full_scale - M + m + 0.001, None, 0) * np.random.rand(3)

    points += offset

    if isinstance(x, MeshItem):
        x.vertices = points
    else:
        x.data['points'] = points
    return x

specific_translate = Transform(_specific_translate)


#def _sparse(x: SparseItem):
#    d = x.data.copy()

#    points = d['points']

#    coords = points.astype(np.int64)

#    res = {'coords': coords,
#           'features': d['features'],
#           'labels': d['labels'].astype(np.int64),
#           }

#    transfer_keys(d, res)

#    return SparseItem(res)


#sparse = Transform(_sparse)

