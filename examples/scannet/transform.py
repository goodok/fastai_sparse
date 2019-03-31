# -*- coding: utf-8 -*-

import numpy as np
import math
import scipy

from fastai_sparse.data_items import MeshItem, SparseItem, PointsItem
from fastai_sparse.transform import Transform, transfer_keys

from fastai_sparse.transform import (extract_points, sample_points, mean,  # normalize,
                                     rotate, rotate_XY, translate, rand_translate, scale, flip_x,
                                     noise_transformation, 
                                     filter_points,
                                     merge_features, sparse,
                                     log_transforms)


from fastai_sparse import utils

import fastai_sparse.transform as transform_base
transform_base.TRANSFER_KEYS = [
    'id', 'random_seed', 'num_classes', 'filtred_mask', 'labels_raw']


# TODO: inplace wrapper, global option


def _remap_labels(x, remapper, inplace=False):
    if inplace:
        d = x.data
    else:
        d = x.data.copy()

    d['labels'] = remapper[d['labels']]

    #transfer_keys(d, d2)
    if inplace:
        return x
    else:
        return PointsItem(d)

remap_labels = Transform(_remap_labels)

def _mean_colors(x, inplace=False):
    """
    Mean colors.
    """
    if inplace:
        d = x.data
    else:
        d = x.data.copy()

    d = x.data
    colors = d['colors']
    colors = colors[:, :3] / 127.5 - 1
    colors = colors.astype(np.float32)

    d['colors'] = colors

    # transfer_keys(d, d2)
    if inplace:
        return x
    else:
        return PointsItem(d)

mean_colors = Transform(_mean_colors)


def _noise_color(x, amplitude=0.1):
    d = x.data
    # noise
    d['colors'] = d['colors'] + \
        np.random.randn(3).astype(np.float32) * amplitude
    return x

noise_color = Transform(_noise_color)


# TODO: remove  old
def _random_spatial_transform(x, amplitude=0.1, flip_x=False, scale=20, rotate_in_XY=True, inplace=False):
    # scale=20  #Voxel size = 1/scale
    if inplace:
        d = x.data
    else:
        d = x.data.copy()

    # noise
    m = np.eye(3).astype(np.float32) + \
        np.random.randn(3, 3).astype(np.float32) * amplitude

    #print(m, m.dtype)
    # print(amplitude)

    # flip
    if flip_x:
        m[0][0] *= np.random.randint(0, 2) * 2 - 1

    # scale
    m *= scale


    # rotate_in_XY
    if rotate_in_XY:
        theta = np.random.rand() * 2 * math.pi
        rot = np.array([[math.cos(theta), math.sin(theta), 0],
                        [-math.sin(theta), math.cos(theta), 0],
                        [0, 0, 1]],
                       dtype=np.float32)
        m = np.matmul(rot, m )

    d['points'] = np.matmul(d['points'], m.T)

    #transfer_keys(d, d2)

    if inplace:
        return x
    else:
        return PointsItem(d)


random_spatial_transform = Transform(_random_spatial_transform)

# TODO: remove old
def _random_shift(o, full_scale=4096, low=-2, high=2, inplace=False):
    if inplace:
        d = o.data
    else:
        d = o.data.copy()
    points = d['points']

    points += full_scale / 2 + np.random.uniform(low=low, high=high, size=3).astype(np.float32)

    d['points'] = points
    if inplace:
        return o
    else:
        return PointsItem(d)


random_shift = Transform(_random_shift)



blur0 = np.ones((3, 1, 1)).astype('float32')/3
blur1 = np.ones((1, 3, 1)).astype('float32')/3
blur2 = np.ones((1, 1, 3)).astype('float32')/3


def _elastic(o, gran, mag, inplace=False):
    # from https://github.com/facebookresearch/SparseConvNet/blob/master/examples/ScanNet/data.py
    if inplace:
        d = o.data
    else:
        d = o.data.copy()
    x = d['points']

    # original begin
    bb = np.abs(x).max(0).astype(np.int32)//gran+3
    noise = [np.random.randn(bb[0], bb[1], bb[2]).astype(
        'float32') for _ in range(3)]
    noise = [scipy.ndimage.filters.convolve(
        n, blur0, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(
        n, blur1, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(
        n, blur2, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(
        n, blur0, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(
        n, blur1, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(
        n, blur2, mode='constant', cval=0) for n in noise]
    ax = [np.linspace(-(b-1)*gran, (b-1)*gran, b) for b in bb]
    interp = [scipy.interpolate.RegularGridInterpolator(
        ax, n, bounds_error=0, fill_value=0) for n in noise]

    def g(x_):
        return np.hstack([i(x_)[:, None] for i in interp])

    # return x+g(x)*mag
    x = x+g(x)*mag

    # original end

    d['points'] = x

    if inplace:
        return o
    else:
        return PointsItem(d)


elastic = Transform(_elastic)


# TODO: remove old
def _clip_and_filter(o, full_scale=4096, inplace=False):
    # full_scale - Input field size
    # TODO: what to do with filtred points while prediction ? (Tensor.max([0, 0, 0, ...0])[1] ==> 19

    # from https://github.com/facebookresearch/SparseConvNet/blob/master/examples/ScanNet/data.py
    if inplace:
        d = o.data
    else:
        d = o.data.copy()
    points = d['points']

    # original begin

    m = points.min(0)
    M = points.max(0)
    # q = M - m

    offset = -m + np.clip(full_scale - M + m - 0.001, 0, None) * np.random.rand(3) + np.clip(full_scale - M + m + 0.001, None, 0) * np.random.rand(3)
    points += offset

    # filter
    idxs = (points.min(1) >= 0) & (points.max(1) < full_scale)

#    # debug, emulate filtred points
#    for debug_i in range(5):
#        idxs[1 + debug_i] = False

    d['labels_raw'] = d['labels']

    d['points'] = points[idxs]
    d['colors'] = d['colors'][idxs]
    d['labels'] = d['labels'][idxs]

    # original end

    # save indexes
    d['filtred_mask'] = idxs

    if inplace:
        return o
    else:
        return PointsItem(d)


clip_and_filter = Transform(_clip_and_filter)

def _special_translate(x, full_scale=4096):
    # from https://github.com/facebookresearch/SparseConvNet/blob/master/examples/ScanNet/data.py
    d = x.data
    points = d['points']

    m = points.min(0)   # min corner of cube
    M = points.max(0)   # max corner of cube
    # q = M - m

    offset = -m + \
        np.clip(full_scale - M + m - 0.001, 0, None) * np.random.rand(3) + \
        np.clip(full_scale - M + m + 0.001, None, 0) * np.random.rand(3)

    points += offset
    return x

special_translate = Transform(_special_translate)


def _sparse(x: SparseItem):
    d = x.data.copy()

    points = d['points']

    coords = points.astype(np.int64)

    res = {'coords': coords,
           'features': d['features'],
           'labels': d['labels'].astype(np.int64),
           }

    transfer_keys(d, res)

    return SparseItem(res)


sparse = Transform(_sparse)

