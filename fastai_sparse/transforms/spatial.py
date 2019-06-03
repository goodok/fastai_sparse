# -*- coding: utf-8 -*-

import trimesh
import numpy as np
import scipy

from .main import Transform
from ..data_items import ItemBase, PointsItem, MeshItem

__all__ = ['normalize_spatial', 'fit_to_sphere', 'crop_points', 'elastic']


class TfmSpatial(Transform):
    "Transfomation of spatial coordinates of points"
    pass


def _normalize_spatial(x, mean=True, std=False):
    """
    Mean points.
    """
    if isinstance(x, MeshItem):
        points = x.vertices
    else:
        points = x.data['points']

    if mean:
        points = points - points.mean(0)
    if std:
        points = points / points.std(0)

    if isinstance(x, MeshItem):
        x.vertices = points
    else:
        x.data['points'] = points

    # variant 2
    # x.refresh()  # if already is not refreshed
    # offset = - points.mean(0)
    # m = _translate(offset)
    # x.affine_mat = m @ x.affine_mat
    return x


normalize_spatial = Transform(_normalize_spatial)


def _fit_to_sphere(x: ItemBase, center=True):
    if isinstance(x, PointsItem):
        d = x.data
        points = d['points']

        if center:
            points = points - points.mean(0)

        max_distance = (points ** 2).sum(1).max() ** 0.5

        points /= max_distance
        d['points'] = points

        return x

    elif isinstance(x, MeshItem):
        mesh = x.data
        points = mesh.vertices

        if center:
            m = trimesh.transformations.translation_matrix(- points.mean(0))
            mesh.apply_transform(m)

        points = mesh.vertices
        max_distance = (points ** 2).sum(1).max() ** 0.5

        m = trimesh.transformations.scale_matrix(1 / max_distance)
        mesh.apply_transform(m)
        return x

    else:
        raise NotImplementedError


fit_to_sphere = Transform(_fit_to_sphere)


def _crop_points(x, low=0, high=1, debug_simulation=0):
    """
    Filter points by their coords. Remains in input field size.

    Validation function must oprated  with `filtred_mask`, `labels_raw`

    `labels` will put to th net (prediction and loss function)

    """
    # full_scale - Input field size
    # TODO: what to do with filtred points while prediction ? (Tensor.max([0, 0, 0, ...0])[1] ==> 19

    d = x.data
    points = d['points']

    # filter
    indexer = (points.min(1) >= low) & (points.max(1) < high)

    if debug_simulation:
        # debug, emulate filtred points
        for debug_i in range(5):
            indexer[1 + debug_i] = False

    # store original labels
    d['labels_raw'] = d['labels']

    # filter all linked params
    d['points'] = points[indexer]
    d['colors'] = d['colors'][indexer]

    is_multilabels = isinstance(d['labels'], (list, tuple))
    if is_multilabels:
        labels = []
        for l in d['labels']:
            labels.append(l[indexer])
        d['labels'] = labels
    else:
        d['labels'] = d['labels'][indexer]

    # save indexer
    assert 'filtred_mask' not in d
    d['filtred_mask'] = indexer

    return x


crop_points = Transform(_crop_points)

blur0 = np.ones((3, 1, 1)).astype('float32') / 3
blur1 = np.ones((1, 3, 1)).astype('float32') / 3
blur2 = np.ones((1, 1, 3)).astype('float32') / 3


def _elastic(o, gran, mag):
    # from https://github.com/facebookresearch/SparseConvNet/blob/master/examples/ScanNet/data.py
    if isinstance(o, MeshItem):
        x = o.vertices.astype(np.float32)
    else:
        x = o.data['points']

    # original begin
    bb = np.abs(x).max(0).astype(np.int32) // gran + 3
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
    ax = [np.linspace(-(b - 1) * gran, (b - 1) * gran, b) for b in bb]
    interp = [scipy.interpolate.RegularGridInterpolator(
        ax, n, bounds_error=0, fill_value=0) for n in noise]

    def g(x_):
        return np.hstack([i(x_)[:, None] for i in interp])

    # return x+g(x)*mag
    x = x + g(x) * mag

    # original end

    if isinstance(o, MeshItem):
        o.vertices = x
    else:
        o.data['points'] = x

    return o


elastic = Transform(_elastic)
