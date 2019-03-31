# -*- coding: utf-8 -*-

import numpy as np
import warnings
from functools import partial

from fastai.basics import copy, dataclass, field, Optional, Collection, List
from fastai.core import Callable, listify, is_listy
from fastai.vision.image import _get_default_args

from . import utils

from .data_items import MeshItem, SparseItem, PointsItem


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


def _extract_points(x: MeshItem, method='centres'):
    assert method in ['centres', 'vertices']
    if method == 'centres':
        d = _extract_points_by_centers(x)
    elif method == 'vertices':
        d = _extract_points_by_vertices(x)

    example_id = x.data.metadata['example_id']
    d['id'] = example_id  # TODO: x  has id ??
    state = np.random.get_state()
    d['random_seed'] = f'{state[1][0]}_{state[2]}'
    return PointsItem(d)


def _extract_points_by_centers(self):
    """
    Extract points as center of faces.
    """
    # TODO: option calculate 'normals' or not
    # TODO: option alpha channel
    assert self.is_colors_from_vertices
    assert not self.is_labels_from_vertices

    mesh = self.data

    points = np.array(mesh.vertices, dtype=np.float32)
    colors = np.array(self.colors, dtype=np.float32)[:, :3]  # without alpha channel
    assert len(points) == len(colors)

    faces = np.array(mesh.faces)
    labels = self.labels
    assert len(faces) == len(labels)

    faces_xyz = points[faces]
    faces_rgb = colors[faces]

    # calculate centres
    points = np.mean(faces_xyz, axis=1)
    colors = np.mean(faces_rgb, axis=1)

    normals = np.array(mesh.face_normals, dtype=np.float32)

    # TODO: check dtypes
    return {'points': points, 'normals': normals, 'colors': colors, 'labels': labels}


def _extract_points_by_vertices(self):
    # TODO: normals (from vertices)
    mesh = self.data

    points = np.array(mesh.vertices, dtype=np.float32)
    colors = self.colors
    labels = self.labels

    assert len(points) == len(colors)
    assert len(points) == len(labels)

    return {'points': points, 'colors': colors, 'labels': labels}


class ExtractPoints(Transform):
    order = 0
    pass


extract_points = ExtractPoints(_extract_points)


def _sample_points(x: PointsItem, num_points=50000):

    d = x.data

    points = d['points']
    normals = d['normals']
    colors = d['colors']
    labels = d['labels']

    n = num_points

    # print(num_points)

    # TODO: random usage
    if n > 0:
        n = np.min([n, len(points)])
        indices = np.random.randint(len(points), size=n)

        # TODO: join, gather, then split (for speed)
        points = points[indices]
        normals = normals[indices]
        colors = colors[indices]
        labels = labels[indices]

    d2 = {'points': points, 'normals': normals,
          'colors': colors, 'labels': labels}

    transfer_keys(d, d2)

    return PointsItem(d2)


class SamplePoints(Transform):
    order = 1
    pass


sample_points = SamplePoints(_sample_points)


def _mean(x):
    """
    Mean points.
    """
    points = x.data['points']
    points = points - points.mean(0)

    # offset = - points.mean(0)
    # m = _translate(offset)
    # x.affine_mat = x.affine_mat @ m
    # x.refresh()  # TODO: calculate do_refresh properly to avoid unnessesary refreash
    return x


mean = Transform(_mean)


def _filter_points(x, low=0, high=1, debug_simulation=0):
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
    d['labels'] = d['labels'][indexer]

    # save indexer
    assert 'filtred_mask' not in d
    d['filtred_mask'] = indexer

    return x


filter_points = Transform(_filter_points)


# Affine Tranforms

class TfmAffine(Transform):
    "Decorator for affine tfm funcs."
    _wrap = 'affine'


def _rotate():
    # rotation matrix
    Q, R = np.linalg.qr(np.random.randn(3, 3), mode='reduced')
    Q = Q.astype(np.float32)

    # transformation matrix
    m = np.eye(4, dtype='float32')
    m[:3, :3] = Q

    return m


rotate = TfmAffine(_rotate)


def _rotate_XY():
    # rotation matrix
    theta = np.random.rand() * 2 * np.pi

    Q = np.array([[np.cos(theta), np.sin(theta)],
                  [-np.sin(theta), np.cos(theta)]],
                 dtype=np.float32)

    # transformation matrix
    m = np.eye(4, dtype='float32')
    m[:2, :2] = Q
    return m


rotate_XY = TfmAffine(_rotate_XY)


def _flip_x():
    return np.diag([-1, 1, 1, 1]).astype(np.float32)


flip_x = TfmAffine(_flip_x)


def _scale(scale=1):
    # See fastai.vision.transform._zoom
    # Only simple scaling around O point now.
    s = scale
    return np.diag([s, s, s, 1]).astype(np.float32)


scale = TfmAffine(_scale)


def _rand_scale(scale: np.random.uniform = 1.0):
    # TODO: this is differ from fastai.vision.transform.rand_zoom, but more correct?
    "Randomized version of `moove`."
    return _scale(scale=scale)


rand_scale = TfmAffine(_rand_scale)


def _translate(offset=0):
    v = offset
    if is_listy(v) or isinstance(v, np.ndarray):
        assert len(v) == 3
    else:
        v = [v, v, v]
    v = np.array(v, dtype=np.float32)
    m = np.eye(4).astype(np.float32)
    m[:3, 3] = v
    return m


translate = TfmAffine(_translate)


def _rand_translate(offset: np.random.uniform = 1.0):
    # TODO: this is differ from fastai.vision.transform.rand_zoom: return result of fucntion, not transform, is it more correct?
    "Randomized version of `translate`."
    # TODO: partial(np.random.uniform, size=3)
    assert len(offset) == 3
    return _translate(offset=offset)


rand_translate = TfmAffine(_rand_translate)


def _noise_transformation(amplitude=0.1):
    """
    Tansform with random transformation matrix (I + amplitude * random 3x3).
    """
    m = np.eye(4).astype(np.float32)
    m[:3, :3] += np.random.randn(3, 3).astype(np.float32) * amplitude
    return m


noise_transformation = TfmAffine(_noise_transformation)


def _merge_features(x: PointsItem, ones=True, normals=False, colors=False):
    # TODO: inplace

    append_ones = ones
    append_normals = normals
    append_colors = colors

    d = x.data.copy()

    points = d['points']
    normals = d.get('normals', None)
    colors = d.get('colors', None)
    n_points = points.shape[0]

    # create features
    features = []
    if append_ones:
        features.append(np.ones((n_points, 1)).astype(np.float32))

    if append_normals:
        if normals is not None:
            features.append(normals)
        else:
            utils.warn_always('merge_features: append_normals is True, but there is no normals')

    if append_colors:
        if colors is not None:
            features.append(colors)
        else:
            utils.warn_always('merge_features: append_colors is True, but there is no colors')

    features = np.hstack(features)

    res = {'points': points, 'features': features, 'labels': d['labels']}

    # TODO: global/parameter ?
    transfer_keys(d, res)

    return PointsItem(res)


class MergeFeatures(Transform):
    order = 1
    pass


merge_features = MergeFeatures(_merge_features)


def _sparse(x: SparseItem):
    d = x.data.copy()

    points = d['points']

    # TODO: is floor better then simply astype(np.int64) ? For x > 0 there is no differences
    # Some spreadsheet programs calculate the “floor-towards-zero”, in other words floor(-2.5) == -2. NumPy instead uses the definition of floor where floor(-2.5) == -3.
    # >>> a = np.array([-1.7, -1.5, -0.2, 0.2, 0.5, 0.7, 1.3, 1.5, 1.7, 2.0, 2.5, 2.9])
    # >>> b = np.floor(a)
    # >>> c = a.astype(np.int64)
    # >>> pd.DataFrame([a, b, c])
    #    	-1.7 	-1.5 	-0.2 	0.2 	0.5 	0.7 	1.3 	1.5 	1.7 	2.0 	2.5 	2.9
    #    	-2.0 	-2.0 	-1.0 	0.0 	0.0 	0.0 	1.0 	1.0 	1.0 	2.0 	2.0 	2.0
    #    	-1.0 	-1.0 	0.0 	0.0 	0.0 	0.0 	1.0 	1.0 	1.0 	2.0 	2.0 	2.0

    coords = np.floor(points).astype(np.int64)

    res = {'coords': coords,
           'features': d['features'],
           'labels': d['labels'].astype(np.int64),
           }

    transfer_keys(d, res)

    return SparseItem(res)


class Sparse(Transform):
    order = 1
    pass


sparse = Sparse(_sparse)
