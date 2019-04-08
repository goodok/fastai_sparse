# -*- coding: utf-8 -*-

# Affine Tranforms

import numpy as np

from .main import Transform
from ..core import is_listy

__all__ = ['TfmAffine',
           'rotate', 'rotate_XY', 'flip_x',
           'scale', 'rand_scale',
           'translate', 'rand_translate',
           'noise_affine',
           ]


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


def _noise_affine(amplitude=0.1):
    """
    Tansform with random transformation matrix (I + amplitude * random 3x3).
    """
    m = np.eye(4).astype(np.float32)
    m[:3, :3] += np.random.randn(3, 3).astype(np.float32) * amplitude
    return m


noise_affine = TfmAffine(_noise_affine)
