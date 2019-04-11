# -*- coding: utf-8 -*-

import numpy as np

from .main import Transform

__all__ = ['TfmColors', 'colors_noise', 'colors_normalize']


class TfmColors(Transform):
    "Transfomation of colors"
    pass


def _colors_noise(x, amplitude=0.1):
    d = x.data
    # noise
    d['colors'] = d['colors'] + \
        np.random.randn(3).astype(np.float32) * amplitude
    return x


colors_noise = TfmColors(_colors_noise)


def _colors_normalize(x, center=127.5, scale=1 / 127.5):
    """
    Normalize colors.

    [0..255] ---> [-1.0, 1.0]
    """
    d = x.data
    colors = d['colors']
    colors = (colors[:, :3] - center) * scale
    colors = colors.astype(np.float32)

    d['colors'] = colors

    return x


colors_normalize = TfmColors(_colors_normalize)
