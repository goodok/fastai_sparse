# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division

import sys
import subprocess
import humanize
import numpy as np
import pandas as pd
import re
import contextlib

import json
import os
from IPython.display import display, HTML
import warnings
from collections import OrderedDict
import platform


def save_array(fname, arr):
    np.save(fname, arr)


def load_array(fname):
    return np.load(fname)


def save_json(fname, d, pretty=False):
    fname = str(fname)
    with open(fname, 'w') as f:
        if pretty:
            json.dump(d, f, indent=4, sort_keys=True)
        else:
            json.dump(d, f)


def load_json(fname):
    fname = str(fname)
    with open(fname) as f:
        return json.load(f)


@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)


def wide_notebook(percents=70):
    display(HTML("<style>.container { width:70% !important; }</style>"))
    log_options['max_name_length'] = 25
    log_options['max_shape_length'] = 20


# TODO: refactor: arguments - just list
def watermark(python=True, virtualenv=True, keras=False, tensorflow=False, nvidia=True, cudnn=True,
              hostname=False, torch=True, fastai=True, fastai_sparse=True, **kwargs):
    lines = OrderedDict()
    if virtualenv:
        r = None
        if 'PS1' in os.environ:
            r = os.environ['PS1']
        elif 'VIRTUAL_ENV' in os.environ:
            r = os.environ['VIRTUAL_ENV']
        lines['virtualenv'] = r
    if python:
        r = sys.version.splitlines()[0]
        m = re.compile(r'([\d\.]+)').match(r)
        if m:
            r = m.groups()[0]
        lines['python'] = r
    if hostname:
        lines["hostname"] = platform.node()

    def find_in_lines(pip_list, package_name, remove_name=True):
        res = ''
        for line in pip_list:
            if hasattr(line, 'decode'):
                line = line.decode('utf-8')
            if package_name in line and line.startswith(package_name):
                if remove_name:
                    res = line.split(package_name)[1].strip()
                else:
                    res = line.strip()
                break
        return res

    if nvidia:
        lines['nvidia driver'] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]).splitlines()[0]
        r = subprocess.check_output(["nvcc", "--version"]).splitlines()
        r = find_in_lines(r, 'release', False)
        r = r.split('release')[1].strip()
        lines['nvidia cuda'] = r

    if cudnn and sys.platform.startswith('linux'):
        with open('/usr/local/cuda/include/cudnn.h', 'r') as f:
            r = f.readlines()
        v1 = find_in_lines(r, '#define CUDNN_MAJOR')
        v2 = find_in_lines(r, '#define CUDNN_MINOR')
        v3 = find_in_lines(r, '#define CUDNN_PATCHLEVEL')
        lines['cudnn'] = "{}.{}.{}".format(v1, v2, v3)

    pip_list = subprocess.check_output(["pip", "list"]).splitlines()

    if keras:
        lines['keras'] = find_in_lines(pip_list, 'Keras')
    if tensorflow:
        lines['tensorflow-gpu'] = find_in_lines(pip_list, 'tensorflow-gpu')
    if torch:
        lines['torch'] = find_in_lines(pip_list, 'torch')

    for key, val in kwargs.items():
        if val:
            lines[key] = find_in_lines(pip_list, key)

    if fastai:
        try:
            from fastai import version
            v = version.__version__
        except:
            v = None
        if v:
            lines['fastai'] = v

    if fastai_sparse:
        try:
            from fastai_sparse import version
            v = version.__version__
        except:
            v = None

        if v:
            lines['fastai_sparse'] = v

    res = ["{: <15} {}".format(k + ":", v) for (k, v) in lines.items()]

    print("\n".join(res))


def df_order_columns(df, columns_ordered=[]):
    """
    Order some columns, and remain other as was
    """
    columns = []
    for c in columns_ordered:
        if c in df.columns:
            columns.append(c)
    remains = [c for c in df.columns if c not in columns]
    columns = columns + remains

    assert len(columns) == len(df.columns)
    return df[columns]


def df_split_random(df, N, random_seed=None):
    """
    Split DataFrame on two parts.

    N : int
        Size of first part
    """
    random = np.random.RandomState(random_seed)

    all_local_indices = np.arange(len(df))
    shuffled = random.permutation(all_local_indices)

    df1 = df.iloc[shuffled[:N]]
    df2 = df.iloc[shuffled[N:]]
    return df1, df2


log_options = {
    'max_name_length': 14,
    'max_shape_length': 14,
}

scalar_types = (int, np.int, np.int32, np.int64, np.uint, np.uint32, np.uint64)


def log(text, array=None, indent=''):
    """Prints a text message. And, optionally, if a Numpy array is provided iterators
    prints iterators's shape, min, and max values.
    """
    if isinstance(text, dict) or isinstance(array, dict):
        log_dict(text, array)
        return
    if isinstance(array, list):
        d = dict([(str(i), v) for i, v in enumerate(array)])
        log_dict(text, d)
        return
    if array is not None:
        text = indent + text.ljust(log_options['max_name_length'])
        # if scalar
        if isinstance(array, scalar_types):
            text += f"{array}"
        else:
            try:
                s_mean = '{:10.5f}'.format(array.mean()) if array.size else ""
            except:
                try:
                    s_mean = '{:10.5f}'.format(
                        np.array(array).mean()) if array.size else ""
                except:
                    s_mean = ""
            s_min = calc_and_format_minmax(array, 'min')
            s_max = calc_and_format_minmax(array, 'max')

            dtype = str(array.dtype)
            s_shape = '{:' + str(log_options['max_shape_length']) + '}'
            s_shape = s_shape.format(str(tuple(array.shape)))
            text += ("shape: {}  dtype: {:13}  min: {},  max: {},  mean: {}".format(
                s_shape,
                dtype,
                s_min,
                s_max,
                s_mean))
    print(text)


def log_as_dict(text, array=None, indent=''):
    r = {}
    r['title'] = indent + text
    if array is not None:
        if isinstance(array, scalar_types):
            r['value'] = array
            r['dtype'] = str(type(array))
        else:
            r['dtype'] = str(array.dtype)
            try:
                s_mean = '{:10.5f}'.format(array.mean()) if array.size else ""
            except:
                try:
                    s_mean = '{:10.5f}'.format(
                        np.array(array).mean()) if array.size else ""
                except:
                    s_mean = ""
            r['mean'] = s_mean
            r['min'] = calc_and_format_minmax(array, 'min')
            r['max'] = calc_and_format_minmax(array, 'max')
            # s_shape = '{:' + str(log_options['max_shape_length']) + '}'
            r['shape'] = str(tuple(array.shape))
    return r


def log_dict(text, d=None):
    if d is None:
        d = text
    else:
        print(f'{text}:')
    indent = ' ' * 3
    for key in d:
        try:
            value = d[key]
            s_key = '{:<' + str(log_options['max_name_length']) + '}'
            s_key = s_key.format(key)

            if isinstance(value, str):
                print(f"{indent}{s_key}'{value}'")
            elif isinstance(value, list):
                print(f"{indent}{s_key}{value}")
            else:
                log(key, value, indent=indent)
        except Exception as e:
            # warnings.warn(f"Can't log key='{key}': {e}")
            warn_always(f"Can't log key='{key}': {e}")


def calc_and_format_minmax(array, fn='min'):
    s = ''
    if array.size:
        if fn == 'min':
            v = array.min()
        else:
            v = array.max()

    if is_array_integer(array):
        s = '{:10}'.format(v)
    else:
        s = '{:10.5f}'.format(v)
    return s


def is_array_integer(a):
    s_integers = ['torch.int16', 'torch.int32', 'torch.int64',
                  'int16', 'int32', 'int64',
                  ]
    return str(a.dtype) in s_integers


def print_trainable_parameters(model, max_rows=200, max_colwidth=100):
    params = []
    total_number = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            shape = " x ".join([str(i) for i in list(p.shape)])
            number = p.numel()
            params.append({'name': name, 'shape': shape, 'number': p.numel()})
            total_number += number
    df_params = pd.DataFrame(params)
    print('Total:', humanize.intcomma(total_number))
    with pd.option_context("display.max_rows", max_rows, 'max_colwidth', max_colwidth):
        display(df_params)


def get_random_states(cuda=True):
    import torch
    import random
    d = {}
    d['torch'] = torch.initial_seed()

    v = np.random.get_state()
    d['numpy'] = f"{v[0]}, {v[1][0]:10} {v[1][1]:10} {v[2]}"

    v = random.getstate()
    d['python'] = f"{v[0]}, {v[1][0]}"

    if cuda:
        v = torch.cuda.get_rng_state()
        d['cuda'] = f"{v[0:10]}"
    return d


def print_random_states(cuda=True):
    d = get_random_states(cuda=cuda)
    print(d)


def warn_always(s):
    with warnings.catch_warnings(record=True) as caught_warnings:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        warnings.warn(s, UserWarning)

    for warning in caught_warnings:
        if warning.category == UserWarning:
            warnings.warn(warning.message)
