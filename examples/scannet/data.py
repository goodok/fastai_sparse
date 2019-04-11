# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from os.path import join, exists, basename
import numbers

import torch
#import torchnet
from torch.utils.data import Dataset, DataLoader
import glob
import math
import os
import scipy
import scipy.ndimage
import trimesh
from pathlib import Path
from matplotlib import pyplot as plt
from tqdm import tqdm
import warnings

#import sparseconvnet as scn

import fastai

from fastai.basic_data import DataBunch, DeviceDataLoader
from fastai.data_block import ItemList, ItemLists, PreProcessor, LabelLists
from fastai.core import array, PathOrStr, Callable, ifnone, defaults, Optional, TfmList, Collection, ItemBase, listify, is_listy
from fastai.imports import F
from fastai.torch_core import data_collate, try_int, to_data, ItemsList, Tensor, to_device
#from fastai.vision.data import _prep_tfm_kwargs

from typing import Any, Iterator, Dict, List
from torch.utils.data import Dataset


from fastai_sparse import utils
from fastai_sparse.data_items import extract_data

def merge_fn(batch: ItemsList)->Tensor:
    """
    Convert `batch` items to tensor data (dictionary for SparseConvNet).

    Changes from SparseConvNet examples [1] in function `merge`
        - batch = [xb, yb]
        - xb['x'] = [..., ...]  replaced by xb['coords'] and xb['features']
        - remove `mask`
        - `category` is just list by examples, not arrays by every points

    [1] https://github.com/facebookresearch/SparseConvNet/blob/master/examples/3d_segmentation/data.py

    """
    # TODO:
    #  --- replace 'nPoints' --> n_points
    #  --- nClasses --> num_classes
    #  --- xf --> id


    # see
    #    fastai.torch_core.data_collate
    #    torch.utils.data.dataloader.default_collate

    # extract .data property of items in batch list
    data_batch = extract_data(batch)

    # TODO: rename vatiables and dict keys
    xl_ = []
    xf_ = []
    y_ = []
    labels_raw_ = []
    num_points_ = []

    seeds = []
    filtred_mask = []

    for d in data_batch:
        xl_.append(d['coords'])
        xf_.append(d['features'])
        if 'labels' in d:
            y = d['labels']
            y_.append(y)

        if 'labels_raw' in d:
            labels_raw_.append(d['labels_raw'])

        num_points_.append(y.shape[0])

        filtred_mask.append(d['filtred_mask'])

        if 'random_seed' in d:
            seeds.append(d['random_seed'])



    # Transform some keys to tensors

    # coords
    xl_ = [np.hstack([x, idx*np.ones((x.shape[0], 1), dtype='int64')])
           for idx, x in enumerate(xl_)]
    xl_ = np.vstack(xl_)
    xl_ = torch.from_numpy(xl_)
    # features
    xf_ = np.vstack(xf_)
    xf_ = torch.from_numpy(xf_)

    # labels
    y_ = torch.from_numpy(np.hstack(y_))

    ids = [d['id'] for d in data_batch]

    batch = {
        'coords': xl_,
        'features': xf_,
        'ids':          ids,
        'num_points':     num_points_,
        'filtred_mask': filtred_mask,
        'labels_raw': labels_raw_}
    if seeds:
        batch['seeds'] = seeds


    y = None
    if len(y_):
        #y = {'y': y_}
        y = y_

    return batch, y

