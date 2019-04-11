# -*- coding: utf-8 -*-

import numpy as np

import torch

from fastai_sparse.core import Collection
from fastai_sparse.data_items import extract_data


def merge_fn(batch: Collection):
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
    #  --- rename vatiables and dict keys

    # see
    #    fastai.torch_core.data_collate
    #    torch.utils.data.dataloader.default_collate

    # exstract .data property of items in batch list
    data_batch = extract_data(batch)

    xl_ = []
    xf_ = []
    y_ = []
    categ_ = []
    classOffset_ = []
    nClasses_ = []
    nPoints_ = []

    seeds = []

    for d in data_batch:
        xl_.append(d['coords'])
        xf_.append(d['features'])
        if 'labels' in d:
            y = d['labels']
            y_.append(y)

        categ_.append(d['categ'])
        classOffset_.append(d['class_offset'])
        nClasses_.append(d['num_classes'])

        nPoints_.append(y.shape[0])

        if 'random_seed' in d:
            seeds.append(d['random_seed'])

    # Transform some keys to tensors

    # coords
    xl_ = [np.hstack([x, idx * np.ones((x.shape[0], 1), dtype='int64')])
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
        'categ': categ_,   # TODO: is it needed ?
        'classOffset': classOffset_,
        'nClasses': nClasses_,
        'xf': ids,
        'nPoints': nPoints_}
    if seeds:
        batch['seeds'] = seeds

    y = None
    if len(y_):
        y = y_

    return batch, y
