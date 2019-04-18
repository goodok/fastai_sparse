# -*- coding: utf-8 -*-

import numpy as np
from dataclasses import dataclass

from fastai_sparse.datasets import DataSourceConfig as DataSourceConfig_Base
from fastai_sparse.core import Collection
from fastai_sparse.data_items import PointsItem


@dataclass
class DataSourceConfig(DataSourceConfig_Base):

    num_classes: int = 10
    categories: Collection = None
    num_classes_by_category: Collection = None

    def __post_init__(self):
        super().__post_init__()

        if self.num_classes_by_category is not None:
            self.class_offsets = np.cumsum([0] + self.num_classes_by_category)
        else:
            # TODO: Set it when num_classes_by_category
            self.class_offsets = [0]
            self.num_classes_by_category = [self.num_classes]

        self._equal_keys += ['num_classes', 'categories', 'num_classes_by_category']
        self._repr_keys += ['num_classes', 'categories', 'num_classes_by_category', 'class_offsets']

        # self.check()

    def check(self):
        super().check()

        if self.num_classes_by_category is not None:
            assert sum(self.num_classes_by_category) == self.num_classes

    def __repr__(self) -> str:
        return super().__repr__()


def reader_fn(i, row, self=None):
    """Returns instance of ItemBase by its index and supplimented dataframe's row"""
    sc = self.source_config
    data = {}
    data['id'] = self.get_example_id(i)

    fn = self.get_filename(i)
    if fn.name[-4:] == '.npy':
        x = np.load(fn)
    else:
        x = np.loadtxt(fn)

    data['points'] = x[:, 0:3]
    if x.shape[1] == 6:
        data['normals'] = x[:, 3:6]

    # shift labels acordence category
    categ_idx = row['categ_idx']
    data['categ'] = categ_idx
    data['class_offset'] = int(sc.class_offsets[categ_idx])
    data['num_classes'] = sc.num_classes_by_category[categ_idx]

    data['labels'] = self.get_labels(i)

    # TODO: move to transforms
    if data['labels'] is not None:
        data['labels'] += data['class_offset']

        return PointsItem(data)
