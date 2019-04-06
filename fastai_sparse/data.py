# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from os.path import join, exists, basename
import numbers

import torch
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

import fastai

from fastai.basic_data import DataBunch, DeviceDataLoader
from fastai.data_block import ItemList, ItemLists, PreProcessor, LabelLists
from fastai.core import array, PathOrStr, Callable, ifnone, defaults, Optional, TfmList, Collection, ItemBase, listify, is_listy
from fastai.imports import F
from fastai.torch_core import data_collate, try_int, to_data, ItemsList, Tensor, to_device
# from fastai.vision.data import _prep_tfm_kwargs

from typing import Any, Iterator, Dict, List
from torch.utils.data import Dataset


from . import utils
from .data_items import MeshItem, PointsItem, SparseItem
from . import transforms

# TODO:  use @dataclass


class DataSourceConfig():
    def __init__(self, root_dir,
                 df_item_list,
                 resolution=50,
                 spatial_size=50 * 8 + 8,
                 num_sample_points=None,
                 batch_size=64,
                 num_workers=defaults.cpus,
                 file_ext=None,
                 ply_label_name=None,
                 ply_colors_from_vertices=True,
                 ply_labels_from_vertices=True,
                 num_classes=10,
                 categories=None,
                 num_classes_by_category=None,
                 labels_shifting_in_file=0,
                 init_numpy_random_seed=True,  # TODO: rename and comment ?
                 ):
        self.root_dir = root_dir
        if not isinstance(root_dir, Path):
            self.root_dir = Path(root_dir)

        self.df = df_item_list.reset_index(drop=True)
        self.resolution = resolution
        self.spatial_size = spatial_size
        self.num_sample_points = num_sample_points
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.file_ext = file_ext
        self.ply_label_name = ply_label_name
        self.ply_colors_from_vertices = ply_colors_from_vertices
        self.ply_labels_from_vertices = ply_labels_from_vertices

        self.num_classes = num_classes
        self.categories = categories
        self.num_classes_by_category = num_classes_by_category
        self.labels_shifting_in_file = labels_shifting_in_file
        self.init_numpy_random_seed = init_numpy_random_seed

        if num_classes_by_category is not None:
            self.class_offsets = np.cumsum([0] + num_classes_by_category)
        else:
            # TODO: Set it when num_classes_by_category
            self.class_offsets = [0]
            self.num_classes_by_category = [self.num_classes]

        self.check()

    def check(self):
        assert self.root_dir.exists()
        # TODO:
        # - file_ext and df.ext
        # - check num_classes == sum(num_classes_by_category)
        if self.num_classes_by_category is not None:
            assert sum(self.num_classes_by_category) == self.num_classes

    def check_accordance(self, other, equal_keys=['resolution', 'spatial_size', 'num_classes', 'categories', 'num_classes_by_category', 'labels_shifting_in_file', 'ply_colors_from_vertices', 'ply_labels_from_vertices']):

        for key in equal_keys:
            v1 = getattr(self, key)
            v2 = getattr(other, key)
            assert v1 == v2, f"Key '{key}' is not equal"

    def __repr__(self) -> str:
        lines = [f"{self.__class__.__name__};"]
        for key in ['root_dir', 'resolution', 'spatial_size', 'num_sample_points', 'batch_size', 'num_workers', 'file_ext', 'ply_label_name', 'num_classes', 'categories', 'num_classes_by_category', 'class_offsets', 'init_numpy_random_seed']:
            value = getattr(self, key)
            if value is not None:
                lines.append(f'   {key}: {value}')

        value = len(self.df)
        lines.append(f' Items count: {value}')
        s = '\n'.join(lines)
        return s


def find_files(path, ext='.pts.train', ext_labels='.seg', categories=None):
    """
    Find files in the directory. Examples are in the dirs by categories.

    Returns
    -------
    pandas.DataFrame

    """
    if categories == 'auto':
        categories = [fn for fn in os.listdir(path)]
        categories = [fn for fn in categories if Path(path, fn).is_dir()]
        categories.sort()

    rows = []
    if categories is not None:
        for categ_idx, categ_dir in enumerate(categories):
            pattern = str(path / categ_dir / ('*' + ext))
            fnames = glob.glob(pattern)
            for fname in fnames:
                fname = Path(fname)
                row = {}
                row['example_id'] = fname.name.split('.')[0]
                row['subdir'] = categ_dir
                row['categ_idx'] = categ_idx
                row['ext'] = ext
                row['ext_labels'] = ext_labels

                rows.append(row)
    else:
        pattern = str(path / '*' / ('*' + ext))
        fnames = glob.glob(pattern)
        for fname in fnames:
            fname = Path(fname)
            row = {}
            row['example_id'] = fname.name.split('.')[0]
            row['ext'] = ext
            row['ext_labels'] = ext_labels
            rows.append(row)

    df = pd.DataFrame(rows)
    df = utils.df_order_columns(df, ['example_id', 'subdir', 'categ_idx'])
    return df


class BaseDataset(ItemList, Dataset):

    def __init__(self, items, source_config, **kwargs):
        self.tfms = None
        self.tfmargs = None
        self.source_config = source_config
        super().__init__(items, **kwargs)

    @classmethod
    def from_source_config(cls, source_config):
        fnames = []
        df = source_config.df
        t = tqdm(df.iterrows(), total=len(df), desc='Load file names')
        # TODO: try ... finelly t.clear t.close
        for i, row in t:
            fnames.append(cls.get_filename_from_row(row, source_config))
        o = cls(fnames, source_config=source_config,
                path=source_config.root_dir)
        return o

    @classmethod
    def get_filename_from_row(cls, row, source_config):
        fname = source_config.root_dir

        if 'subdir' in row.keys():
            fname = fname / row['subdir']

        # if 'categ_dir' in row.keys():
        #    fname = fname / row['categ_dir']

        ext = source_config.file_ext
        assert (ext is not None) or 'ext' in row.keys(
        ), "Define file_ext in config or column 'ext' in DataFrame'"
        if ext is None:
            ext = row.ext
        return fname / (row.example_id + ext)

    def __len__(self):
        assert len(self.source_config.df) == len(self.items)
        return len(self.items)

    def get_filename(self, i):
        return self.items[i]

    def get_example_id(self, i):
        row = self.source_config.df.iloc[i]
        return row.example_id

    def __getitem__(self, idxs: int) -> Any:
        idxs = try_int(idxs)
        if isinstance(idxs, numbers.Integral):
            item = self.get(idxs)
            if self.tfms or self.tfmargs:
                item = item.apply_tfms(self.tfms, **self.tfmargs)
            return item
        # else: return self.new(self.items[idxs], xtra=index_row(self.xtra, idxs))
        else:
            raise NotImplementedError()

    def transform(self, tfms: TfmList, **kwargs):
        "Set the `tfms` to be applied to the inputs and targets."
        self.tfms = tfms
        self.tfmargs = kwargs
        return self

    def check(self):
        self.check_files_exists()

    def check_files_exists(self, max_num_examples: Optional[int] = None, desc='Check files exist'):
        total = len(self)
        if max_num_examples is not None:
            total = int(max_num_examples)

        t = tqdm(self.items, total=total, desc=desc)
        try:
            for i, item in enumerate(t):
                assert item.exists()
                if max_num_examples is not None:
                    if i >= total:
                        break
        except Exception as e:
            # t.clear()
            t.close()
            print(item)
            raise e

    def calc_class_offset(self, i):
        # TODO: Use it when num_classes_by_category is not None
        sc = self.source_config
        row = sc.df.iloc[i]
        d = {}
        # TODO: Check when   num_classes_by_category is not None and 'categ_idx' not in DataFrame
        if 'categ_idx' in row.keys():

            categ_idx = row['categ_idx']
            d['categ'] = categ_idx
            d['class_offset'] = int(sc.class_offsets[categ_idx])
            d['num_classes'] = sc.num_classes_by_category[categ_idx]
            # TODO: rename num_classes?
        return d


class PointsDataset(BaseDataset):

    def get(self, i):

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

        data.update(self.calc_class_offset(i))

        y = self.get_labels(i)
        if y is not None:
            data['labels'] = y + data['class_offset']

        return PointsItem(data)

    def get_labels(self, i):
        fn = self.get_labels_filename(i)
        if fn.exists():
            if fn.name[-4:] == '.npy':
                y = np.load(fn)
            else:
                y = np.loadtxt(fn)
            y = y.astype('int64')
            shifting = self.source_config.labels_shifting_in_file
            if shifting:
                y = y - shifting
            return y

    def get_labels_filename(self, i):
        fname = self.source_config.root_dir
        row = self.source_config.df.iloc[i]
        if 'subdir' in row.keys():
            fname = fname / row['subdir']

        return fname / (row.example_id + row.ext_labels)

    def check_num_classes(self, max_num_examples: Optional[int] = 100, desc='Check num classes'):
        sc = self.source_config
        df = sc.df
        if sc.num_classes_by_category is not None:
            assert np.sum(sc.num_classes_by_category) == sc.num_classes
            assert 'subdir' in df.columns

            assert len(df.categ_idx.unique()) == len(
                sc.num_classes_by_category)

            for categ_idx in df.categ_idx.unique():
                df_categ = df[df.categ_idx == categ_idx]

                num_classes_in_category = sc.num_classes_by_category[categ_idx]

                total = len(df_categ)
                if max_num_examples is not None:
                    total = int(max_num_examples)

                t = tqdm(df_categ.iloc[:total].iterrows(), total=total, desc=f'{desc} {categ_idx}')

                try:
                    for i, (idx, row) in enumerate(t):
                        y = self.get_labels(idx)
                        if y is not None:
                            u = np.unique(y)
                            u_max = u.max()
                            assert u.min() >= 0
                            assert u_max + 1 >= len(u)
                            assert u_max + 1 <= num_classes_in_category

                except Exception as e:
                    t.clear()
                    t.close()
                    raise e


class MeshesDataset(BaseDataset):
    # _bunch = SparseDataBunch

    def check(self, file_exists: Optional[bool] = True, num_classes: Optional[bool] = False, max_num_examples: Optional[int] = None):
        is_loading_files_needed = False
        if num_classes:
            is_loading_files_needed = True

        t = None
        total = len(self)
        if max_num_examples is not None:
            total = int(max_num_examples)

        if file_exists:
            desc = 'Check files exist'
            if num_classes:
                desc = 'Check files exist and num classes'
            if is_loading_files_needed:
                t = tqdm(self.items, total=total, desc=desc)
            else:
                t = tqdm(self.items, total=total, desc=desc)
        elif num_classes:
            desc = 'Check num classes'
            t = tqdm(self, total=total, desc=desc)

        if t:
            try:
                for i, item in enumerate(t):
                    if num_classes:
                        mesh = self.load_mesh(i)
                        mesh_data = self.get_mesh_data(mesh)
                        labels = mesh_data['labels']
                        assert labels.max() <= self.source_config.num_classes
                        assert len(np.unique(labels)
                                   ) <= self.source_config.num_classes
                    else:
                        assert item.exists(), f"File {item} not exists"
                    if max_num_examples is not None:
                        if i >= total:
                            break
            except Exception as e:
                # t.clear()
                t.close()
                print(item)
                raise e

    def get(self, i):
        # data = self.load_and_process(i, self.source_config.resolution)
        # return SparseItem(data)
        fn = self.get_filename(i)
        example_id = self.get_example_id(i)

        sc = self.source_config
        # load mesh
        mesh_item = MeshItem.from_file(fn, example_id,
                                       label_field=sc.ply_label_name,
                                       colors_from_vertices=sc.ply_colors_from_vertices,
                                       labels_from_vertices=sc.ply_labels_from_vertices,
                                       )

        return mesh_item


class SparseDataBunch(DataBunch):
    "DataBunch suitable for SparseConvNet."

    # from fastai.basic_data
    #   class DataBunch.__init__
    # Fix: DeviceDataLoader

    def __init__(self, train_dl: DataLoader, valid_dl: DataLoader, fix_dl: DataLoader = None, test_dl: Optional[DataLoader] = None,
                 device: torch.device = None, dl_tfms: Optional[Collection[Callable]] = None, path: PathOrStr = '.',
                 collate_fn: Callable = data_collate, no_check: bool = False):
        self.dl_tfms = listify(dl_tfms)
        self.device = defaults.device if device is None else device
        assert not isinstance(train_dl, SparseDeviceDataLoader)

        def _create_dl(dl, **kwargs):
            if dl is None:
                return None
            return SparseDeviceDataLoader(dl, self.device, self.dl_tfms, collate_fn, **kwargs)
        self.train_dl, self.valid_dl, self.fix_dl, self.test_dl = map(
            _create_dl, [train_dl, valid_dl, fix_dl, test_dl])
        if fix_dl is None:
            self.fix_dl = self.train_dl.new(shuffle=False, drop_last=False)
        self.single_dl = _create_dl(DataLoader(
            valid_dl.dataset, batch_size=1, num_workers=0))
        self.path = Path(path)
        if not no_check:
            self.sanity_check()

    # from fastai.vision.data
    #   ImageDataBunch.create_from_ll   (from label lists)
    @classmethod
    def create_from_ll(cls, lls: LabelLists, bs: int = 64, ds_tfms: Optional[TfmList] = None,
                       num_workers: int = defaults.cpus, dl_tfms: Optional[Collection[Callable]] = None, device: torch.device = None,
                       test: Optional[PathOrStr] = None, collate_fn: Callable = data_collate, size: int = None, no_check: bool = False,
                       **kwargs) -> 'ImageDataBunch':
        "Create an `ImageDataBunch` from `LabelLists` `lls` with potential `ds_tfms`."
        # ds_tfms, kwargs = _prep_tfm_kwargs(ds_tfms, size, kwargs)
        lls = lls.transform(tfms=ds_tfms, size=size, **kwargs)
        if test is not None:
            lls.add_test_folder(test)
        return lls.databunch(bs=bs, dl_tfms=dl_tfms, num_workers=num_workers, collate_fn=collate_fn, device=device, no_check=no_check)

    # from fastai.basic_data
    #   class DataBunch.create
    # Fix:
    #   + worker_init_fn
    #   - batch_size
    #   - num_workers
    @classmethod
    def create(cls, train_ds: Dataset, valid_ds: Dataset, test_ds: Optional[Dataset] = None, path: PathOrStr = '.', bs: Optional[int] = None,
               num_workers: Optional[int] = None, dl_tfms: Optional[Collection[Callable]] = None, device: torch.device = None,
               collate_fn: Optional[Callable] = None, no_check: bool = False) -> 'DataBunch':
        "Create a `DataBunch` from `train_ds`, `valid_ds` and maybe `test_ds` with a batch size of `bs`."

        if collate_fn is None:
            collate_fn = cls.merge_fn

        if num_workers is not None:
            utils.warn_always(
                "num_workers is ignored. Getted from dataset configurations")
            # warnings.warn("'num_workers' is ignored. Getted from dataset configurations", UserWarning)

        if bs is not None:
            utils.warn_always(
                "'bs' is ignored. Getted from dataset configurations")
            # warnings.warn("'bs' is ignored. Getted from dataset configurations", UserWarning)

        datasets = cls._init_ds(train_ds, valid_ds, test_ds)

        bs = train_ds.source_config.batch_size
        val_bs = valid_ds.source_config.batch_size

        nw = train_ds.source_config.num_workers
        val_nw = valid_ds.source_config.num_workers

        ir = train_ds.source_config.init_numpy_random_seed
        val_ir = valid_ds.source_config.init_numpy_random_seed

        dls = []
        for d, b, s, nw, ir in zip(datasets,
                                   (bs, val_bs, val_bs, val_bs),
                                   (True, False, False, False),
                                   (nw, val_nw, val_nw, val_nw),
                                   (ir, val_ir, val_ir, val_ir),
                                   ):
            if d is not None:
                if ir:
                    ir_fn = cls.worker_init_fn
                else:
                    ir_fn = None
                dl = DataLoader(d, b, shuffle=s, drop_last=s,
                                num_workers=nw, worker_init_fn=ir_fn)
                dls.append(dl)

        # dls = [DataLoader(d, b, shuffle=s, drop_last=s, num_workers=nw, worker_init_fn=worker_init_fn) for d, b, s, nw in
        #       zip(datasets, (bs, val_bs, val_bs, val_bs), (True, False, False, False), (nw, val_nw, val_nw, val_nw)) if d is not None]
        return cls(*dls, path=path, device=device, dl_tfms=dl_tfms, collate_fn=collate_fn, no_check=no_check)

    # from fastai.basic_data
    #   class DataBunch._init_ds
    # fix: comment .x, .y
    @staticmethod
    def _init_ds(train_ds: Dataset, valid_ds: Dataset, test_ds: Optional[Dataset] = None):
        # train_ds, but without training tfms
        # fix_ds = valid_ds.new(train_ds.x, train_ds.y) if hasattr(valid_ds,'new') else train_ds
        fix_ds = None
        return [o for o in (train_ds, valid_ds, fix_ds, test_ds) if o is not None]

    # TODO: define as parameter of task/experiment
    @staticmethod
    def merge_fn(batch: ItemsList) -> Tensor:
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

        # exstract .data property of items in batch list
        data_batch = to_data(batch)

        # TODO: remove hardcore
        nClassesTotal = 17

        # TODO: rename vatiables and dict keys
        xl_ = []
        xf_ = []
        y_ = []
        categ_ = []
        mask_ = []
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

        batch = {  # 'x':  [xl_, xf_],
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

    @staticmethod
    def worker_init_fn(worker_id, verbose=0):
        """
        Initialize numpy random seed.

        Notes
        -----

        [1] https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
        By default, each worker will have its PyTorch seed set to base_seed + worker_id, where base_seed is a long generated by
        main process using its RNG. However, seeds for other libraies may be duplicated upon initializing workers (w.g., NumPy),
        causing each worker to return identical random numbers.
        (See My data loader workers return identical random numbers section in FAQ.)
        You may use torch.initial_seed() to access the PyTorch seed for each worker in worker_init_fn, and use it to set other
        seeds before data loading.


        [2] https://discuss.pytorch.org/t/dataloader-multi-threading-random-number/27719
        torch.initial_seed() == base_seed + worker_id
        and base_seed is different every epoch.
        """
        rs = int(torch.initial_seed())
        # Seed must be between 0 and 2**32 - 1 (4 294 967 295)
        base = (2 ** 32 - 1)
        rs = rs % base
        np.random.seed(rs)

        if verbose:
            d = utils.get_random_states(use_cuda=False)
            print('worker_id:', worker_id, d)

    @property
    def loss_func(self) -> Dataset:
        return getattr(self.train_ds, 'loss_func', F.cross_entropy)

    def describe(self):
        SequentialSampler = torch.utils.data.sampler.SequentialSampler

        def print_dl(title, dl):
            is_shuffle = not isinstance(dl.sampler, SequentialSampler)

            print(f"{title}: {len(dl.dataset):5}, shuffle: {str(is_shuffle):>5}, batch_size: {dl.batch_size:2}, num_workers: {dl.num_workers:2}, num_batches: {len(dl)}, drop_last: {dl.drop_last}")

        print_dl('Train', self.train_dl)
        print_dl('Valid', self.valid_dl)
        if self.test_dl is not None:
            print_dl('Test', self.test_dl)


MeshesDataset._bunch = SparseDataBunch


class SparseDeviceDataLoader(DeviceDataLoader):
    "Bind a `DataLoader` to a `torch.device`."

    # fix: classname DeviceDataLoader --> SparseDeviceDataLoader
    def new(self, **kwargs):
        "Create a new copy of `self` with `kwargs` replacing current values."
        new_kwargs = {**self.dl.init_kwargs, **kwargs}
        return SparseDeviceDataLoader(self.dl.__class__(self.dl.dataset, **new_kwargs), self.device, self.tfms,
                                      self.collate_fn)

    # Fix: work with dictionary of tensors: to_device --> dict_to_device
    def proc_batch(self, b: Tensor) -> Tensor:
        "Proces batch `b` of `TensorImage`."
        # b = to_device(b, self.device)
        b = dict_to_device(b, self.device, keys=[
                           'coords', 'features', 'y', 'mask'])
        for f in listify(self.tfms):
            b = f(b)
        return b


def dict_to_device(b: Dict, device: torch.device, keys: Optional[List] = None):
    "Recursively put `b` on `device`."
    device = ifnone(device, defaults.device)
    # print(b)
    # print(len(b))
    # print(is_listy(b))

    # if (x, y) then recourse
    if is_listy(b):
        return [dict_to_device(o, device, keys) for o in b]

    # if dict then to_device by keys
    if isinstance(b, dict):
        r = {}
        for key in b.keys():
            v = b[key]
            if keys is None or key in keys:
                assert isinstance(v, torch.Tensor), f"An item with the key '{key}' is not a tensor: {type(v)}"
                r[key] = to_device(v, device)
            else:
                r[key] = v
        return r

    # if tensor
    return to_device(b, device)
