# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import numbers
import glob
import os
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass
# from abc import abstractmethod

from torch.utils.data import Dataset

from . import utils
from .core import defaults, Any, Collection, Callable, Optional, try_int, PathOrStr, listify, PreProcessors, show_some
from .data_items import MeshItem, PointsItem


@dataclass
class DataSourceConfig():
    root_dir: PathOrStr
    df: Any
    batch_size: int = 32
    num_workers: int = defaults.cpus
    init_numpy_random_seed: bool = True   # TODO: rename and comment ?

    file_ext: str = None
    ply_label_name: str = None
    ply_colors_from_vertices: bool = True
    ply_labels_from_vertices: bool = True

    def __post_init__(self):
        if not isinstance(self.root_dir, Path):
            self.root_dir = Path(self.root_dir)

        self.df = self.df.reset_index(drop=True)

        self._equal_keys = ['ply_colors_from_vertices',
                            'ply_labels_from_vertices']

        self._repr_keys = ['root_dir', 'batch_size', 'num_workers',
                           'file_ext', 'ply_label_name',
                           'init_numpy_random_seed']

        self.check()

    def check(self):
        assert self.root_dir.exists()

        # TODO:
        # - check file_ext and df.ext

    def check_accordance(self, other):

        for key in self._equal_keys:
            v1 = getattr(self, key)
            v2 = getattr(other, key)
            assert v1 == v2, f"Key '{key}' is not equal"

    def __repr__(self) -> str:
        lines = [f"{self.__class__.__name__};"]
        for key in self._repr_keys:
            value = getattr(self, key)
            if value is not None:
                lines.append(f'   {key}: {value}')

        value = len(self.df)
        lines.append(f' Items count: {value}')
        s = '\n'.join(lines)
        return s


class BaseDataset(Dataset):

    def __init__(self, items, source_config=None, path: PathOrStr = '.', reader_fn: Callable = None, **kwargs):
        """
        Parameters
        ----------
        items: Collection
            Filenames of examples. For fastai compatibility. (TODO: reorganize it)
        reader_fn: Callable
            Function (self, i, row) that return instance of ItemBase or its subclasses.


        """
        # TODO: store `list(df.values)` in items.
        self.items = items
        self.source_config = source_config
        self.df = source_config.df
        self.path = Path(path)
        self.reader_fn = reader_fn
        self.tfms = None
        self.tfmargs = None

    # .. from fastai  ..
    def process(self, processor: PreProcessors = None):
        "Apply `processor` or `self.processor` to `self`."
        if processor is not None:
            self.processor = processor
        self.processor = listify(self.processor)
        for p in self.processor:
            p.process(self)
        return self

    # .. key methods ..
    def __len__(self):
        assert len(self.source_config.df) == len(self.items)
        return len(self.items)

    def get(self, i):
        row = self.df.iloc[i]

        if self.reader_fn is not None:
            return self.reader_fn(i, row, self=self)
        else:
            return self._reader_fn(i, row)

    def __getitem__(self, idxs: int) -> Any:
        """Return instance ItemBase or subclasses. With transformations."""
        idxs = try_int(idxs)
        if isinstance(idxs, numbers.Integral):
            item = self.get(idxs)
            if self.tfms or self.tfmargs:
                item = item.apply_tfms(self.tfms, **self.tfmargs)
            return item
        # else: return self.new(self.items[idxs], xtra=index_row(self.xtra, idxs))
        else:
            raise NotImplementedError()

    def __add__(self, other):
        # return ConcatDataset([self, other])
        raise NotImplementedError()

    def __repr__(self) -> str:
        items = [self[i] for i in range(min(5, len(self.items)))]
        return f'{self.__class__.__name__} ({len(self.items)} items)\n{show_some(items)}\nPath: {self.path}'

    def transform(self, tfms: Collection, **kwargs):
        "Set the `tfms` to be applied to the inputs and targets."
        self.tfms = tfms
        self.tfmargs = kwargs
        return self

    # .. other methods ..
    def get_filename(self, i):
        return self.items[i]

    def get_example_id(self, i):
        row = self.source_config.df.iloc[i]
        return row.example_id

    @classmethod
    def from_source_config(cls, source_config, reader_fn=None):
        fnames = []
        df = source_config.df
        t = tqdm(df.iterrows(), total=len(df), desc='Load file names')
        try:
            for i, row in t:
                fnames.append(cls.get_filename_from_row(row, source_config))
        finally:
            t.clear()
            t.close()
        o = cls(fnames, source_config=source_config,
                reader_fn=reader_fn,
                path=source_config.root_dir)
        return o

    @classmethod
    def get_filename_from_row(cls, row, source_config):
        fname = source_config.root_dir

        if 'subdir' in row.keys():
            fname = fname / row['subdir']

        ext = source_config.file_ext
        assert (ext is not None) or 'ext' in row.keys(
        ), "Define file_ext in config or column 'ext' in DataFrame'"
        if ext is None:
            ext = row.ext
        return fname / (row.example_id + ext)

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


class PointsDataset(BaseDataset):

    def _reader_fn(self, i, row):
        """Returns instance of ItemBase by its index and supplimented dataframe's row

           Default reader.
           Used if self.reader_fn is None.
        """
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

        data['labels'] = self.get_labels(i)

        return PointsItem(data)

    def get_labels(self, i):
        fn = self.get_labels_filename(i)
        if fn.exists():
            if fn.name[-4:] == '.npy':
                y = np.load(fn)
            else:
                y = np.loadtxt(fn)
            y = y.astype('int64')
            return y

    def get_labels_filename(self, i):
        fname = self.source_config.root_dir
        row = self.source_config.df.iloc[i]
        if 'subdir' in row.keys():
            fname = fname / row['subdir']

        return fname / (row.example_id + row.ext_labels)

    # TODO: move somewere, for shapenet example
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


# TODO: move it somewere, e.g. to utils
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
