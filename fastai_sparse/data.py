# -*- coding: utf-8 -*-

import numpy as np
from warnings import warn

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from fastai.basic_data import DataBunch, DeviceDataLoader

from .datasets import MeshesDataset
from .data_items import extract_data
from .core import (show_some, defaults, listify, is_listy, F, to_device, ifnone,
                   PathOrStr, Callable, Optional, Collection, Tensor, Dict, List)
from . import utils


def data_collate(batch: Collection) -> Tensor:
    "Convert `batch` items to tensor data."
    return torch.utils.data.dataloader.default_collate(extract_data(batch))


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

    # from fastai.basic_data
    #   class DataBunch.create
    # Fix:
    #   + worker_init_fn
    #   + batch_size
    #   + num_workers
    @classmethod
    def create(cls, train_ds: Dataset, valid_ds: Dataset, test_ds: Optional[Dataset] = None, path: PathOrStr = '.', bs: Optional[int] = None,
               num_workers: Optional[int] = None, dl_tfms: Optional[Collection[Callable]] = None, device: torch.device = None,
               collate_fn: Optional[Callable] = None, no_check: bool = False) -> 'DataBunch':
        "Create a `DataBunch` from `train_ds`, `valid_ds` and maybe `test_ds`"

        if collate_fn is None:
            collate_fn = cls.merge_fn

        if num_workers is not None:
            utils.warn_always(
                "num_workers is ignored. Getted from dataset configurations")

        if bs is not None:
            utils.warn_always(
                "'bs' is ignored. Getted from dataset configurations")

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
                                num_workers=nw, worker_init_fn=ir_fn, pin_memory=False)
                dls.append(dl)

        return cls(*dls, path=path, device=device, dl_tfms=dl_tfms, collate_fn=collate_fn, no_check=no_check)

#    # from fastai.basic_data
#    #   class DataBunch._init_ds
#    # fix: comment .x, .y
#    @staticmethod
#    def _init_ds(train_ds: Dataset, valid_ds: Dataset, test_ds: Optional[Dataset] = None):
#        # train_ds, but without training tfms
#        # fix_ds = valid_ds.new(train_ds.x, train_ds.y) if hasattr(valid_ds,'new') else train_ds
#        fix_ds = None

#        return [o for o in (train_ds, valid_ds, fix_ds, test_ds) if o is not None]

    # from fastai.basic_data
    # fix: remove try/except
    def sanity_check(self):
        "Check the underlying data in the training set can be properly loaded."
        final_message = "You can deactivate this warning by passing `no_check=True`."
        if not hasattr(self.train_ds, 'items') or len(self.train_ds.items) == 0 or not hasattr(self.train_dl, 'batch_sampler'):
            return
        if len(self.train_dl) == 0:
            warn(f"""Your training dataloader is empty, you have only {len(self.train_dl.dataset)} items in your training set.
                 Your batch size is {self.train_dl.batch_size}, you should lower it.""")
            print(final_message)
            return
        idx = next(iter(self.train_dl.batch_sampler))
        samples, fails = [], []
        for i in idx:
            try:
                samples.append(self.train_dl.dataset[i])
            except:
                fails.append(i)
        if len(fails) > 0:
            warn_msg = "There seems to be something wrong with your dataset, for example, in the first batch can't access"
            if len(fails) == len(idx):
                warn_msg += f" any element of self.train_ds.\nTried: {show_some(idx)}"
            else:
                warn_msg += f" these elements in self.train_ds: {show_some(fails)}"
            warn(warn_msg)
            print(final_message)
            return
        # try:
        _ = self.collate_fn(samples)
        # except:
        #     message = "It's not possible to collate samples of your dataset together in a batch."
        #     try:
        #         shapes = [[o[i].data.shape for o in samples] for i in range(2)]
        #         message += f'\nShapes of the inputs/targets:\n{shapes}'
        #     except: pass
        #     warn(message)
        # print(final_message)

    # TODO: define as parameter of task/experiment
    @staticmethod
    def merge_fn(batch: Collection,
                 keys_tensors=['coords', 'features', 'labels'],
                 keys_lists=['id', 'random_seed', 'num_points'],
                 separate_labels=True):
        """
        Convert `batch` items to merged batch with tensor data (dictionary).

        Merge fields of every example to the Tensor or just into the list.

        Parameters
        ----------
        batch: Collection
            The list of examples, instances if class  `ItemBase` or its inheritors.
        keys_tensors: list
            This keys will be converted to the Tensors
        keys_lists: list
            This keys will be converted to the lists
        separate_labels: bool
            if True then return tuple (batch, labels), otherwise return just batch (possible with 'labels' field)

        See also
        --------
            fastai.torch_core.data_collate
            torch.utils.data.dataloader.default_collate
        """

        # extract .data property of items in batch list
        data_batch = extract_data(batch)

        res = {}
        for k in keys_tensors:
            # merge to list
            a = [d[k] for d in data_batch if k in d]
            # append column with example's index. It is needed for SparseConvNet
            if k == 'coords':
                a = [np.hstack([x, idx * np.ones((x.shape[0], 1), dtype='int64')])
                     for idx, x in enumerate(a)]
            if a[0].ndim > 1:
                a = np.vstack(a)
            else:
                a = np.hstack(a)
            a = torch.from_numpy(a)
            res[k] = a

        for k in keys_lists:
            # merge to list
            a = [d[k] for d in data_batch if k in d]
            res[k] = a

        if 'num_points' in keys_lists:
            num_points = [len(d['coords']) for d in data_batch]
            res['num_points'] = num_points

        if separate_labels:
            y = None
            if 'labels' in res:
                y = res.pop('labels')
            return res, y
        else:
            return res

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

            print(f"{title}: {len(dl.dataset):5}, shuffle: {str(is_shuffle):>5}, batch_size: {dl.batch_size:2}, "
                  f"num_workers: {dl.num_workers:2}, num_batches: {len(dl)}, drop_last: {dl.drop_last}")

        print_dl('Train', self.train_dl)
        print_dl('Valid', self.valid_dl)
        if self.test_dl is not None:
            print_dl('Test', self.test_dl)


# TODO: remove it??
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
        # TODO: parameterize it.
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
