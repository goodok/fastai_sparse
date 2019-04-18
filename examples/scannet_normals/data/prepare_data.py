#!/usr/bin/env python
# coding: utf-8

# # Plan
#
# - merge two types of file
#     - Obtain vertices, faces, colors from first file, but labels from second file.
#     - And save in single joined file with extantion `.ply` in `scannet_merged_ply` directory
#
# - debug transformations

# # Imports

import os
import sys
import numpy as np
import pandas as pd
import glob
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed, cpu_count

from fastai_sparse.visualize.utils import export_ply
from fastai_sparse.data_items import MeshItem

n_jobs = cpu_count()
n_jobs


# Source

SOURCE_DIR = Path('scannet')
assert SOURCE_DIR.exists()

definition_of_spliting_dir = Path('ScanNet_Tasks_Benchmark')
assert definition_of_spliting_dir.exists()

# Target

TARGET_DIR = Path('scannet_merged_ply')

if not TARGET_DIR.exists():
    os.mkdir(str(TARGET_DIR))


def find_files(path, ext='_vh_clean_2.ply'):
    pattern = str(path / '*' / ('*' + ext))
    fnames = glob.glob(pattern)
    return fnames


print("Number of files found:", len(find_files(SOURCE_DIR)))


#  train / valid / test lists

fn_lists = {}

fn_lists['train'] = definition_of_spliting_dir / 'scannetv1_train.txt'
fn_lists['valid'] = definition_of_spliting_dir / 'scannetv1_val.txt'
fn_lists['test'] = definition_of_spliting_dir / 'scannetv1_test.txt'

for datatype in ['train', 'valid', 'test']:
    assert fn_lists[datatype].exists(), datatype

# Load lists

dfs = {}
for datatype in ['train', 'valid', 'test']:
    df = pd.read_csv(fn_lists[datatype], header=None, names=['scene_id'])
    df = df.assign(datatype=datatype)
    dfs[datatype] = df

    print(f"{datatype} counts: {len(df)}")


print("Valid head():", dfs['valid'].head())

df = pd.concat([dfs['train'], dfs['valid'], dfs['test']])
print("All head():", df.head())

# ## Check existence


files_exts = ['_vh_clean_2.ply', '_vh_clean_2.labels.ply']

t = tqdm(df.iterrows(), total=len(df), desc='Check files exist')
try:
    for i, row in t:
        for ext in files_exts:
            fn = SOURCE_DIR / f"{row.scene_id}" / f"{row.scene_id}{ext}"
            assert fn.exists(), fn
finally:
    t.clear()
    t.close()


# # Convert one


def merge_one_row(row):
    """
    Obtain vertices, faces, colors from first file, but labels from second file.

    And save in single joined file with extantion `.merged.ply`
    """

    fn = SOURCE_DIR / f"{row.scene_id}" / f"{row.scene_id}{files_exts[0]}"
    fn2 = SOURCE_DIR / f"{row.scene_id}" / f"{row.scene_id}{files_exts[1]}"

    dir_out = TARGET_DIR / f"{row.scene_id}"
    fn_out = TARGET_DIR / f"{row.scene_id}" / f"{row.scene_id}.merged.ply"

    if not dir_out.exists():
        os.mkdir(dir_out)

    o = MeshItem.from_file(fn, colors_from_vertices=True)

    o2 = MeshItem.from_file(fn2, label_field='label', labels_from_vertices=True)
    labels = o2.labels

    # trimesh.exchange.ply.export_ply(o.data)
    res = export_ply(o.data, vertex_labels=labels.astype(np.uint16), label_type='ushort')
    with open(fn_out, "wb") as f:
        f.write(res)


row = df.iloc[0]
merge_one_row(row)

# Test loading of the merged file

fn_out = TARGET_DIR / f"{row.scene_id}" / f"{row.scene_id}.merged.ply"
o = MeshItem.from_file(fn_out)
o.describe()
print(o)


# # Convert all
t = tqdm(df.iterrows(), total=len(df), desc="Convert files")

try:
    res = Parallel(n_jobs=n_jobs)(delayed(merge_one_row)(row) for i, row in t)
    # for fn in t:
    #     convert_one(fn)
finally:
    t.clear()
    t.close()
    sys.stderr.flush()
