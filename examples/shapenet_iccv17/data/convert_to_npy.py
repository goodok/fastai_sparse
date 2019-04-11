#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[1]:


import os, sys
import time
import math
import numpy as np
import pandas as pd
import datetime
import glob
from IPython.display import display, HTML, FileLink
from pathlib import Path
from os.path import join, exists, basename, splitext
from matplotlib import pyplot as plt
from matplotlib import cm
import shutil
from tqdm import tqdm
from joblib import Parallel, delayed, cpu_count

# autoreload python modules on the fly when its source is changed
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# In[2]:


#from IPython.core.display import display, HTML\n",
# display(HTML("<style>.container { width:70% !important; }</style>"))


# In[3]:


fastai_path = '../'
assert os.path.exists(fastai_path)
sys.path.append(fastai_path)

from fastai_sparse import utils
from fastai_sparse.utils import log


# In[4]:


n_jobs = cpu_count()    
n_jobs


# # Results dir

# In[5]:


results_dir = 'npy'

print("results_dir:" , results_dir)

if not exists(results_dir):
    os.makedirs(results_dir)


# # Train / Valid Dataset 

# ## Create DataFrames

# In[6]:



SOURCE_DIR = Path('./').expanduser()
assert SOURCE_DIR.exists()

DIR_TRAIN_VAL = SOURCE_DIR / 'train_val'
assert DIR_TRAIN_VAL.exists(), "Hint: run `download_and_split_data.sh`"

print(SOURCE_DIR)
print(DIR_TRAIN_VAL)


# In[7]:


categories = [
    "02691156", "02773838", "02954340", "02958343", "03001627", "03261776",
    "03467517", "03624134", "03636649", "03642806", "03790512", "03797390",
    "03948459", "04099429", "04225987", "04379243"
]

classes = [
    'Airplane', 'Bag', 'Cap', 'Car', 'Chair', 'Earphone', 'Guitar', 'Knife',
    'Lamp', 'Laptop', 'Motorbike', 'Mug', 'Pistol', 'Rocket', 'Skateboard',
    'Table'
]

num_classes_by_category = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]

len(categories), len(classes)


# In[8]:


def get_trainval_df(path=DIR_TRAIN_VAL, categ='all', categories=categories, ext='.pts.train'):
    rows = []
    if categ == 'all':
        for categ_idx, categ_dir in enumerate(categories):
            pattern = str(path / categ_dir / ('*' + ext))
            fnames = glob.glob(pattern)
            for fname in fnames:
                fname = Path(fname)
                row = {}
                row['example_id'] = fname.name.split('.')[0]
                row['categ_dir'] = categ_dir
                row['categ_idx'] = categ_idx
                row['ext'] = ext
                row['ext_labels'] = '.seg'
                
                rows.append(row)
    df = pd.DataFrame(rows)
    df = utils.df_order_columns(df, ['example_id', 'categ_dir'])
    return df


# In[9]:


df_train = get_trainval_df()
df_valid = get_trainval_df(ext='.pts.valid')

print(len(df_train))
df_train.head()


# In[10]:


print(len(df_valid))
df_valid.head()


# # Test initial speed

# In[11]:


def get_file_name(row, rootpath=DIR_TRAIN_VAL, labels=False):
    ext = row.ext
    if labels:
        ext = row.ext_labels
    return rootpath / row.categ_dir / f"{row.example_id}{ext}"


# Test reading speed from `*.pts` files.  
# 100 examples

# In[12]:


df_sample = df_train.sample(len(df_train))
df_sample = df_sample.head(100)


# In[13]:


t = tqdm(df_sample.iterrows(), total=len(df_sample))
for i, row in t:
    fn = get_file_name(row)
    x = np.loadtxt(fn)
    
    fn = get_file_name(row, labels=True)
    y = np.loadtxt(fn) - 1
    pass


# # Convert

# ## Convert one

# In[14]:


root_dest = Path(results_dir, 'train')

if root_dest.exists():
    shutil.rmtree(root_dest)
os.makedirs(str(root_dest))

for categ_dir in df_train.categ_dir.unique():
    categ_dir = root_dest / categ_dir
    
    if not categ_dir.exists():
        os.makedirs(str(categ_dir))
    


# In[15]:


def convert_one(row, root_source=DIR_TRAIN_VAL, root_dest=Path(results_dir, 'train')):
    
    fn = get_file_name(row, root_source)
    x = np.loadtxt(fn).astype(np.float32)
    
    y = None
    fn = get_file_name(row, root_source, labels=True)
    if fn.exists():
        y = np.loadtxt(fn).astype(np.int32)
        
    fn_res = root_dest / row.categ_dir / f'{row.example_id}.points.npy'
    np.save(fn_res, x)
    
    fn_res = root_dest / row.categ_dir / f'{row.example_id}.labels.npy'
    np.save(fn_res, y)
    
    
def convert_df(df, root_source, root_dest, n_jobs=8):
    if root_dest.exists():
        shutil.rmtree(root_dest)

    os.makedirs(str(root_dest))
    print(len(df), root_source, '---->', root_dest)
    sys.stdout.flush()
    
    categories = df.categ_dir.unique()
    t = tqdm(categories, total=len(categories), desc='Make categories subdirs')
    try:
        for categ_dir in t:
            categ_dir = root_dest / categ_dir

            if not categ_dir.exists():
                os.makedirs(str(categ_dir))
    finally:
        t.clear()
        t.close()
        sys.stderr.flush()
            
    sys.stdout.flush()
    t = tqdm(df.iterrows(), total=len(df), desc="Convert files")

    try:
        res = Parallel(n_jobs=n_jobs)(delayed(convert_one)(row, root_source, root_dest) for i, row in t)

    finally:
        t.clear()
        t.close()
        sys.stderr.flush()
  


# In[16]:


row = df_train.iloc[0]
convert_one(row, root_source=DIR_TRAIN_VAL, root_dest=Path(results_dir, 'train'))


# ## Convert all

# In[17]:


df = df_train
root_source = DIR_TRAIN_VAL
root_dest = Path(results_dir, 'train')


# In[18]:


convert_df(df, root_source, root_dest, n_jobs=n_jobs)


# In[19]:


convert_df(df_valid, DIR_TRAIN_VAL, Path(results_dir, 'valid'), n_jobs=n_jobs)


# In[20]:


# TODO:
# convert_df(df_test, SOURCE_DIR / 'test', Path(results_dir, 'test'))


# # Test speed

# Test reading speed from `*.npy` files.  
# All ~ 7000 examples

# In[21]:


df_sample = df_train.sample(len(df_train))
#df_sample = df_sample.head(100)
df_sample.ext = '.points.npy'
df_sample.ext_labels = '.labels.npy'


# In[22]:


path = Path(results_dir, 'train')


# In[23]:


t = tqdm(df_sample.iterrows(), total=len(df_sample))
for i, row in t:
    fn = get_file_name(row, rootpath=path)
    x = np.load(fn)
    fn = get_file_name(row, rootpath=path, labels=True)
    y = np.load(fn)
    pass


