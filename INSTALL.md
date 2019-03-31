# fastai_sparse

[Logo/Demo image]

# Installation
<font color="red">TBD</font>

**fastai_sparse** is compatible with: Python 3.6, PyTorch 1.0+

**Some key dependences:**  
- **[Fast.ai](https://github.com/fastai/fastai#installation)**  
- PyTorch sparse convolution models: **SparseConvNet** (https://github.com/facebookresearch/SparseConvNet )  
- PLY file reader and 3D geometry mesh transforms are implemented by **trimesh** (https://github.com/mikedh/trimesh)  
- For interactive visualisation in jupyter notebooks examples **ipyvolume** (http://ipyvolume.readthedocs.io/).

### Install dependences

####  Ubuntu 18.04


Normal install, go further to the next section "Pre instalation"


#### Ubuntu 16.04

On Ubuntu 16.04 python >3.5 is not installed

Check version of installed python:

```bash
python3.6  --version
python3.7  --version
python3  --version
```


If python 3.6 and python 3.7 are not installed then install python 3.6 manually, see [askubuntu.com](https://askubuntu.com/questions/865554/how-do-i-install-python-3-6-using-apt-get/865569#865569)

```bash
    sudo add-apt-repository ppa:deadsnakes/ppa
    sudo apt-get update
    sudo apt-get install python3.6

    sudo apt-get install python3.6-dev

    sudo apt-get install python3.6-tk
```

if instalation in virtual environment then

```bash
mkvirtualenv -p python3.6 <environmentname>
```

To run examples of SparseConvNet and if SparseConvNet is needed to compile, add

    export PYTHON_INCLUDE_PATH="/usr/include/python3.6m"

to ~/.virtualenvs/<environmentname>/bin/postactivate

then 

```bash
deactivate
workon <environmentname>
```




### Pre instalation

It would be nice if ~/.cache/pip is cleared if any problems encounters during instalation .

```bash
mv ~/.cache/pip ~/.cache/pip.bak
```

Also  backup ~/.jupyter  directory.

```bash
cp -r ~/.jupyter ~/.jupyter.before.bak
```

And if any problems clear it too.

```bash
mv ~/.jupyter ~/.jupyter.current.bak
```


Course bug of `bottleneck` install it first.

```bash
pip install bottleneck
pip install ipyvolume
```

### Install fastai_sparse

```bash
git clone https://github.com/goodok/fastai_sparse/
cd fastai_sparse
python setup.py develop

cd ../
```


### fastai


```bash
git clone https://github.com/fastai/fastai/
cd fastai
git checkout 33aae8f7b4b7d323d943c178d9ba58afcf8f19b8
python setup.py install
cd ../
```

##### Details

Clone fastai repository

```bash
git clone https://github.com/fastai/fastai/
cd fastai
```

View and choos releases

```bash
git branch -r

```

Current version is tested for the commit `33aae8f7b4b7d323d943c178d9ba58afcf8f19b8`, so switch to it:

```bash
git checkout 33aae8f7b4b7d323d943c178d9ba58afcf8f19b8
```

Install

```bash
python setup.py install
```

if error `AttributeError: 'dict' object has no attribute '__NUMPY_SETUP__' then 
```bash
pip install bottleneck
````

and try again python setup.py install

Exit if needed

```bash
cd ../
```


### SparseConvNet

It is needed for the running examples  which use SparseConvNet models.

See details for pre installations:
https://github.com/facebookresearch/SparseConvNet#setup

```bash
sudo apt-get install libsparsehash-dev
```

Compile and install

```bash
git clone https://github.com/facebookresearch/SparseConvNet
cd SparseConvNet
git checkout cd155261c151c0de34de02608c081b57be67ee5f
sh develop.sh
cd ../
```



### Tune jupyter extentions

See:  
https://github.com/ipython-contrib/jupyter_contrib_nbextensions


```bash
jupyter contrib nbextension install --user

jupyter nbextensions_configurator enable --user
```

Check panel "Configurable nbextensions"

http://127.0.0.1:<port>/tree#nbextensions_configurator


# Features:
* fast.ai train/inference loop concept (Model + DataBunch --> Learner) 
* model training best practices provided by fast.ai (Learning Rate Finder, One Cycle policy) 
* 3D transforms for data preprocessing and augmentation: 
  - mesh-level transforms and features extraction (surface normals, triangle area,...)      
  - points-level spatial transforms (affine, elastic,...)
  - points-level features (color, brightness)  
  - mesh to points
  - points to sparse voxels
* metrics (IoU, avgIoU, ) calculation and tracking 
* visualisation utils (batch generator output)  

# Quick start guide


* 3D scene semantic segmentation (ScanNet) notebook
* Detailed 3D scene segmentation (ScanNet) notebook 
* ShapeNet 3D semantic segmentation

# TODO

## Priority 1:
- [ ] ShapeNet example (with surface normals)
- [ ] ScanNet advanced example (large model with tweaks)
- [ ] Prediction pipeline
- [ ] TTA
- [ ] Classification/regression examples
- [ ] spatial targets (bounding box, key points, axes)


## Priority 2:
- [ ] 3D advansed augmentation library with key points, spatial targets
- [ ] multi-GPU
- [ ] PointNet-like feature extraction layer ("VoxelNet" architecture)
- [ ] confidence / heatmap / kernels visualization 

## Priority 3 (2020)
- 3D GAN, sparse pattern generative layer


