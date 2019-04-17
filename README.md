# fastai_sparse

This is an extension of the [fast.ai](https://github.com/fastai/fastai) library to train  Submanifold Sparse Convolution Networks that apply to 2D/3D sparse data, such as 3D geometric meshes or point clouds in euclidian space

Currently, this library has [SparseConvNet](https://github.com/facebookresearch/SparseConvNet) under the hood which is the best in 3D (ScanNet benchmark, ShapeNet workshop) so far.


# Installation

**fastai_sparse** is compatible with: Python 3.6, PyTorch 1.0+

**Some key dependences:**  
- [Fast.ai](https://github.com/fastai/fastai#installation)  
- PyTorch sparse convolution models: [SparseConvNet](https://github.com/facebookresearch/SparseConvNet). 
- PLY file reader and 3D geometry mesh transforms are implemented by [trimesh](https://github.com/mikedh/trimesh).    
- [ipyvolume](http://ipyvolume.readthedocs.io/) is used for interactive visualisation in jupyter notebooks examples.

See details in [INSTALL.md](INSTALL.md)


# Features:
* fast.ai train/inference loop concept (Model + DataBunch --> Learner)
<a href="https://goodok.github.io/fastai_sparse/overview/classes.svg">Classes overview</a>  
* model training best practices provided by fast.ai (Learning Rate Finder, One Cycle policy)  
* 3D transforms for data preprocessing and augmentation:  
  - mesh-level transforms and features extraction (surface normals, triangle area,...)  
  - points-level spatial transforms (affine, elastic,...)  
  - points-level features (color, brightness)  
  - mesh to points
  - points to sparse voxels
* metrics (IoU, avgIoU, ) calculation and tracking
* visualization utils (batch generator output)  

# Notebooks with examples
- [x] 3D Transformation examples [notebook](https://nbviewer.jupyter.org/github/goodok/fastai_sparse/blob/master/notebooks/transforms/transforms.ipynb)
- [x] ScanNet 3D indoor scene semantic segmentation [detailed notebook](https://nbviewer.jupyter.org/github/goodok/fastai_sparse/blob/master/examples/scannet/scannet_unet_detailed.ipynb)
- [x] ScanNet 3D example with surface normals [notebook](https://nbviewer.jupyter.org/github/goodok/fastai_sparse/blob/master/examples/scannet_normals/unet_normals_detailed.ipynb)
- [x] ShapeNet 3D semantic segmentation [detailed notebook](https://nbviewer.jupyter.org/github/goodok/fastai_sparse/blob/master/examples/shapenet_iccv17/unet_24_detailed.ipynb)

# TODO

## Priority 1:
- [ ] Separate 3D augmentation library with key points, spatial targets
- [ ] Prediction pipeline
- [ ] Classification/regression examples
- [ ] Spatial targets (bounding box, key points, axes)


## Priority 2:

- [ ] TTA
- [ ] Multi-GPU
- [ ] PointNet-like feature extraction layer ("VoxelNet" architecture)
- [ ] Confidence / heatmap / kernels visualization 
