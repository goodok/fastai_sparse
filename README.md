# fastai_sparse

This is [fast.ai](https://github.com/fastai/fastai) library extension for training Sparse Convolution Networks applicable to 2D/3D sparse data like 3D geometry mesh or points cloud in euclidian space.  
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
fast.ai train/inference loop concept (Model + DataBunch --> Learner) 
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


- [ ] 3D scene semantic segmentation (ScanNet) notebook
- [x] Detailed 3D scene segmentation (ScanNet) notebook 
- [ ] ShapeNet 3D semantic segmentation

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
