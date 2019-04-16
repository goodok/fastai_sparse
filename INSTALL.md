# fastai_sparse

# Installation

**fastai_sparse** is compatible with: Python 3.6, PyTorch 1.0+

**Some key dependences:**  
- [Fast.ai](https://github.com/fastai/fastai#installation)  
- PLY file reader and 3D geometry mesh transforms are implemented by [trimesh](https://github.com/mikedh/trimesh).    
- [ipyvolume](http://ipyvolume.readthedocs.io/) is used for interactive visualisation in jupyter notebooks examples.
- PyTorch sparse convolution models: [SparseConvNet](https://github.com/facebookresearch/SparseConvNet).  

### Install dependences

#### For Ubuntu 18.04

Normal install, go further to the next section "install fastai"

#### For Ubuntu 16.04

Ubuntu 16.04 does not have a python version greater than 3.6 installed by default. You can check the installed python version of the system by executing the following commands:

```bash
python3.6  --version
python3.7  --version
python3  --version
```


If python 3.6 or python 3.7 is not installed in the system, then it can be installed 
1. together with [conda](https://docs.continuum.io/anaconda/install/)


2. or manually using the [askubuntu.com](https://askubuntu.com/questions/865554/how-do-i-install-python-3-6-using-apt-get/865569#865569)

```bash
    sudo add-apt-repository ppa:deadsnakes/ppa
    sudo apt-get update
    sudo apt-get install python3.6

    sudo apt-get install python3.6-dev

    sudo apt-get install python3.6-tk
```

In the latter case, and if the installation is carried out using a separate virtual environment, it should be created by the

```bash
mkvirtualenv -p python3.6 <environmentname>
```


### install fastai

```bash
pip install fastai==1.0.48
```

### Install fastai_sparse

```bash
git clone https://github.com/goodok/fastai_sparse/
cd fastai_sparse
python setup.py develop

cd ../
```


### SparseConvNet


Installing SparseConvNet is only required to run the demo examples that use it. The installation instructions can be found [here](https://github.com/facebookresearch/SparseConvNet#setup).


Breafly, installation of dependencies
```bash
sudo apt-get install libsparsehash-dev
```


Clone the source code, compile and install

```bash
git clone https://github.com/facebookresearch/SparseConvNet
cd SparseConvNet
git checkout cd155261c151c0de34de02608c081b57be67ee5f
sh develop.sh
cd ../
```

##### Note for Ubuntu 16.04

If a virtual environment is used and pyhon 3.6 has been installed manually, the python sources must be specified for successful compilation. To do this, insert into this file
    `~/.virtualenvs/`<environmentname>`/bin/postactivate`
a line:

```
    export PYTHON_INCLUDE_PATH="/usr/include/python3.6m"
```

then reload it:

```bash
deactivate
workon <environmentname>
```




### Tune jupyter extentions and ipyvolume

To visualize 3D objects in notebooks, [ipyvolume](https://github.com/maartenbreddels/ipyvolume)  is used. It allows you to rotate and view them interactively. To make it work, you need to set up extensions 

Details:

- https://github.com/maartenbreddels/ipyvolume#installation
- https://github.com/ipython-contrib/jupyter_contrib_nbextensions

Briefly:


```bash
jupyter contrib nbextension install --sys-prefix

jupyter nbextensions_configurator enable --sys-prefix

jupyter nbextension install --py --sys-prefix widgetsnbextension
jupyter nbextension enable --py --sys-prefix widgetsnbextension

jupyter nbextension install --sys-prefix --py pythreejs
jupyter nbextension enable --sys-prefix --py pythreejs

jupyter nbextension install --sys-prefix --py ipyvolume
jupyter nbextension enable --py --sys-prefix ipyvolume
```

To check, run a laptop and a test example:

Launch `jupyter`
```bash
  jupyter notebook --no-browser
```

Checking a simple example in a file `tests/test_ipyvolume.ipynb`


