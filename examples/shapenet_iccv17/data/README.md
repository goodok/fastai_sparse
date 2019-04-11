# Download compitition dataset Shapenet iccv17 segmentation

https://shapenet.cs.stanford.edu/iccv17/

1. Download zip files:

```bash
    sh download_iccv17_pl_seg.sh
```

2. Unpack and re-split the Train/Validation data 50-50 to increase the size of the validation set.

```bash
     sh unpack.sh
```


3. Convert .pts to numpy .npy

Run notebook `convert_to_npy.ipynb` or

```bash
     python convert_to_npy.py
```

