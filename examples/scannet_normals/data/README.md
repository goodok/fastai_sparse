# Download ScanNet


Download from:

- http://www.scan-net.org/
- https://github.com/ScanNet/ScanNet


License

The ScanNet data is released under the [ScanNet Terms of Use](http://dovahkiin.stanford.edu/scannet-public/ScanNet_TOS.pdf), and the code is released under the MIT license.


Also you can use this utility:

```bash
git clone http://cvlibs.net:3000/ageiger/rob_devkit.git

cd rob_devkit

python segmentation/download_scannet.py -o 'scannet' --type '_vh_clean_2.ply'

python segmentation/download_scannet.py -o 'scannet' --type '_vh_clean_2.labels.ply'
```


Train / Valid / Test splitting is defined in the directory `ScanNet_Tasks_Benchmark` cloned from 

https://github.com/ScanNet/ScanNet/tree/master/Tasks/
