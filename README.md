# Neural radiance fields-based multi-view endoscopic scene reconstruction for surgical simulation
We propose an Endoscope-NeRF network for implicit radiance fields reconstruction of endoscopic scene under non-fixed light source, and synthesize novel views

## Installation
Clone this repo with submodules:
```
git clone --recurse-submodules https://github.com/qinzhibao123/Endoscope-NeRF
cd Endoscope-NeRF/
```

The code is tested with Python3.7, PyTorch == 1.5 and CUDA == 10.2. We recommend you to use [anaconda](https://www.anaconda.com/) to make sure that all dependencies are in place. To create an anaconda environment:
```
conda env create -f environment.yml
conda activate ibrnet
```
