![slide](doc/source/_static/logo.png)
--------------------------------------------------------------------------------
[![PyPI version](https://badge.fury.io/py/solt.svg)](https://badge.fury.io/py/solt)
[![Build Status](https://travis-ci.org/MIPT-Oulu/solt.svg?branch=master)](https://travis-ci.org/MIPT-Oulu/solt)
[![Codecoverage](https://codecov.io/gh/MIPT-Oulu/solt/branch/master/graph/badge.svg)](https://codecov.io/gh/MIPT-Oulu/solt)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/80bb13f72fe645b29ded3d6cabaacf15)](https://www.codacy.com/app/lext/solt?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=MIPT-Oulu/solt&amp;utm_campaign=Badge_Grade)
[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE.md)
[![DOI](https://zenodo.org/badge/143310844.svg)](https://zenodo.org/badge/latestdoi/143310844)

## Description
Data augmentation libarary for Deep Learning, which supports images, segmentation masks, labels and keypoints. 
Furthermore, SOLT is fast and has OpenCV in its backend. 
Full auto-generated docs and 
examples are available here: [https://mipt-oulu.github.io/solt/](https://mipt-oulu.github.io/solt/).

## Installation
The most recent version is available in pip:
```
pip install solt
```
You can fetch the most fresh changes from this repository:
```
pip install git+https://github.com/MIPT-Oulu/solt
```
## Papers that use SOLT
The aim of building SOLT was to create a tool for reproducible research. At MIPT, we use SOLT in our projects:

1. https://arxiv.org/abs/1907.05089
2. https://arxiv.org/abs/1904.06236
3. https://arxiv.org/abs/1907.08020
4. https://arxiv.org/abs/1907.12237

If you use SOLT and cite it in your research, please, don't hesitate to sent an email to Aleksei Tiulpin. It will be added here.

## Benchmark
We have conducted a fair benchmark of several augmentation libraries by 
comparing how many images they process per second. In this benchmark, we measured
the transform itself, as well as the conversion to torch.Tensor and also
a subtraction of the ImageNet mean. 

Here is how you can run the benchmark yourself:

```
export DATA_DIR="<PATH to ImageNet val>"
conda env create -f benchmark/augbench.yaml
conda activate augbench
pip install git+https://github.com/MIPT-Oulu/solt@master#egg-name=solt
pip install -e benchmark
python -u -m augbench.benchmark -i 500
```

Benchmark results:


## Author
Aleksei Tiulpin, 
Research Unit of Medical Imaging, 
Physics and Technology, 
University of Oulu, Finalnd.

## How to cite
```
@misc{aleksei_tiulpin_2019_3351977,
  author       = {Aleksei Tiulpin},
  title        = {SOLT: Streaming over Lightweight Transformations},
  month        = jul,
  year         = 2019,
  doi          = {10.5281/zenodo.3351977},
  url          = {https://doi.org/10.5281/zenodo.3351977},
}
```
