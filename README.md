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

## Features

- Support of Images, masks and keypoints for all the transforms (including multiple items at the time)
- Fast and PyTorch-integrated
- Convenient and flexible serialization API
- Excellent documentation
- Easy to extend
- 100% Code coverage

## Examples
![Cats](examples/results/cats.png)
![Knee MRI](examples/results/mri.png)

## Installation
The most recent version is available in pip:
```
pip install solt
```
You can fetch the most fresh changes from this repository:
```
pip install git+https://github.com/MIPT-Oulu/solt
```

## Benchmark

We propose a fair benchmark based on the refactored version of the one proposed by albumentations 
team (number of images per second):

|                |albumentations<br><small>0.4.3</small>|torchvision (Pillow-SIMD backend)<br><small>0.5.0</small>|augmentor<br><small>0.2.8</small>|solt<br><small>0.1.9</small>|
|----------------|:------------------------------------:|:-------------------------------------------------------:|:-------------------------------:|:--------------------------:|
|HorizontalFlip  |                 2251                 |                          2622                           |              2582               |         **16544**          |
|VerticalFlip    |                 2455                 |                          2607                           |              2571               |         **25958**          |
|RotateAny       |                 1532                 |                          1432                           |               666               |          **3885**          |
|Crop224         |                 2693                 |                          3091                           |              3006               |         **24998**          |
|Crop128         |                 5613                 |                          5958                           |              5748               |         **24801**          |
|Crop64          |                 9622                 |                          9524                           |              9024               |         **25036**          |
|Crop32          |                12407                 |                          11303                          |              10671              |         **25048**          |
|Pad300          |                 1715                 |                           103                           |                -                |         **16007**          |
|VHFlipRotateCrop|                 1598                 |                          1683                           |               659               |          **1866**          |
|HFlipCrop       |                 2460                 |                          2902                           |              2862               |          **3514**          |

Python and library versions: Python 3.7.0 (default, Oct  9 2018, 10:31:47) [GCC 7.3.0], numpy 1.18.1, pillow-simd 7.0.0.post3, opencv-python 4.2.0.32, scikit-image 0.16.2, scipy 1.4.1.
Please find the details about the benchmark [here](BENCHMARK.md).

## Papers that use SOLT
The aim of building SOLT was to create a tool for reproducible research. At MIPT, we use SOLT in our projects:

1. https://arxiv.org/abs/1907.05089
2. https://arxiv.org/abs/1904.06236
3. https://arxiv.org/abs/1907.08020
4. https://arxiv.org/abs/1907.12237

If you use SOLT and cite it in your research, please, don't hesitate to sent an email to Aleksei Tiulpin. It will be added here.

## How to contribute
Follow the guidelines described [here](CONTRIBUTING.md). 

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
