#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages


requirements = ('numpy', 'opencv-python')

setup_requirements = ()

test_requirements = ('pytest',)

description = """Data augmentation libarary for Deep Learning, which supports images, segmentation masks, labels and keypoints. 
Furthermore, SOLT is fast and has OpenCV in its backend. 
Full auto-generated docs and 
examples are available here: [https://mipt-oulu.github.io/solt/](https://mipt-oulu.github.io/solt/).

"""

setup(
    author="Aleksei Tiulpin",
    author_email='aleksei.tiulpin@oulu.fi',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="Optimized data augmentation library for Deep Learning",
    install_requires=requirements,
    license="MIT license",
    long_description=description,
    include_package_data=True,
    keywords='data augmentations, deeep learning',
    name='solt',
    packages=find_packages(include=['solt']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/mipt-oulu/solt',
    version='0.0.5',
    zip_safe=False,
)
