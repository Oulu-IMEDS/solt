#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import find_packages, setup


def parse_requirements_file(filename):
    with open(filename) as f:
        requires = [line.strip() for line in f.readlines() if line]
    return requires


requirements_base = parse_requirements_file("requirements_base.txt")
requirements_test = parse_requirements_file("requirements_test.txt")

description = (
    "Data augmentation library for Deep Learning, which supports images, segmentation "
    "masks, labels and keypoints. Furthermore, SOLT is fast and has OpenCV in its "
    "backend. Full auto-generated docs and examples are available here: "
    "https://mipt-oulu.github.io/solt/."
)

setup(
    author="Aleksei Tiulpin",
    author_email="aleksei.tiulpin@oulu.fi",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
    ],
    description="Optimized data augmentation library for Deep Learning",
    install_requires=requirements_base,
    license="MIT license",
    long_description=description,
    include_package_data=True,
    keywords="data augmentations, deeep learning",
    name="solt",
    packages=find_packages(include=["solt"]),
    test_suite="tests",
    tests_require=requirements_test,
    url="https://github.com/mipt-oulu/solt",
    version="0.1.9",
    zip_safe=False,
)
