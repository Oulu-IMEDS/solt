#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages


requirements = [ ]

setup_requirements = [ ]

test_requirements = ['pytest', ]

setup(
    author="Aleksei Tiulpin",
    author_email='aleksei.tiulpin@protonmail.ch',
    classifiers=[
        'Development Status :: 2 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
    description="JIT-optimized data augmentation library for machine learning",
    install_requires=requirements,
    license="MIT license",
    long_description='',
    include_package_data=True,
    keywords='data augmentations, deeep learning',
    name='fastaug',
    packages=find_packages(include=['fastaug']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/lext/fastaug',
    version='0.1.0',
    zip_safe=False,
)