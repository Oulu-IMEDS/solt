#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages


requirements = ('numpy', 'opencv-python')

setup_requirements = ()

test_requirements = ('pytest',)

setup(
    author="Aleksei Tiulpin",
    author_email='aleksei.tiulpin@oulu.fi',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
    ],
    description="Optimized data augmentation library for Deep Learning",
    install_requires=requirements,
    license="MIT license",
    long_description='',
    include_package_data=True,
    keywords='data augmentations, deeep learning',
    name='solt',
    packages=find_packages(include=['solt']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/mipt-oulu/solt',
    version='0.0.1',
    zip_safe=False,
)