#!/bin/bash

if [ $TRAVIS_OS_NAME = "osx" ];
then
    echo "Getting Conda for OSX...\n"
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O $HOME/miniconda.sh;
fi

if [ $TRAVIS_OS_NAME = "linux" ];
then
    echo "Getting Conda for Linux...\n"
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $HOME/miniconda.sh;
fi