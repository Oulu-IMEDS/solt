#!/bin/sh

if [ "`uname`" == "Darwin" ];
then
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O $HOME/miniconda.sh;
fi

if [ "`uname`" == "Linux" ];
then
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $HOME/miniconda.sh;
fi