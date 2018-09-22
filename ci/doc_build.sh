#!/bin/sh

if [ $TRAVIS_PYTHON_VERSION = 3.7 ] && [ $TRAVIS_OS_NAME = "linux" ];
then
    travis-sphinx build --source=./doc/source/
    travis-sphinx deploy
fi