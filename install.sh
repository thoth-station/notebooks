#!/bin/sh

set -ex

[ -d jupyter-notebook ] || git clone https://github.com/thoth-station/jupyter-notebook
pushd jupyter-notebook
git pull
popd
# We explicitly set version of Python to 3.6.
pipenv install --python 3.6
# numpy has to be installed first otherwise jupyter installation will fail
pipenv install numpy
pipenv install jupyter
pushd jupyter-notebook
git pull
popd
pipenv install -r jupyter-notebook/requirements.txt
