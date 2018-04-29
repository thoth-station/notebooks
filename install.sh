#!/bin/sh

set -ex

[ -d jupyter-notebook ] || git clone https://github.com/thoth-station/jupyter-notebook
pushd jupyter-notebook
git pull
popd
# numpy has to be installed first otherwise jupyter installation will fail
pipenv install numpy
pipenv install jupyter
pushd jupyter-notebook
git pull
popd
pipenv install -r jupyter-notebook/requirements.txt
