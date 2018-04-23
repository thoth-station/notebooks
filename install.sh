#!/bin/sh

set -ex

[ -d thoth-jupyter-notebook ] || git clone https://github.com/thoth-station/jupyter-notebook
pushd thoth-jupyter-notebook
git pull
popd
# numpy has to be installed first otherwise jupyter installation will fail
pipenv install numpy
pipenv install jupyter
pushd thoth-jupyter-notebook
git pull
popd
pipenv install -r thoth-jupyter-notebook/requirements.txt
