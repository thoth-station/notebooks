Jupyter notebooks for Thoth
---------------------------

A set of Jupyter notebooks for proejct Thoth.

Installation and running
========================

.. code-block:: console

  $ git clone git@github.com:thoth-station/notebooks.git
  $ cd notebooks
  $ ./install.sh
  ...
  $ ./run.sh

The host system has to have Python3 and pipenv installed.

If you would like to access resources on the cluster, adjust the `run.sh` script and provide requested environment variables. Do NOT commit credentials!
