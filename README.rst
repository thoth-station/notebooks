Jupyter notebooks for Thoth
---------------------------

A set of Jupyter notebooks for proejct Thoth.

Installation and running
========================

.. code-block:: console

  $ git clone git@github.com:thoth-station/notebooks.git
  $ cd notebooks
  $ ./install.sh  # coffee break suggested
  ...
  $ ./run.sh

Alternatively we can run `./run.sh --install` which runs the `./install.sh` command. 

Run `run.sh --help` for more information


NOTES
=====

The host system has to have Python3 and pipenv installed.

For local run, there are used Thoth's libraries as cloned on the local machine.

If you would like to access resources on the cluster, provide the `run.sh` script required environment variables.


THOTH NOTEBOOKS TEMPLATE
==============================

Project Thoth uses template notebook to repeat analysis automatically.

Thoth notebooks templates are stored in `templates <https://github.com/thoth-station/notebooks/tree/master/notebooks/templates>`_,
check the README files in that folder for more details.
