#!/bin/sh

# If running locally, these variables need to be provided in order to make notebooks work.
#
#    !!! DO NOT COMMIT CHANGES !!!
export THOTH_DEPLOYMENT_NAME=
export THOTH_CEPH_BUCKET=
export THOTH_CEPH_KEY_ID=
export THOTH_CEPH_SECRET_KEY=
export THOTH_CEPH_HOST=
#    !!! DO NOT COMMIT CHANGES !!!

pipenv run jupyter notebook --notebook-dir=notebooks
