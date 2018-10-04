#!/bin/bash

if [ ! -z "$VAR" ]
then
	eval $(gopass show aicoe/thoth/ceph.sh)
	export THOTH_DEPLOYMENT_NAME=''
       	export THOTH_CEPH_BUCKET=''
	export THOTH_CEPH_HOST=''
	export THOTH_CEPH_BUCKET_PREFIX=''
	export THOTH_JANUSGRAPH_HOST=''
	export THOTH_JANUSGRAPH_PORT=''
	pipenv run jupyter notebook --notebook-dir=notebooks
else
	echo "You do not have access. Please check your accesibility rights"
fi
