#!/bin/bash

set -e
set -x

if [ ! -z "$VAR" ]
then
	eval $(gopass show aicoe/thoth/ceph.sh)
	export THOTH_DEPLOYMENT_NAME=''
       	export THOTH_CEPH_BUCKET=''
	export THOTH_CEPH_HOST=''
	export THOTH_CEPH_BUCKET_PREFIX=''
	export JANUSGRAPH_SERVICE_HOST=''
	export JANUSGRAPH_SERVICE_PORT=''
	ulimit -Sn 4096
	pipenv run jupyter notebook --notebook-dir=notebooks --NotebookApp.token=''
else
	echo "You do not have access. Please check your accesibility rights"
fi
