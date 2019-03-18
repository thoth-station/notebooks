#!/bin/bash

set -e

CREDENTIALS="\
	THOTH_DEPLOYMENT_NAME		custom deployment name	[optional], defaults to \$USER
	THOTH_CEPH_HOST				Ceph s3 host url
	THOTH_CEPH_BUCKET			Ceph s3 bucket name
	THOTH_CEPH_BUCKET_PREFIX	Ceph s3 bucket prefix
	JANUSGRAPH_SERVICE_HOST		Janusgraph host url
	JANUSGRAPH_SERVICE_PORT		Janusgraph host port    [optional], defaults to 8182
"

display_usage() {
  echo
  echo "Usage: $0"
  echo
  echo " -h, --help     Display this help and exit"
  echo " -c, --cluster  Use cluster components. This requires proper credential setup: \n $CREDENTIALS"
  echo
}


LOCAL=true

INSTALL=false
VERBOSE=false

if [[ -z $1 ]] ; then
	echo -e "\033[33;1mINFO: Running with local setup.\033[0m"
else
	while (("$#")); do
		case $1 in
			-h|--help)
				display_usage
				exit 0
				;;
			-i|--install)
				INSTALL=true
				shift
				;;
			-c|--cluster)
				LOCAL=false;
				shift
				;;
			-l|--local)
				LOCAL=true;
				shift
				;;
			-v|--verbose)
				VERBOSE=true;
				shift
				;;
			-*|--*=)
				echo -e "\033[33;1mERROR: Unsupported flaggs.\033[0m"
				display_usage
				exit 1
				;;
			*)
				shift
				;;
		esac
	done
fi

[ "${VERBOSE}" = "true" ] && set -x

echo -e "\033[33;1mThoth sends his regards.\033[0m"

THOTH_DEPLOYMENT_NAME=${THOTH_DEPLOYMENT_NAME:="$USER"}
THOTH_CEPH_HOST=${THOTH_CEPH_HOST:=""}
THOTH_CEPH_BUCKET=${THOTH_CEPH_BUCKET:=""}
THOTH_CEPH_BUCKET_PREFIX=${THOTH_CEPH_BUCKET_PREFIX:=""}
JANUSGRAPH_SERVICE_HOST=${JANUSGRAPH_SERVICE_HOST:=""}
JANUSGRAPH_SERVICE_PORT=${JANUSGRAPH_SERVICE_PORT:="8182"}


if [ "${INSTALL}" = "true" ]; then
	echo -e "\033[33;1mINFO: Installing...\033[0m"
	echo -e "\033[33;1mNOTE: Grab a coffee, this will take a while.\033[0m"
	if [ "${VERBOSE}" = "true" ]; then
		/usr/bin/env bash ./install.sh || exit $?
	else
		/usr/bin/env bash ./install.sh > /dev/null || exit ${PIPESTATUS[0]}
	fi
fi


if [ "${LOCAL}" = "false" ]; then
	echo -e "\033[33;1mINFO: Attempt to run with cluster components.\033[0m"
	eval $(gopass show aicoe/thoth/ceph.sh)

	# These must be set by environment
	echo -e "\033[33;1mINFO: Checking required environment variables.\033[0m"
	[ -z "$THOTH_CEPH_HOST" ]          && echo "ERROR: THOTH_CEPH_HOST must be set"  		 && exit 1
	[ -z "$THOTH_CEPH_BUCKET" ]		   && echo "ERROR: THOTH_CEPH_BUCKET must be set" 	  	 && exit 1
	[ -z "$THOTH_CEPH_BUCKET_PREFIX" ] && echo "ERROR: THOTH_CEPH_BUCKET_PREFIX must be set" && exit 1
	[ -z "$JANUSGRAPH_SERVICE_HOST" ]  && echo "ERROR: JANUSGRAPH_SERVICE_HOST must be set"  && exit 1

	# These are optional
	echo -e "\033[33;1mINFO: Checking optional environment variables.\033[0m"
	[ -z "$THOTH_DEPLOYMENT_NAME" ]    && echo "WARNING: THOTH_DEPLOYMENT_NAME was not set: using $THOTH_DEPLOYMENT_NAME"
	[ -z "$JANUSGRAPH_SERVICE_PORT" ]  && echo "WARNING: JANUSGRAPH_SERVICE_PORT was not set: using $JANUSGRAPH_SERVICE_PORT"

	ulimit -Sn 4096
fi

pipenv run jupyter notebook --notebook-dir=notebooks --NotebookApp.token=''
