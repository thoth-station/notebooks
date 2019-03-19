#!/bin/bash

set -e

CREDENTIALS="\
THOTH_DEPLOYMENT_NAME\tcustom deployment name\t[optional], defaults to \$USER\n
THOTH_CEPH_HOST\t Ceph s3 host url\n
THOTH_CEPH_BUCKET\t Ceph s3 bucket name\n
THOTH_CEPH_BUCKET_PREFIX\t Ceph s3 bucket prefix\t\n
JANUSGRAPH_SERVICE_HOST\t Janusgraph host url\t\n
JANUSGRAPH_SERVICE_PORT\t Janusgraph host port\t [optional], defaults to 8182\n
"

function die() {
    echo "$*" 1>&2
    exit 1
}

display_usage() {
  echo
  echo "Usage: $0"
  echo
  echo -e " -h, --help     Display this help and exit"
  echo -e " -c, --cluster  Use cluster components. This requires proper credential setup:"
  echo
  echo -e "Credentials:\n"
  # echo -e " $(echo -e $CREDENTIALS | awk -F ',' '{printf "%-30s%-30s%-20s\n",$1,$2,$3}')"
  echo -e " $(echo -e $CREDENTIALS | column -t -s $'\t')"
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
		/usr/bin/env bash ./install.sh
	else
		/usr/bin/env bash ./install.sh > /dev/null
	fi
fi


if [ "${LOCAL}" = "false" ]; then
	echo -e "\033[33;1mINFO: Attempt to run with cluster components.\033[0m"
	eval $(gopass show aicoe/thoth/ceph.sh)

	# These must be set by environment
	echo -e "\033[33;1mINFO: Checking required environment variables.\033[0m"
	[ -z "$THOTH_CEPH_HOST" ]          && die "ERROR: THOTH_CEPH_HOST must be set"
	[ -z "$THOTH_CEPH_BUCKET" ]        && die "ERROR: THOTH_CEPH_BUCKET must be set"
	[ -z "$THOTH_CEPH_BUCKET_PREFIX" ] && die "ERROR: THOTH_CEPH_BUCKET_PREFIX must be set"
	[ -z "$JANUSGRAPH_SERVICE_HOST" ]  && die "ERROR: JANUSGRAPH_SERVICE_HOST must be set"

	# These are optional
	echo -e "\033[33;1mINFO: Checking optional environment variables.\033[0m"
	[ -z "$THOTH_DEPLOYMENT_NAME" ]    && echo "WARNING: THOTH_DEPLOYMENT_NAME was not set: using $THOTH_DEPLOYMENT_NAME"
	[ -z "$JANUSGRAPH_SERVICE_PORT" ]  && echo "WARNING: JANUSGRAPH_SERVICE_PORT was not set: using $JANUSGRAPH_SERVICE_PORT"

	ulimit -Sn 4096
fi

pipenv run jupyter notebook --notebook-dir=notebooks --NotebookApp.token=''
