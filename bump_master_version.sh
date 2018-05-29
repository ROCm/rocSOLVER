#!/bin/sh

# This script needs to be edited to bump old develop version to new master version for new release.
# - run this script in develop branch 
# - after running this script merge develop into master
# - after running this script and merging develop into master, run bump_develop_version.sh in master and
#   merge master into develop

OLD_ROCSOLVER_VERSION="0.1.1"
NEW_ROCSOLVER_VERSION="0.2.0"

sed -i "s/${OLD_ROCSOLVER_VERSION}/${NEW_ROCSOLVER_VERSION}/g" CMakeLists.txt
