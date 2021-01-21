#!/bin/sh

# This script needs to be edited to bump rocsolver version for new release.
# - run this script in develop before merging developi/staging into master

OLD_ROCSOLVER_VERSION="3.10.0"  #Released in October/November 2020
NEW_ROCSOLVER_VERSION="3.11.0"  #To release in February 2021
sed -i "s/${OLD_ROCSOLVER_VERSION}/${NEW_ROCSOLVER_VERSION}/g" CMakeLists.txt

# for documentation

OLD_ROCSOLVER_VERSION="3.10"
NEW_ROCSOLVER_VERSION="3.11"
sed -i "s/${OLD_ROCSOLVER_VERSION}/${NEW_ROCSOLVER_VERSION}/g" docs/source/conf.py
