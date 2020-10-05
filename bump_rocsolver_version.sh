#!/bin/sh

# This script needs to be edited to bump rocsolver version for new release.
# - run this script in develop before merging develop into staging

OLD_ROCSOLVER_VERSION="3.9.0"   #Released in September/October 2020
NEW_ROCSOLVER_VERSION="3.10.0"  #To release in October/November 2020
sed -i "s/${OLD_ROCSOLVER_VERSION}/${NEW_ROCSOLVER_VERSION}/g" CMakeLists.txt

# for documentation

OLD_ROCSOLVER_VERSION="3.9"
NEW_ROCSOLVER_VERSION="3.10"
sed -i "s/${OLD_ROCSOLVER_VERSION}/${NEW_ROCSOLVER_VERSION}/g" docs/source/conf.py
