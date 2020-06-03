#!/bin/sh

# This script needs to be edited to bump rocsolver version for new release.
# - run this script in develop before merging develop into staging

OLD_ROCSOLVER_VERSION="3.5.0"   #Released in May/June 2020
NEW_ROCSOLVER_VERSION="3.6.0"   #To release in June/July 2020

sed -i "s/${OLD_ROCSOLVER_VERSION}/${NEW_ROCSOLVER_VERSION}/g" CMakeLists.txt

