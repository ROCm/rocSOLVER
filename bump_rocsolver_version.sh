#!/bin/sh

# This script needs to be edited to bump rocsolver version for new release.
# - run this script in develop before merging develop into staging

OLD_ROCSOLVER_VERSION="3.7.0"   #Released in July/August 2020
NEW_ROCSOLVER_VERSION="3.8.0"   #To release in August/September 2020

sed -i "s/${OLD_ROCSOLVER_VERSION}/${NEW_ROCSOLVER_VERSION}/g" CMakeLists.txt

