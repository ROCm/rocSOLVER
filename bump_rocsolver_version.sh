#!/bin/sh

# This script needs to be edited to bump rocsolver version for new release.
# - run this script in develop after merging develop/staging into master

OLD_ROCSOLVER_VERSION="3.13.0"  #To release in April/May 2021
NEW_ROCSOLVER_VERSION="3.14.0"
sed -i "s/${OLD_ROCSOLVER_VERSION}/${NEW_ROCSOLVER_VERSION}/g" CMakeLists.txt

# for documentation
OLD_ROCSOLVER_DOCS_VERSION="3.13"
NEW_ROCSOLVER_DOCS_VERSION="3.14"
sed -i "s/${OLD_ROCSOLVER_DOCS_VERSION}/${NEW_ROCSOLVER_DOCS_VERSION}/g" docs/source/conf.py

# for rocBLAS package requirements
OLD_ROCBLAS_VERSION_DOWN="2.39"
NEW_ROCBLAS_VERSION_DOWN="2.41"
OLD_ROCBLAS_VERSION_UP="2.40"
NEW_ROCBLAS_VERSION_UP="2.42"
sed -i "s/${OLD_ROCBLAS_VERSION_UP}/${NEW_ROCBLAS_VERSION_UP}/g" library/CMakeLists.txt
sed -i "s/${OLD_ROCBLAS_VERSION_DOWN}/${NEW_ROCBLAS_VERSION_DOWN}/g" library/CMakeLists.txt

