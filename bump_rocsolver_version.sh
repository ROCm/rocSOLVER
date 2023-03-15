#!/bin/sh

# run this script in develop after merging develop/staging into master at the feature-complete date
# Edit script to bump versions for new development cycle/release.

OLD_ROCSOLVER_VERSION="3.22.0"
NEW_ROCSOLVER_VERSION="3.23.0"
sed -i "s/${OLD_ROCSOLVER_VERSION}/${NEW_ROCSOLVER_VERSION}/g" CMakeLists.txt

# for documentation
OLD_ROCSOLVER_DOCS_VERSION="3.22"
NEW_ROCSOLVER_DOCS_VERSION="3.23"
sed -i "s/${OLD_ROCSOLVER_DOCS_VERSION}/${NEW_ROCSOLVER_DOCS_VERSION}/g" docs/source/conf.py

# for rocBLAS package requirements
OLD_ROCBLAS_VERSION_DOWN="3.0"
NEW_ROCBLAS_VERSION_DOWN="3.1"
OLD_ROCBLAS_VERSION_UP="3.1"
NEW_ROCBLAS_VERSION_UP="3.2"
sed -i "s/${OLD_ROCBLAS_VERSION_UP}/${NEW_ROCBLAS_VERSION_UP}/g" CMakeLists.txt
sed -i "s/${OLD_ROCBLAS_VERSION_DOWN}/${NEW_ROCBLAS_VERSION_DOWN}/g" CMakeLists.txt

