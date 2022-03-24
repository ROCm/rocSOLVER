#!/bin/sh

# run this script in develop after merging develop/staging into master at the feature-complete date
# Edit script to bump versions for new development cycle/release.

OLD_ROCSOLVER_VERSION="3.19.0"
NEW_ROCSOLVER_VERSION="3.20.0"
sed -i "s/${OLD_ROCSOLVER_VERSION}/${NEW_ROCSOLVER_VERSION}/g" CMakeLists.txt

# for documentation
OLD_ROCSOLVER_DOCS_VERSION="3.19"
NEW_ROCSOLVER_DOCS_VERSION="3.20"
sed -i "s/${OLD_ROCSOLVER_DOCS_VERSION}/${NEW_ROCSOLVER_DOCS_VERSION}/g" docs/source/conf.py

# for rocBLAS package requirements
OLD_ROCBLAS_VERSION_DOWN="2.45"
NEW_ROCBLAS_VERSION_DOWN="2.46"
OLD_ROCBLAS_VERSION_UP="2.46"
NEW_ROCBLAS_VERSION_UP="2.47"
sed -i "s/${OLD_ROCBLAS_VERSION_UP}/${NEW_ROCBLAS_VERSION_UP}/g" CMakeLists.txt
sed -i "s/${OLD_ROCBLAS_VERSION_DOWN}/${NEW_ROCBLAS_VERSION_DOWN}/g" CMakeLists.txt

