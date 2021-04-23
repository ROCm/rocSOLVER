#!/bin/sh

# run this script in develop after merging develop/staging into master at the feature-complete date
# Edit script to bump versions for new development cycle/release.

OLD_ROCSOLVER_VERSION="3.14.0"  # Distributed with ROCm 4.4
NEW_ROCSOLVER_VERSION="3.15.0"  # Distributed with ROCm 4.5
sed -i "s/${OLD_ROCSOLVER_VERSION}/${NEW_ROCSOLVER_VERSION}/g" CMakeLists.txt

# for documentation
OLD_ROCSOLVER_DOCS_VERSION="3.14"
NEW_ROCSOLVER_DOCS_VERSION="3.15"
sed -i "s/${OLD_ROCSOLVER_DOCS_VERSION}/${NEW_ROCSOLVER_DOCS_VERSION}/g" docs/source/conf.py

# for rocBLAS package requirements
OLD_ROCBLAS_VERSION_DOWN="2.40"
NEW_ROCBLAS_VERSION_DOWN="2.41"
OLD_ROCBLAS_VERSION_UP="2.41"
NEW_ROCBLAS_VERSION_UP="2.42"
sed -i "s/${OLD_ROCBLAS_VERSION_UP}/${NEW_ROCBLAS_VERSION_UP}/g" library/CMakeLists.txt
sed -i "s/${OLD_ROCBLAS_VERSION_DOWN}/${NEW_ROCBLAS_VERSION_DOWN}/g" library/CMakeLists.txt

