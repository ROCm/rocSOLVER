#!/bin/sh

# run this script in develop after merging develop/staging into master at the feature-complete date
# Edit script to bump versions for new development cycle/release.

OLD_ROCSOLVER_VERSION="3.16.0"  # Distributed with ROCm 5.0
NEW_ROCSOLVER_VERSION="3.17.0"  # Distributed with ROCm 5.1
sed -i "s/${OLD_ROCSOLVER_VERSION}/${NEW_ROCSOLVER_VERSION}/g" CMakeLists.txt

# for documentation
OLD_ROCSOLVER_DOCS_VERSION="3.16"
NEW_ROCSOLVER_DOCS_VERSION="3.17"
sed -i "s/${OLD_ROCSOLVER_DOCS_VERSION}/${NEW_ROCSOLVER_DOCS_VERSION}/g" docs/source/conf.py

# for rocBLAS package requirements
OLD_ROCBLAS_VERSION_DOWN="2.42"
NEW_ROCBLAS_VERSION_DOWN="2.43"
OLD_ROCBLAS_VERSION_UP="2.43"
NEW_ROCBLAS_VERSION_UP="2.44"
sed -i "s/${OLD_ROCBLAS_VERSION_UP}/${NEW_ROCBLAS_VERSION_UP}/g" library/CMakeLists.txt
sed -i "s/${OLD_ROCBLAS_VERSION_DOWN}/${NEW_ROCBLAS_VERSION_DOWN}/g" library/CMakeLists.txt

