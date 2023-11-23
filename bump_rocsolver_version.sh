#!/bin/sh

# run this script in develop after creating release-staging branch for feature-complete date
# Edit script to bump versions for new development cycle/release.

# for rocSOLVER version string
OLD_ROCSOLVER_VERSION="3.26.0"
NEW_ROCSOLVER_VERSION="3.27.0"
sed -i "s/${OLD_ROCSOLVER_VERSION}/${NEW_ROCSOLVER_VERSION}/g" CMakeLists.txt

# for rocBLAS package requirements
OLD_ROCBLAS_VERSION="4.2"
NEW_ROCBLAS_VERSION="4.3"
sed -i "s/${OLD_ROCBLAS_VERSION}/${NEW_ROCBLAS_VERSION}/g" CMakeLists.txt

