#!/bin/sh

# run this script in develop after merging develop/staging into master at the feature-complete date
# Edit script to bump versions for new development cycle/release.

OLD_ROCSOLVER_VERSION="3.17.0"  
NEW_ROCSOLVER_VERSION="3.18.0"  
sed -i "s/${OLD_ROCSOLVER_VERSION}/${NEW_ROCSOLVER_VERSION}/g" CMakeLists.txt

# for documentation
OLD_ROCSOLVER_DOCS_VERSION="3.17"
NEW_ROCSOLVER_DOCS_VERSION="3.18"
sed -i "s/${OLD_ROCSOLVER_DOCS_VERSION}/${NEW_ROCSOLVER_DOCS_VERSION}/g" docs/source/conf.py

# for rocBLAS package requirements
OLD_ROCBLAS_VERSION_DOWN="2.43"
NEW_ROCBLAS_VERSION_DOWN="2.44"
OLD_ROCBLAS_VERSION_UP="2.44"
NEW_ROCBLAS_VERSION_UP="2.45"
sed -i "s/${OLD_ROCBLAS_VERSION_UP}/${NEW_ROCBLAS_VERSION_UP}/g" library/CMakeLists.txt
sed -i "s/${OLD_ROCBLAS_VERSION_DOWN}/${NEW_ROCBLAS_VERSION_DOWN}/g" library/CMakeLists.txt

