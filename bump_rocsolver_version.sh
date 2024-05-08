#!/bin/sh

# run this script in develop after creating release-staging branch for feature-complete date
# Edit script to bump versions for new development cycle/release.

# for rocSOLVER version string
OLD_ROCSOLVER_VERSION="3\.27\.0"
NEW_ROCSOLVER_VERSION="3.28.0"
sed -i "s/${OLD_ROCSOLVER_VERSION}/${NEW_ROCSOLVER_VERSION}/g" CMakeLists.txt

# for rocSOLVER library name
OLD_ROCSOLVER_SOVERSION="0\.3"
NEW_ROCSOLVER_SOVERSION="0.4"
sed -i "s/${OLD_ROCSOLVER_SOVERSION}/${NEW_ROCSOLVER_SOVERSION}/g" library/CMakeLists.txt
