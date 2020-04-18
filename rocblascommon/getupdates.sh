#!/bin/bash

blasdir=$1
filelist="cmake/get-cli-arguments.cmake \
          cmake/os-detection.cmake \
          cmake/package-functions.cmake \
          deps/CMakeLists.txt \
          deps/external-boost.cmake \
          deps/external-gtest.cmake \
          deps/external-lapack.cmake \
          library/src/include/definitions.h \
          library/src/include/utility.h"

for file in ${filelist}; do
    echo $file
    rsync ${blasdir}/$file ./$file
done




