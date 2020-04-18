#!/bin/bash

if [ -d docBin ]; then
    rm -rf docBin
fi

sed -e 's/ROCSOLVER_EXPORT //g' ../library/include/rocsolver.h > rocsolver.h
sed -e 's/ROCSOLVER_EXPORT //g' ../library/include/rocsolver-functions.h > rocsolver-functions.h
sed -e 's/ROCSOLVER_EXPORT //g' ../library/include/rocsolver-types.h > rocsolver-types.h
sed -e 's/ROCSOLVER_EXPORT //g' -e 's/__inline //g' ../library/include/rocsolver-auxiliary.h > rocsolver-auxiliary.h

doxygen Doxyfile
rm *.h
