#!/bin/bash

set -eu

# Make this directory the PWD
cd "$(dirname "${BASH_SOURCE[0]}")"

mkdir -p ../build/docs
rm -rf ../build/docs/doxygen
doxygen Doxyfile
