#!/bin/bash

set -eu

# Make this directory the PWD
cd "$(dirname "${BASH_SOURCE[0]}")"

# Build doxygen info
bash run_doxygen.sh

# Build sphinx docs
mkdir -p ../build/docs/sphinx
cd source
make clean
make html
make latexpdf
