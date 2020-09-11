#!/bin/bash

set -eu

# Make this directory the PWD
cd "$(dirname "${BASH_SOURCE[0]}")"

bash run_doxygen.sh

cd source
make clean
make html
make latexpdf
