# rocSOLVER

rocSOLVER is a work-in-progress implementation of a subset of LAPACK functionality on the ROCm platform. It uses rocBLAS as a companion GPU BLAS implementation.

# Build
Requires `cmake` and `ROCm` including `hcc` and `rocBLAS` to be installed.

```bash
mkdir build && cd build
CXX=/opt/rocm/bin/hcc cmake ..
make
```
# Implemented functions in LAPACK notation
Cholesky decomposition: `rocsolver_spotf2() rocsolver_dpotf2()`
unblocked LU decomposition: `rocsolver_sgetf2() rocsolver_dgetf2()`
blocked LU decomposition: `rocsolver_sgetrf() rocsolver_dgetrf()`
