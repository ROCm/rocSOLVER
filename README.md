# rocSOLVER

rocSOLVER is a work-in-progress implementation of a subset of LAPACK functionality on the ROCm platform. It requires rocBLAS as a companion GPU BLAS implementation.

# Build
Requires `cmake` and `ROCm` including `hcc` and `rocBLAS` to be installed.

```bash
mkdir build && cd build
CXX=/opt/rocm/bin/hcc cmake ..
make
```
# Implemented functions in LAPACK notation
<pre>
Single and double precision:

Cholesky decomposition:                 rocsolver_potf2() 
unblocked LU decomposition:             rocsolver_getf2() 
                                        rocsolver_getf2_batched()
                                        rocsolver_getf2_strided_batched()
blocked LU decomposition:               rocsolver_getrf()
                                        rocsolver_getrf_batched()
                                        rocsolver_getrf_strided_batched()
unblocked QR decomposition:             rocsolver_geqr2()
                                        rocsolver_geqr2_batched()
                                        rocsolver_geqr2_strided_batched()
solution of system of linear equations: rocsolver_sgetrs() rocsolver_dgetrs()  
</pre>
