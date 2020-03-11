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

| Lapack Auxiliary Function | single | double | single complex | double complex |
| ------------------------- | ------ | ------ | -------------- | -------------- |
|**rocsolver_laswp**        |     x  |    x   |      x         |   x            |
|**rocsolver_larfg**        |     x  |    x   |                |                |
|**rocsolver_larft**        |     x  |    x   |                |                |
|**rocsolver_larf**         |     x  |    x   |                |                |
|**rocsolver_larfb**        |     x  |    x   |                |                |
|**rocsolver_org2r**        |     x  |    x   |                |                |
|**rocsolver_orgqr**        |     x  |    x   |                |                |
|**rocsolver_orgl2**        |     x  |    x   |                |                |
|**rocsolver_orglq**        |     x  |    x   |                |                |
|**rocsolver_orgbr**        |     x  |    x   |                |                |
|**rocsolver_orm2r**        |     x  |    x   |                |                |
|**rocsolver_ormqr**        |     x  |    x   |                |                |

| Lapack Function                 | single | double | single complex | double complex |
| ------------------------------- | ------ | ------ | -------------- | -------------- |
|**rocsolver_potf2**              |     x  |    x   |                |                |
|rocsolver_potf2_batched          |     x  |    x   |                |                |
|rocsolver_potf2_strided_batched  |     x  |    x   |                |                |
|**rocsolver_potrf**              |     x  |    x   |                |                |
|rocsolver_potrf_batched          |     x  |    x   |                |                |
|rocsolver_potrf_strided_batched  |     x  |    x   |                |                |
|**rocsolver_getf2**              |     x  |    x   |   x            |  x             |
|rocsolver_getf2_batched          |     x  |    x   |   x            |  x             |
|rocsolver_getf2_strided_batched  |     x  |    x   |   x            |  x             |
|**rocsolver_getrf**              |     x  |    x   |   x            |  x             |
|rocsolver_getrf_batched          |     x  |    x   |   x            |  x             |
|rocsolver_getrf_strided_batched  |     x  |    x   |   x            |  x             |
|**rocsolver_geqr2**              |     x  |    x   |                |                |
|rocsolver_geqr2_batched          |     x  |    x   |                |                |
|rocsolver_geqr2_strided_batched  |     x  |    x   |                |                |
|**rocsolver_geqrf**              |     x  |    x   |                |                |
|rocsolver_geqrf_batched          |     x  |    x   |                |                |
|rocsolver_geqrf_strided_batched  |     x  |    x   |                |                |
|**rocsolver_gelq2**              |     x  |    x   |                |                |
|rocsolver_gelq2_batched          |     x  |    x   |                |                |
|rocsolver_gelq2_strided_batched  |     x  |    x   |                |                |
|**rocsolver_gelqf**              |     x  |    x   |                |                |
|rocsolver_gelqf_batched          |     x  |    x   |                |                |
|rocsolver_gelqf_strided_batched  |     x  |    x   |                |                |
|**rocsolver_getrs**              |     x  |    x   |   x            |  x             |
|rocsolver_getrs_batched          |     x  |    x   |   x            |  x             |
|rocsolver_getrs_strided_batched  |     x  |    x   |   x            |  x             |
