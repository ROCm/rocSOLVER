# Changelog for rocSOLVER

Documentation for rocSOLVER is available at
[https://rocm.docs.amd.com/projects/rocSOLVER/en/latest/](https://rocm.docs.amd.com/projects/rocSOLVER/en/latest/).

## (Unreleased) rocSOLVER
### Changes
### Deprecations
### Fixes
### Known Issues
### Security

## (unreleased) rocSOLVER 3.25.0 for ROCm 6.1.0

### Changes

* Relaxed array length requirements for GESVDX with `rocblas_srange_index`

### Fixes

* Singular vector normalization in BDSVDX and GESVDX
* Potential memory access fault in STEIN, SYEVX/HEEVX, SYGVX/HEGVX, BDSVDX, and GESVDX

## rocSOLVER 3.24.0 for ROCm 6.0.0

### Additions

* Cholesky refactorization for sparse matrices: `CSRRF_REFACTCHOL`
* Added `rocsolver_rfinfo_mode` and the ability to specify the desired refactorization routine (see `rocsolver_set_rfinfo_mode`)

### Changes

* `CSRRF_ANALYSIS` and `CSRRF_SOLVE` now support sparse Cholesky factorization

## rocSOLVER 3.23.0 for ROCm 5.7.0

### Additions

* LU factorization without pivoting for block tridiagonal matrices:
  * `GEBLTTRF_NPVT` now supports the `interleaved_batched` format
* Linear system solver without pivoting for block tridiagonal matrices:
  * `GEBLTTRS_NPVT` now supports the `interleaved_batched` format

### Fixes

* Stack overflow in sparse tests on Windows

### Changes

* Changed `rocsolver-test` sparse input data search paths to be relative to the test executable
* Changed build scripts to default to compressed debug symbols in debug builds

## rocSOLVER 3.22.0 for ROCm 5.6.0

### Additions

* LU refactorization for sparse matrices:
  * `CSRRF_ANALYSIS`
  * `CSRRF_SUMLU`
  * `CSRRF_SPLITLU`
  * `CSRRF_REFACTLU`
* Linear system solver for sparse matrices
  * `CSRRF_SOLVE`
* Added type `rocsolver_rfinfo` for use with sparse matrix routines

### Optimizations

* Improved the performance of BDSQR and GESVD when singular vectors are requested

### Fixes

* BDSQR and GESVD no longer hangs when the input contains `NaN` or `Inf`

## rocSOLVER 3.21.0 for ROCm 5.5.0

### Additions

* SVD for general matrices using the Jacobi algorithm:
  * GESVDJ (with `batched` and `strided_batched` versions)
* LU factorization without pivoting for block tridiagonal matrices:
  * `GEBLTTRF_NPVT` (with `batched` and `strided_batched` versions)
* Linear system solver without pivoting for block tridiagonal matrices:
  * `GEBLTTRS_NPVT` (with `batched` and `strided_batched` versions)
* Product of triangular matrices
  * LAUUM
* Added experimental hipGraph support for rocSOLVER functions

### Optimizations

* Improved the performance of SYEVJ/HEEVJ

### Changes

* STEDC, SYEVD/HEEVD and SYGVD/HEGVD now use fully implemented divide-and-conquer approach

### Fixes

* SYEVJ/HEEVJ should now be invariant under matrix scaling
* SYEVJ/HEEVJ should now properly output the eigenvalues when no sweeps are executed
* `GETF2_NPVT` and `GETRF_NPVT` input data initialization in tests and benchmarks
* Fixed rocBLAS missing from the dependency list of the rocSOLVER deb and rpm packages

## rocSOLVER 3.20.0 for ROCm 5.4.0

### Additions

* Partial SVD for bidiagonal matrices: `BDSVDX`
* Partial SVD for general matrices: `GESVDX` (with `batched` and `strided_batched` versions)

### Changes

* Changed `ROCSOLVER_EMBED_FMT` default to `ON` for users building directly with CMake; this
  matches the existing default when building with `install.sh` or `rmake.py`

## rocSOLVER 3.19.0 for ROCm 5.3.0

### Additions

* Partial eigensolver routines for symmetric and hermitian matrices:
  * `SYEVX` (with `batched` and `strided_batched` versions)
  * `HEEVX` (with `batched` and `strided_batched` versions)
* Generalized symmetric- and hermitian-definite partial eigensolvers:
  * `SYGVX` (with `batched` and `strided_batched` versions)
  * `HEGVX` (with `batched` and `strided_batched` versions)
* Eigensolver routines for symmetric and hermitian matrices using the Jacobi algorithm:
    * `SYEVJ` (with `batched` and `strided_batched` versions)
    * `HEEVJ` (with `batched` and `strided_batched` versions)
* Generalized symmetric- and hermitian-definite eigensolvers using Jacobi algorithm:
  * `SYGVJ` (with `batched` and `strided_batched` versions)
  * `HEGVJ` (with `batched` and `strided_batched` versions)
* Added `--profile_kernels` option to `rocsolver-bench`, which include kernel calls in the profile log (if
  profile logging is enabled with `--profile`)

### Changes

* `rocsolver-bench` result label `cpu_time` is now `cpu_time_us`
* `rocsolver-bench` result label `gpu_time` is now `gpu_time_us`

### Removals

* Removed dependency on CBLAS from rocSOLVER test and benchmark clients

### Fixes

* Incorrect SYGS2/HEGS2, SYGST/HEGST, SYGV/HEGV, and SYGVD/HEGVD results for batch counts
  larger than 32
* STEIN memory access fault when nev is 0
* Incorrect STEBZ results for close eigenvalues when `range = index`
* Git unsafe repository error when building with `./install.sh -cd` as a non-root user

## rocSOLVER 3.18.0 for ROCm 5.2.0

### Additions

* Partial eigenvalue decomposition routines: `STEBZ` and `STEIN`
* Package generation for test and benchmark executables on all supported operating systems using
  CPack
* Added tests for multi-level logging
* Added tests for `rocsolver-bench` client
* File and folder reorganization with backward compatibility support using ROCm-CMake wrapper
  functions

### Fixes

* Compatibility issue with libfmt 8.1

## rocSOLVER 3.17.0 for ROCm 5.1.0

### Optimizations

* Optimized non-pivoting and batch cases of the LU factorization

### Fixes

* Fixed missing synchronization in SYTRF with `rocblas_fill_lower`, which could potentially result in
  incorrect pivot values
* Fixed multi-level logging output to file with the `ROCSOLVER_LOG_PATH`,
  `ROCSOLVER_LOG_TRACE_PATH`, `ROCSOLVER_LOG_BENCH_PATH`, and `ROCSOLVER_LOG_PROFILE_PATH`
  environment variables
* Fixed performance regression in the batched LU factorization of tiny matrices

## rocSOLVER 3.16.0 for ROCm 5.0.0

### Additions

* Symmetric matrix factorizations:
  * LASYF
  * SYTF2 and SYTRF (with `batched` and `strided_batched` versions)
* `rocsolver_get_version_string_size` to help with version string queries
* `rocblas_layer_mode_ex` and the ability to print kernel calls in trace and profile logs
* Expanded `batched` and `strided_batched` sample programs

### Optimizations

* Improved general performance of LU factorization
* Increased parallelism of specialized kernels when compiling from source, reducing build times on
  multi-core systems

### Changes

* The `rocsolver-test` client now prints the rocSOLVER version used to run the tests, rather than the
  version used to build them
* The `rocsolver-bench` client now prints the rocSOLVER version used in the benchmark

### Fixes

* Added missing `stdint.h` include to `rocsolver.h`

## rocSOLVER 3.15.0 for ROCm 4.5.0

### Additions

* Eigensolver routines for symmetric and hermitian matrices using the divide-and-conquer algorithm:
  * STEDC
  * SYEVD (with `batched` and `strided_batched` versions)
  * HEEVD (with `batched` and `strided_batched` versions)
* Generalized symmetric- and hermitian-definite eigensolvers using the divide-and-conquer algorithm:
  * SYGVD (with `batched` and `strided_batched` versions)
  * HEGVD (with `batched` and `strided_batched` versions)
* Added the `--mem_query` option to `rocsolver-bench`, which prints the amount of device memory
  required by a function
* Added the `--profile` option to `rocsolver-bench`, which prints profile logging results for a function
* RQ factorization routines:
  * GERQ2 and GERQF (with `batched` and `strided_batched` versions)
* Linear solvers for general square systems:
  * GESV (with `batched` and `strided_batched` versions)
* Linear solvers for symmetric and hermitian positive definite systems:
  * POTRS (with `batched` and `strided_batched` versions)
  * POSV (with `batched` and `strided_batched` versions)
* Inverse of symmetric and hermitian positive definite matrices:
  * POTRI (with batched and strided\_batched versions)
* General matrix inversion without pivoting:
  * `GETRI_NPVT` (with `batched` and `strided_batched` versions)
  * `GETRI_NPVT_OUTOFPLACE` (with `batched` and `strided_batched` versions)

### Optimizations

* Improved the performance of LU factorization (especially for large matrix sizes)

### Changes

* The `-h` option of `install.sh` now prints a help message, instead of doing nothing
* libfmt 7.1 is now a dependency
* Raised minimum requirement for building rocSOLVER from source to CMake 3.13
* Raised reference LAPACK version used for rocSOLVER test and benchmark clients to v3.9.1
* Minor CMake improvements for users building from source without install.sh:
  * Removed fmt::fmt from rocsolver's public usage requirements
  * Enabled small-size optimizations by default
* Split packaging into a runtime package ('rocsolver') and a development package ('rocsolver-devel')
  The development package depends on the runtime package. To aid in the transition, the runtime
  package suggests the development package (except on CentOS 7). This use of the suggests feature
  is deprecated and will be removed in a future ROCm release

### Fixes

* Use of the GCC/Clang `__attribute__((deprecated(...)))` extension is now guarded by a compiler
  detection macros

## rocSOLVER 3.13.0 for ROCm 4.3.0

### Additions

* Linear solvers for general non-square systems: GELS now supports underdetermined and transposed
  cases
* Inverse of triangular matrices: TRTRI (with `batched` and `strided_batched` versions)
* Out-of-place general matrix inversion: GETRI_OUTOFPLACE (with `batched` and `strided_batched`
  versions)

### Optimizations

* Improved general performance of matrix inversion (GETRI)

### Changes

* Argument names for the benchmark client now match argument names from the public API

### Fixes

* Known issues with Thin-SVD
  * The problem was identified in the test specification, not in the thin-SVD implementation or the
    rocBLAS `gemm_batched` routines
* The benchmark client will no longer crash when leading dimension or stride arguments are not
    provided in the command line

## rocSOLVER 3.12.0 for ROCm 4.2.0

### Additions

* Multi-level logging functionality
* Implementation of the Thin-SVD algorithm
* Reductions of generalized symmetric- and hermitian-definite eigenproblems:
  * SYGS2, SYGST (with `batched` and strided_batched versions)
  * HEGS2, HEGST (with `batched` and strided_batched versions)
* Symmetric and hermitian matrix eigensolvers:
  * SYEV (with `batched` and `strided_batched` versions)
  * HEEV (with `batched` and `strided_batched` versions)
* Generalized symmetric- and hermitian-definite eigensolvers:
  * SYGV (with `batched` and `strided_batched` versions)
  * HEGV (with `batched` and `strided_batched` versions)

### Changes

* Sorting method in STERF (original quick-sort was failing for large sizes)

### Removals

* Removed hcc compiler support

### Fixes

* GELS overwriting B, even when info != 0
* Error when calling STEQR with n=1 from batched routines
* Added `roc::rocblas` to the `roc::rocsolver` CMake usage requirements
* Added rocBLAS to the dependency list of the rocSOLVER deb and rpm packages
* Fixed rocBLAS symbol loading with dlopen and the `RTLD_NOW | RTLD_LOCAL` options

### Known issues

* Thin-SVD implementation is failing in some cases (in particular m=300, n=120) due to a possible
  bug in rocBLAS `gemm_batched` routines

## rocSOLVER 3.11.0 for ROCm 4.1.0

### Additions

* Eigensolver routines for symmetric and hermitian matrices:
  * STERF and STEQR
* Linear solvers for general non-square systems:
  * GELS (with `batched` and `strided_batched` versions)
    * Only the overdetermined, non-transpose case is implemented in this release (other cases return
      `rocblas_status_not_implemented` status)
* Extended test coverage for functions returning info
* Changelog file
* Tridiagonalization routines for symmetric and hermitian matrices:
  * LATRD
  * SYTD2, SYTRD (with `batched` and `strided_batched` versions)
  * HETD2, HETRD (with `batched` and `strided_batched` versions)
* Sample code and unit test for unified memory model and Heterogeneous Memory Management
  (HMM)

### Optimizations

* Improved performance of LU factorization of small and mid-size matrices (n <= 2048)

### Changes

* CMake 3.8 is now the minimum requirement for building rocSOLVER from source
* Switched to semantic versioning
* Enabled automatic reallocation of memory workspace in rocSOLVER clients

### Removals

* Removed `-DOPTIMAL` from the `roc::rocsolver` CMake usage requirements (this is an internal
  definition)

### Fixes

* Runtime errors in debug mode caused by incorrect kernel launch bounds
* Complex unit test bug caused by incorrect ZAXPY function signature
* Eliminated a small memory transfer on the default stream
* GESVD right singular vectors for 1x1 matrices

## rocSOLVER 3.10.0 for ROCm 3.10.0

### Additions

* Orthonormal and unitary matrix generator routines (reverse order):
  * ORG2L, UNG2L, ORGQL, UNGQL
  * ORGTR, UNGTR
* Orthonormal and unitary matrix multiplications routines (reverse order):
  * ORM2L, UNM2L, ORMQL, UNMQL
  * ORMTR, UNMTR

### Changes

* Major library refactoring to adopt the rocBLAS memory model

### Fixes

* Returned values in the parameter information of functions dealing with singularities

## rocSOLVER 3.9.0 for ROCm 3.9.0

### Additions

* Improved debug build mode
* QL factorization routines:
  * GEQL2, GEQLF (with `batched` and `strided_batched` versions)
* SVD of general matrices routines:
  * GESVD (with `batched` and `strided_batched` versions)

### Optimizations

* Improved performance of mid-size matrix inversions (64 < n <= 2048)

## rocSOLVER 3.8.0 for ROCm 3.8.0

### Additions

* Sample codes for C, C++, and FORTRAN
* LU factorization without pivoting routines:
  * GETF2_NPVT, GETRF_NPVT (with `batched` and `strided_batched` versions)

### Optimizations

* Improved performance of LU factorization of mid-size matrices (64 < n <= 2048)
* Improved performance of small-size matrix inversion (n <= 64)

### Fixes

* Ensures the public API is C compatible

## rocSOLVER 3.7.0 for ROCm 3.7.0

### Additions

* LU-factorization-based matrix inverse routines:
  * GETRI (with `batched` and `strided_batched` versions)
* SVD of bidiagonal matrices routine:
  * BDSQR

### Fixes

* Congruency on input data when running performance tests (benchmarks)

## rocSOLVER 3.6.0 for ROCm 3.6.0

### Additions

* Complex precision support for all existing rocSOLVER functions
* Bidiagonalization routines:
  * LABRD
  * GEBD2, GEBRD (with `batched` and `strided_batched` versions)
* Integration of rocSOLVER to hipBLAS

### Optimizations

* Improved the performance of LU factorization of tiny matrices (n <= 64)

### Changes

* Major client refactoring to achieve better test coverage and benchmarking

## rocSOLVER 3.5.0 for ROCm 3.5.0

### Additions

* Installation script and new build procedure
* Documentation and integration with ReadTheDocs
* Orthonormal matrix multiplication routines:
  * ORM2R, ORMQR
  * ORML2, ORMLQ
  * ORMBR

### Changes

* Switched to use all rocBLAS types and enumerations
* Major library refactoring to achieve better integration and rocBLAS support
* hip-clang is now the default compiler

### Deprecations

* rocSOLVER types and enumerations
* hcc compiler support
