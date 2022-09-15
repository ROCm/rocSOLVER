# Change Log for rocSOLVER

Full documentation for rocSOLVER is available at [rocsolver.readthedocs.io](https://rocsolver.readthedocs.io/en/latest/).

## rocSOLVER 3.20.0 for ROCm 5.4.0
### Added
- Partial SVD for bidiagonal matrices:
    - BDSVDX
- Partial SVD for general matrices:
    - GESVDX (with batched and strided\_batched versions)

### Changed
- Changed `ROCSOLVER_EMBED_FMT` default to `ON` for users building directly with CMake.
  This matches the existing default when building with install.sh or rmake.py.


## rocSOLVER 3.19.0 for ROCm 5.3.0
### Added
- Partial eigensolver routines for symmetric/hermitian matrices:
    - SYEVX (with batched and strided\_batched versions)
    - HEEVX (with batched and strided\_batched versions)
- Generalized symmetric- and hermitian-definite partial eigensolvers:
    - SYGVX (with batched and strided\_batched versions)
    - HEGVX (with batched and strided\_batched versions)
- Eigensolver routines for symmetric/hermitian matrices using Jacobi algorithm:
    - SYEVJ (with batched and strided\_batched versions)
    - HEEVJ (with batched and strided\_batched versions)
- Generalized symmetric- and hermitian-definite eigensolvers using Jacobi algorithm:
    - SYGVJ (with batched and strided\_batched versions)
    - HEGVJ (with batched and strided\_batched versions)
- Added --profile_kernels option to rocsolver-bench, which will include kernel calls in the
  profile log (if profile logging is enabled with --profile).

### Changed
- Changed rocsolver-bench result labels `cpu_time` and `gpu_time` to
  `cpu_time_us` and `gpu_time_us`, respectively.

### Removed
- Removed dependency on cblas from the rocsolver test and benchmark clients.

### Fixed
- Fixed incorrect SYGS2/HEGS2, SYGST/HEGST, SYGV/HEGV, and SYGVD/HEGVD results for batch counts
  larger than 32.
- Fixed STEIN memory access fault when nev is 0.
- Fixed incorrect STEBZ results for close eigenvalues when range = index.
- Fixed git unsafe repository error when building with `./install.sh -cd` as a non-root user.


## rocSOLVER 3.18.0 for ROCm 5.2.0
### Added
- Partial eigenvalue decomposition routines:
    - STEBZ
    - STEIN
- Package generation for test and benchmark executables on all supported OSes using CPack.
- Added tests for multi-level logging
- Added tests for rocsolver-bench client
- File/Folder Reorg
  - Added File/Folder Reorg Changes with backward compatibility support using ROCM-CMAKE wrapper functions.

### Fixed
- Fixed compatibility with libfmt 8.1


## rocSOLVER 3.17.0 for ROCm 5.1.0
### Optimized
- Optimized non-pivoting and batch cases of the LU factorization

### Fixed
- Fixed missing synchronization in SYTRF with `rocblas_fill_lower` that could potentially
  result in incorrect pivot values.
- Fixed multi-level logging output to file with the `ROCSOLVER_LOG_PATH`,
  `ROCSOLVER_LOG_TRACE_PATH`, `ROCSOLVER_LOG_BENCH_PATH` and `ROCSOLVER_LOG_PROFILE_PATH`
  environment variables.
- Fixed performance regression in the batched LU factorization of tiny matrices


## rocSOLVER 3.16.0 for ROCm 5.0.0
### Added
- Symmetric matrix factorizations:
    - LASYF
    - SYTF2, SYTRF (with batched and strided\_batched versions)
- Added `rocsolver_get_version_string_size` to help with version string queries
- Added `rocblas_layer_mode_ex` and the ability to print kernel calls in the trace and profile logs
- Expanded batched and strided\_batched sample programs.

### Optimized
- Improved general performance of LU factorization
- Increased parallelism of specialized kernels when compiling from source, reducing build times on multi-core systems.

### Changed
- The rocsolver-test client now prints the rocSOLVER version used to run the tests,
  rather than the version used to build them
- The rocsolver-bench client now prints the rocSOLVER version used in the benchmark

### Fixed
- Added missing stdint.h include to rocsolver.h


## rocSOLVER 3.15.0 for ROCm 4.5.0
### Added
- Eigensolver routines for symmetric/hermitian matrices using Divide and Conquer algorithm:
    - STEDC
    - SYEVD (with batched and strided\_batched versions)
    - HEEVD (with batched and strided\_batched versions)
- Generalized symmetric- and hermitian-definite eigensolvers using Divide and Conquer algorithm:
    - SYGVD (with batched and strided\_batched versions)
    - HEGVD (with batched and strided\_batched versions)
- Added --mem\_query option to rocsolver-bench, which will print the amount of device memory required
  by a function.
- Added --profile option to rocsolver-bench, which will print profile logging results for a function.
- RQ factorization routines:
    - GERQ2, GERQF (with batched and strided\_batched versions)
- Linear solvers for general square systems:
    - GESV (with batched and strided\_batched versions)
- Linear solvers for symmetric/hermitian positive definite systems:
    - POTRS (with batched and strided\_batched versions)
    - POSV (with batched and strided\_batched versions)
- Inverse of symmetric/hermitian positive definite matrices:
    - POTRI (with batched and strided\_batched versions)
- General matrix inversion without pivoting:
    - GETRI\_NPVT (with batched and strided\_batched versions)
    - GETRI\_NPVT\_OUTOFPLACE (with batched and strided\_batched versions)

### Optimized
- Improved performance of LU factorization (especially for large matrix sizes)

### Changed
- The -h option of install.sh now prints a help message, instead of doing nothing.
- libfmt 7.1 is now a dependency
- Raised minimum requirement for building rocSOLVER from source to CMake 3.13
- Raised reference LAPACK version used for rocSOLVER test and benchmark clients to v3.9.1
- Minor CMake improvements for users building from source without install.sh:
    - Removed fmt::fmt from rocsolver's public usage requirements
    - Enabled small-size optimizations by default
- Split packaging into a runtime package ('rocsolver') and a development package ('rocsolver-devel').
  The development package depends on the runtime package. To aid in the transition, the runtime
  package suggests the development package (except on CentOS 7). This use of the suggests feature
  is deprecated and will be removed in a future ROCm release.

### Fixed
- Use of the GCC / Clang `__attribute__((deprecated(...)))` extension is now guarded by compiler
  detection macros.


## rocSOLVER 3.13.0 for ROCm 4.3.0
### Added
- Linear solvers for general non-square systems:
    - GELS now supports underdetermined and transposed cases
- Inverse of triangular matrices
    - TRTRI (with batched and strided\_batched versions)
- Out-of-place general matrix inversion
    - GETRI\_OUTOFPLACE (with batched and strided\_batched versions)

### Optimized
- Improved general performance of matrix inversion (GETRI)

### Changed
- Argument names for the benchmark client now match argument names from the public API

### Fixed
- Fixed known issues with Thin-SVD. The problem was identified in the test specification, not in the thin-SVD
  implementation or the rocBLAS gemm\_batched routines.
- Benchmark client will no longer crash as a result of leading dimension or stride arguments not being provided
  on the command line.


## rocSOLVER 3.12.0 for ROCm 4.2.0
### Added
- Multi-level logging functionality
- Implementation of the Thin-SVD algorithm
- Reductions of generalized symmetric- and hermitian-definite eigenproblems:
    - SYGS2, SYGST (with batched and strided\_batched versions)
    - HEGS2, HEGST (with batched and strided\_batched versions)
- Symmetric and hermitian matrix eigensolvers:
    - SYEV (with batched and strided\_batched versions)
    - HEEV (with batched and strided\_batched versions)
- Generalized symmetric- and hermitian-definite eigensolvers:
    - SYGV (with batched and strided\_batched versions)
    - HEGV (with batched and strided\_batched versions)

### Changed
- Sorting method in STERF as original quick-sort was failing for large sizes.

### Removed
- Removed hcc compiler support

### Fixed
- Fixed GELS overwriting B even when info != 0
- Error when calling STEQR with n=1 from batched routines
- Added `roc::rocblas` to the `roc::rocsolver` CMake usage requirements
- Added rocblas to the dependency list of the rocsolver deb and rpm packages
- Fixed rocblas symbol loading with dlopen and the `RTLD_NOW | RTLD_LOCAL` options

### Known Issues
- Thin-SVD implementation is failing in some cases (in particular m=300, n=120) due to a possible
  bug in the gemm\_batched routines of rocBLAS.


## rocSOLVER 3.11.0 for ROCm 4.1.0
### Added
- Eigensolver routines for symmetric/hermitian matrices:
    - STERF, STEQR
- Linear solvers for general non-square systems:
    - GELS (API added with batched and strided\_batched versions. Only the overdetermined
      non-transpose case is implemented in this release. Other cases will return
      `rocblas_status_not_implemented` status for now.)
- Extended test coverage for functions returning info
- Changelog file
- Tridiagonalization routines for symmetric and hermitian matrices:
    - LATRD
    - SYTD2, SYTRD (with batched and strided\_batched versions)
    - HETD2, HETRD (with batched and strided\_batched versions)
- Sample code and unit test for unified memory model/Heterogeneous Memory Management (HMM)

### Optimized
- Improved performance of LU factorization of small and mid-size matrices (n <= 2048)

### Changed
- Raised minimum requirement for building rocSOLVER from source to CMake 3.8
- Switched to use semantic versioning for the library
- Enabled automatic reallocation of memory workspace in rocsolver clients

### Removed
- Removed `-DOPTIMAL` from the `roc::rocsolver` CMake usage requirements. This is an internal
  rocSOLVER definition, and does not need to be defined by library users

### Fixed
- Fixed runtime errors in debug mode caused by incorrect kernel launch bounds
- Fixed complex unit test bug caused by incorrect zaxpy function signature
- Eliminated a small memory transfer that was being done on the default stream
- Fixed GESVD right singular vectors for 1x1 matrices


## rocSOLVER 3.10.0 for ROCm 3.10.0
### Added
- Orthonormal/Unitary matrix generator routines (reverse order):
    - ORG2L, UNG2L, ORGQL, UNGQL
    - ORGTR, UNGTR
- Orthonormal/Unitary matrix multiplications routines (reverse order):
    - ORM2L, UNM2L, ORMQL, UNMQL
    - ORMTR, UNMTR

### Changed
- Major library refactoring to adopt rocBLAS memory model

### Fixed
- Returned values in parameter info of functions dealing with singularities


## rocSOLVER 3.9.0 for ROCm 3.9.0
### Added
- Improved debug build mode for developers
- QL factorization routines:
    - GEQL2, GEQLF (with batched and strided\_batched versions)
- SVD of general matrices routines:
    - GESVD (with batched and strided\_batched versions)

### Optimized
- Improved performance of mid-size matrix inversion (64 < n <= 2048)


## rocSOLVER 3.8.0 for ROCm 3.8.0
### Added
- Sample codes for C, C++ and FORTRAN
- LU factorization without pivoting routines:
    - GETF2\_NPVT, GETRF\_NPVT (with batched and strided\_batched versions)

### Optimized
- Improved performance of LU factorization of mid-size matrices (64 < n <= 2048)
- Improved performance of small-size matrix inversion (n <= 64)

### Fixed
- Ensure the public API is C compatible


## rocSOLVER 3.7.0 for ROCm 3.7.0
### Added
- LU-factorization-based matrix inverse routines:
    - GETRI (with batched and strided\_batched versions)
- SVD of bidiagonal matrices routine:
    - BDSQR

### Fixed
- Ensure congruency on the input data when executing performance tests (benchmarks)


## rocSOLVER 3.6.0 for ROCm 3.6.0
### Added
- Complex precision support for all existing rocSOLVER functions
- Bidiagonalization routines:
    - LABRD
    - GEBD2, GEBRD (with batched and strided\_batched versions)
- Integration of rocSOLVER to hipBLAS

### Optimized
- Improved performance of LU factorization of tiny matrices (n <= 64)

### Changed
- Major clients refactoring to achieve better test coverage and benchmarking


## rocSOLVER 3.5.0 for ROCm 3.5.0
### Added
- Installation script and new build procedure
- Documentation and integration with ReadTheDocs
- Orthonormal matrix multiplication routines:
    - ORM2R, ORMQR
    - ORML2, ORMLQ
    - ORMBR

### Changed
- Switched to use all rocBLAS types and enumerations
- Major library refactoring to achieve better integration and rocBLAS support
- hip-clang is now default compiler

### Deprecated
- rocSOLVER types and enumerations
- hcc compiler support

