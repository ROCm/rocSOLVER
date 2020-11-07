# Change Log for rocSOLVER

Full documentation for rocSOLVER is available at [rocsolver.readthedocs.io](https://rocsolver.readthedocs.io/en/latest/).

## [(Unreleased) rocSOLVER for ROCm 4.0.0]
### Added
__anchor__TA

__anchor__CB

__anchor__JZ
- Extended test coverage for functions returning info
- Changelog file
- Tridiagonalization routines for symmetric and hermitian matrices:
    - LATRD
    - SYTD2, SYTRD (with batched and strided\_batched versions)
    - HETD2, HETRD (with batched and strided\_batched versions)

### Optimizations
__anchor__TA

__anchor__CB

__anchor__JZ

### Changed
__anchor__TA

__anchor__CB

__anchor__JZ
- Switched to use semantic versioning for the library

### Deprecated
__anchor__TA

__anchor__CB

__anchor__JZ

### Removed
__anchor__TA

__anchor__CB

__anchor__JZ

### Fixed
__anchor__TA

__anchor__CB

__anchor__JZ

### Known Issues
__anchor__TA

__anchor__CB

__anchor__JZ

### Security
__anchor__TA

__anchor__CB

__anchor__JZ



## [(Unreleased) rocSOLVER 3.10.0 for ROCm 3.10.0]
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



## [rocSOLVER 3.9.0 for ROCm 3.9.0]
### Added
- Improved debug build mode for developers
- QL factorization routines:
    - GEQL2, GEQLF (with batched and strided\_batched versions)
- SVD of general matrices routines:
    - GESVD (with batched and strided\_batched versions)

### Optimizations
- Improved performance of mid-size matrix inversion (64 < n <= 2048)



## [rocSOLVER 3.8.0 for ROCm 3.8.0]
### Added
- Sample codes for C, C++ and FORTRAN
- LU factorization without pivoting routines:
    - GETF2\_NPVT, GETRF\_NPVT (with batched and strided\_batched versions)

### Optimizations
- Improved performance of LU factorization of mid-size matrices (64 < n <= 2048)
- Improved performance of small-size matrix inversion (n <= 64)

### Fixed
- Ensure the public API is C compatible


## [rocSOLVER 3.7.0 for ROCm 3.7.0]
### Added
- LU-factorization-based matrix inverse routines:
    - GETRI (with batched and strided\_batched versions)
- SVD of bidiagonal matrices routine:
    - BDSQR

### Fixed
- Ensure congruency on the input data when executing performance tests (benchmarks)



## [rocSOLVER 3.6.0 for ROCm 3.6.0]
### Added
- Complex precision support for all existing rocSOLVER functions
- Bidiagonalization routines:
    - LABRD
    - GEBD2, GEBRD (with batched and strided\_batched versions)
- Integration of rocSOLVER to hipBLAS

### Optimizations
- Improved performance of LU factorization of tiny matrices (n <= 64)

### Changed
- Major clients refactoring to achieve better test coverage and benchmarking



## [rocSOLVER 3.5.0 for ROCm 3.5.0]
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

