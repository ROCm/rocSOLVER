# Change Log for rocSOLVER
 
Full documentation for rocSOLVER is available at [rocsolver.readthedocs.io](https://rocsolver.readthedocs.io/en/latest/).
 
## [(Unreleased) rocSOLVER x.y.z for ROCm 4.0.0]
### Added
__anchor__TA

__anchor__CB

__anchor__JZ
- Extended test coverage for functions returning info 

### Optimizations
__anchor__TA

__anchor__CB

__anchor__JZ

### Changed
__anchor__TA

__anchor__CB

__anchor__JZ

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
- Documentation improvements
- Orthonormal/Unitary matrix generator routines (reverse order):
    - ORG2L, UNG2L, ORGQL, UNGQL
    - ORGTR, UNGTR
- Orthonormal/Unitary matrix multiplications routines (reverse order):
    - ORM2L, UNM2L, ORMQL, UNMQL
    - ORMTR, UNMTR

### Changed
- Major library refactoring to adopt rocBLAS memory model

### Fixed 
- Different bugs in unit tests and clients
- Return values in info parameter of functions dealing with singularities



## [rocSOLVER 3.9.0 for ROCm 3.9.0]
### Added
- Option to build documentation from source
- Improved debug build mode for developers
- QL factorization routines:
    - GEQL2, GEQLF (with batched and strided batched versions)
- SVD of general matrices routines:
    - GESVD (with batched and strided\_batched versions)

### Optimizations
- Mid-size matrix inversion (64 < n <= 2048)

### Changed
- Code is now clang-formated 



## [rocSOLVER 3.8.0 for ROCm 3.8.0]
### Added
- Sample codes for C, C++ and FORTRAN
- Documentation items and documentation improvements
- LU factorization without pivoting routines:
    - GETF2\_NPVT, GETRF\_NPVT (with batched and strided\_batched versions)

### Optimizations
- LU factorization of mid-size matrices (64 < n <= 2048)
- Small-size matrix inversion (n <= 64)



## [rocSOLVER 3.7.0 for ROCm 3.7.0]
### Added
- LU-factorization-based matrix inverse routines:
    - GETRI (with batched and strided\_batched versions)
- SVD of bidiagonal matrices routine:
    - BDSQR
- Documentation items and documentation improvements

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
- LU factorization of tiny matrices (n <= 64)

### Changed
- Major clients refactoring to achieve better test coverage and bechmarking



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

















