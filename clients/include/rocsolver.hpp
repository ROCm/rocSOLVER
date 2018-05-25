#ifndef ROCSOLVER_HPP
#define ROCSOLVER_HPP

#include "rocsolver.h"

template <typename T>
rocblas_status
rocsolver_potf2(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, T* A, rocblas_int lda);

template <>
rocblas_status
rocsolver_potf2(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, float* A, rocblas_int lda)
{
    return rocsolver_spotf2(handle, uplo, n, A, lda);
}

template <>
rocblas_status
rocsolver_potf2(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, double* A, rocblas_int lda)
{
    return rocsolver_dpotf2(handle, uplo, n, A, lda);
}

#endif /* ROCSOLVER_HPP */

