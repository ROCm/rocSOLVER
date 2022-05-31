/* ************************************************************************
 * Copyright (c) 2021-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_potrs.hpp"

template <typename T>
rocblas_status rocsolver_potrs_impl(rocblas_handle handle,
                                    const rocblas_fill uplo,
                                    const rocblas_int n,
                                    const rocblas_int nrhs,
                                    T* A,
                                    const rocblas_int lda,
                                    T* B,
                                    const rocblas_int ldb)
{
    ROCSOLVER_ENTER_TOP("potrs", "--uplo", uplo, "-n", n, "--nrhs", nrhs, "--lda", lda, "--ldb", ldb);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_potrs_argCheck(handle, uplo, n, nrhs, lda, ldb, A, B);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;
    rocblas_int shiftB = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideA = 0;
    rocblas_stride strideB = 0;
    rocblas_int batch_count = 1;

    // memory workspace sizes:
    // size of workspace (for calling TRSM)
    bool optim_mem;
    size_t size_work1, size_work2, size_work3, size_work4;
    rocsolver_potrs_getMemorySize<false, false, T>(n, nrhs, batch_count, &size_work1, &size_work2,
                                                   &size_work3, &size_work4, &optim_mem);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_work1, size_work2, size_work3,
                                                      size_work4);

    // memory workspace allocation
    void *work1, *work2, *work3, *work4;
    rocblas_device_malloc mem(handle, size_work1, size_work2, size_work3, size_work4);

    if(!mem)
        return rocblas_status_memory_error;

    work1 = mem[0];
    work2 = mem[1];
    work3 = mem[2];
    work4 = mem[3];

    // execution
    return rocsolver_potrs_template<false, false, T>(handle, uplo, n, nrhs, A, shiftA, lda, strideA,
                                                     B, shiftB, ldb, strideB, batch_count, work1,
                                                     work2, work3, work4, optim_mem);
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocblas_status rocsolver_spotrs(rocblas_handle handle,
                                           const rocblas_fill uplo,
                                           const rocblas_int n,
                                           const rocblas_int nrhs,
                                           float* A,
                                           const rocblas_int lda,
                                           float* B,
                                           const rocblas_int ldb)
{
    return rocsolver_potrs_impl<float>(handle, uplo, n, nrhs, A, lda, B, ldb);
}

extern "C" rocblas_status rocsolver_dpotrs(rocblas_handle handle,
                                           const rocblas_fill uplo,
                                           const rocblas_int n,
                                           const rocblas_int nrhs,
                                           double* A,
                                           const rocblas_int lda,
                                           double* B,
                                           const rocblas_int ldb)
{
    return rocsolver_potrs_impl<double>(handle, uplo, n, nrhs, A, lda, B, ldb);
}

extern "C" rocblas_status rocsolver_cpotrs(rocblas_handle handle,
                                           const rocblas_fill uplo,
                                           const rocblas_int n,
                                           const rocblas_int nrhs,
                                           rocblas_float_complex* A,
                                           const rocblas_int lda,
                                           rocblas_float_complex* B,
                                           const rocblas_int ldb)
{
    return rocsolver_potrs_impl<rocblas_float_complex>(handle, uplo, n, nrhs, A, lda, B, ldb);
}

extern "C" rocblas_status rocsolver_zpotrs(rocblas_handle handle,
                                           const rocblas_fill uplo,
                                           const rocblas_int n,
                                           const rocblas_int nrhs,
                                           rocblas_double_complex* A,
                                           const rocblas_int lda,
                                           rocblas_double_complex* B,
                                           const rocblas_int ldb)
{
    return rocsolver_potrs_impl<rocblas_double_complex>(handle, uplo, n, nrhs, A, lda, B, ldb);
}
