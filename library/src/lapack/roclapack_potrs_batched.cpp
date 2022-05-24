/* ************************************************************************
 * Copyright (c) 2019-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_potrs.hpp"

template <typename T, typename U>
rocblas_status rocsolver_potrs_batched_impl(rocblas_handle handle,
                                            const rocblas_fill uplo,
                                            const rocblas_int n,
                                            const rocblas_int nrhs,
                                            U A,
                                            const rocblas_int lda,
                                            U B,
                                            const rocblas_int ldb,
                                            const rocblas_int batch_count)
{
    ROCSOLVER_ENTER_TOP("potrs_batched", "--uplo", uplo, "-n", n, "--nrhs", nrhs, "--lda", lda,
                        "--ldb", ldb, "--batch_count", batch_count);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_potrs_argCheck(handle, uplo, n, nrhs, lda, ldb, A, B, batch_count);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;
    rocblas_int shiftB = 0;

    // batched execution
    rocblas_stride strideA = 0;
    rocblas_stride strideB = 0;

    // memory workspace sizes:
    // size of workspace (for calling TRSM)
    bool optim_mem;
    size_t size_work1, size_work2, size_work3, size_work4;
    rocsolver_potrs_getMemorySize<true, false, T>(n, nrhs, batch_count, &size_work1, &size_work2,
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
    return rocsolver_potrs_template<true, false, T>(handle, uplo, n, nrhs, A, shiftA, lda, strideA,
                                                    B, shiftB, ldb, strideB, batch_count, work1,
                                                    work2, work3, work4, optim_mem);
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocblas_status rocsolver_spotrs_batched(rocblas_handle handle,
                                                   const rocblas_fill uplo,
                                                   const rocblas_int n,
                                                   const rocblas_int nrhs,
                                                   float* const A[],
                                                   const rocblas_int lda,
                                                   float* const B[],
                                                   const rocblas_int ldb,
                                                   const rocblas_int batch_count)
{
    return rocsolver_potrs_batched_impl<float>(handle, uplo, n, nrhs, A, lda, B, ldb, batch_count);
}

extern "C" rocblas_status rocsolver_dpotrs_batched(rocblas_handle handle,
                                                   const rocblas_fill uplo,
                                                   const rocblas_int n,
                                                   const rocblas_int nrhs,
                                                   double* const A[],
                                                   const rocblas_int lda,
                                                   double* const B[],
                                                   const rocblas_int ldb,
                                                   const rocblas_int batch_count)
{
    return rocsolver_potrs_batched_impl<double>(handle, uplo, n, nrhs, A, lda, B, ldb, batch_count);
}

extern "C" rocblas_status rocsolver_cpotrs_batched(rocblas_handle handle,
                                                   const rocblas_fill uplo,
                                                   const rocblas_int n,
                                                   const rocblas_int nrhs,
                                                   rocblas_float_complex* const A[],
                                                   const rocblas_int lda,
                                                   rocblas_float_complex* const B[],
                                                   const rocblas_int ldb,
                                                   const rocblas_int batch_count)
{
    return rocsolver_potrs_batched_impl<rocblas_float_complex>(handle, uplo, n, nrhs, A, lda, B,
                                                               ldb, batch_count);
}

extern "C" rocblas_status rocsolver_zpotrs_batched(rocblas_handle handle,
                                                   const rocblas_fill uplo,
                                                   const rocblas_int n,
                                                   const rocblas_int nrhs,
                                                   rocblas_double_complex* const A[],
                                                   const rocblas_int lda,
                                                   rocblas_double_complex* const B[],
                                                   const rocblas_int ldb,
                                                   const rocblas_int batch_count)
{
    return rocsolver_potrs_batched_impl<rocblas_double_complex>(handle, uplo, n, nrhs, A, lda, B,
                                                                ldb, batch_count);
}
