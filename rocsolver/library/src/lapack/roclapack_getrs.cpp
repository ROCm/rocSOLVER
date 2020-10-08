/* ************************************************************************
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_getrs.hpp"

template <typename T>
rocblas_status rocsolver_getrs_impl(rocblas_handle handle,
                                    const rocblas_operation trans,
                                    const rocblas_int n,
                                    const rocblas_int nrhs,
                                    T* A,
                                    const rocblas_int lda,
                                    const rocblas_int* ipiv,
                                    T* B,
                                    const rocblas_int ldb)
{
    if(!handle)
        return rocblas_status_invalid_handle;

    // logging is missing ???

    // argument checking
    rocblas_status st = rocsolver_getrs_argCheck(trans, n, nrhs, lda, ldb, A, B, ipiv);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;
    rocblas_int shiftB = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideA = 0;
    rocblas_stride strideB = 0;
    rocblas_stride strideP = 0;
    rocblas_int batch_count = 1;

    // memory workspace sizes:
    // size of workspace (for calling TRSM)
    size_t size_work1, size_work2, size_work3, size_work4;
    rocsolver_getrs_getMemorySize<false, T>(n, nrhs, batch_count, &size_work1, &size_work2,
                                            &size_work3, &size_work4);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_work1, size_work2, size_work3,
                                                      size_work4);

    // always allocate all required memory for TRSM optimal performance
    bool optim_mem = true;

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
    return rocsolver_getrs_template<false, T>(handle, trans, n, nrhs, A, shiftA, lda, strideA, ipiv,
                                              strideP, B, shiftB, ldb, strideB, batch_count, work1,
                                              work2, work3, work4, optim_mem);
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocblas_status rocsolver_sgetrs(rocblas_handle handle,
                                           const rocblas_operation trans,
                                           const rocblas_int n,
                                           const rocblas_int nrhs,
                                           float* A,
                                           const rocblas_int lda,
                                           const rocblas_int* ipiv,
                                           float* B,
                                           const rocblas_int ldb)
{
    return rocsolver_getrs_impl<float>(handle, trans, n, nrhs, A, lda, ipiv, B, ldb);
}

extern "C" rocblas_status rocsolver_dgetrs(rocblas_handle handle,
                                           const rocblas_operation trans,
                                           const rocblas_int n,
                                           const rocblas_int nrhs,
                                           double* A,
                                           const rocblas_int lda,
                                           const rocblas_int* ipiv,
                                           double* B,
                                           const rocblas_int ldb)
{
    return rocsolver_getrs_impl<double>(handle, trans, n, nrhs, A, lda, ipiv, B, ldb);
}

extern "C" rocblas_status rocsolver_cgetrs(rocblas_handle handle,
                                           const rocblas_operation trans,
                                           const rocblas_int n,
                                           const rocblas_int nrhs,
                                           rocblas_float_complex* A,
                                           const rocblas_int lda,
                                           const rocblas_int* ipiv,
                                           rocblas_float_complex* B,
                                           const rocblas_int ldb)
{
    return rocsolver_getrs_impl<rocblas_float_complex>(handle, trans, n, nrhs, A, lda, ipiv, B, ldb);
}

extern "C" rocblas_status rocsolver_zgetrs(rocblas_handle handle,
                                           const rocblas_operation trans,
                                           const rocblas_int n,
                                           const rocblas_int nrhs,
                                           rocblas_double_complex* A,
                                           const rocblas_int lda,
                                           const rocblas_int* ipiv,
                                           rocblas_double_complex* B,
                                           const rocblas_int ldb)
{
    return rocsolver_getrs_impl<rocblas_double_complex>(handle, trans, n, nrhs, A, lda, ipiv, B, ldb);
}
