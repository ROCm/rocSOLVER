/* ************************************************************************
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_getri_outofplace.hpp"

template <typename T, typename U>
rocblas_status rocsolver_getri_outofplace_impl(rocblas_handle handle,
                                               const rocblas_int n,
                                               U A,
                                               const rocblas_int lda,
                                               rocblas_int* ipiv,
                                               U C,
                                               const rocblas_int ldc,
                                               rocblas_int* info)
{
    ROCSOLVER_ENTER_TOP("getri_outofplace", "-n", n, "--lda", lda, "--ldc", ldc);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_getri_outofplace_argCheck(handle, n, lda, ldc, A, C, ipiv, info);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;
    rocblas_int shiftP = 0;
    rocblas_int shiftC = 0;

    // normal execution
    rocblas_stride strideA = 0;
    rocblas_stride strideC = 0;
    rocblas_stride strideP = 0;
    rocblas_int batch_count = 1;

    // memory workspace sizes:
    // size of reusable workspace (for calling TRSM)
    size_t size_work1, size_work2, size_work3, size_work4;

    rocsolver_getri_outofplace_getMemorySize<false, T>(n, batch_count, &size_work1, &size_work2,
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

    // Execution
    return rocsolver_getri_outofplace_template<false, T>(
        handle, n, A, shiftA, lda, strideA, ipiv, shiftP, strideP, C, shiftC, ldc, strideC, info,
        batch_count, work1, work2, work3, work4, optim_mem);
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_sgetri_outofplace(rocblas_handle handle,
                                           const rocblas_int n,
                                           float* A,
                                           const rocblas_int lda,
                                           rocblas_int* ipiv,
                                           float* C,
                                           const rocblas_int ldc,
                                           rocblas_int* info)
{
    return rocsolver_getri_outofplace_impl<float>(handle, n, A, lda, ipiv, C, ldc, info);
}

rocblas_status rocsolver_dgetri_outofplace(rocblas_handle handle,
                                           const rocblas_int n,
                                           double* A,
                                           const rocblas_int lda,
                                           rocblas_int* ipiv,
                                           double* C,
                                           const rocblas_int ldc,
                                           rocblas_int* info)
{
    return rocsolver_getri_outofplace_impl<double>(handle, n, A, lda, ipiv, C, ldc, info);
}

rocblas_status rocsolver_cgetri_outofplace(rocblas_handle handle,
                                           const rocblas_int n,
                                           rocblas_float_complex* A,
                                           const rocblas_int lda,
                                           rocblas_int* ipiv,
                                           rocblas_float_complex* C,
                                           const rocblas_int ldc,
                                           rocblas_int* info)
{
    return rocsolver_getri_outofplace_impl<rocblas_float_complex>(handle, n, A, lda, ipiv, C, ldc,
                                                                  info);
}

rocblas_status rocsolver_zgetri_outofplace(rocblas_handle handle,
                                           const rocblas_int n,
                                           rocblas_double_complex* A,
                                           const rocblas_int lda,
                                           rocblas_int* ipiv,
                                           rocblas_double_complex* C,
                                           const rocblas_int ldc,
                                           rocblas_int* info)
{
    return rocsolver_getri_outofplace_impl<rocblas_double_complex>(handle, n, A, lda, ipiv, C, ldc,
                                                                   info);
}

} // extern C
