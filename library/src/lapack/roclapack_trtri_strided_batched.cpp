/* ************************************************************************
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_trtri.hpp"

template <typename T, typename U>
rocblas_status rocsolver_trtri_strided_batched_impl(rocblas_handle handle,
                                                    const rocblas_fill uplo,
                                                    const rocblas_diagonal diag,
                                                    const rocblas_int n,
                                                    U A,
                                                    const rocblas_int lda,
                                                    const rocblas_stride strideA,
                                                    rocblas_int* info,
                                                    const rocblas_int batch_count)
{
    ROCSOLVER_ENTER_TOP("trtri_strided_batched", "--uplo", uplo, "--diag", diag, "-n", n, "--lda",
                        lda, "--strideA", strideA, "--batch_count", batch_count);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_trtri_argCheck(handle, uplo, diag, n, lda, A, info, batch_count);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;

    // memory workspace sizes:
    // size of reusable workspace (for calling TRSM)
    bool optim_mem;
    size_t size_work1, size_work2, size_work3, size_work4;
    // size of temporary array required for copies
    size_t size_tmpcopy;
    // size of arrays of pointers (for batched cases)
    size_t size_workArr;
    rocsolver_trtri_getMemorySize<false, true, T>(diag, n, batch_count, &size_work1, &size_work2,
                                                  &size_work3, &size_work4, &size_tmpcopy,
                                                  &size_workArr, &optim_mem);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_work1, size_work2, size_work3,
                                                      size_work4, size_tmpcopy, size_workArr);

    // memory workspace allocation
    void *work1, *work2, *work3, *work4, *tmpcopy, *workArr;
    rocblas_device_malloc mem(handle, size_work1, size_work2, size_work3, size_work4, size_tmpcopy,
                              size_workArr);

    if(!mem)
        return rocblas_status_memory_error;

    work1 = mem[0];
    work2 = mem[1];
    work3 = mem[2];
    work4 = mem[3];
    tmpcopy = mem[4];
    workArr = mem[5];

    // execution
    return rocsolver_trtri_template<false, true, T>(handle, uplo, diag, n, A, shiftA, lda, strideA,
                                                    info, batch_count, work1, work2, work3, work4,
                                                    (T*)tmpcopy, (T**)workArr, optim_mem);
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_strtri_strided_batched(rocblas_handle handle,
                                                const rocblas_fill uplo,
                                                const rocblas_diagonal diag,
                                                const rocblas_int n,
                                                float* A,
                                                const rocblas_int lda,
                                                const rocblas_stride strideA,
                                                rocblas_int* info,
                                                const rocblas_int batch_count)
{
    return rocsolver_trtri_strided_batched_impl<float>(handle, uplo, diag, n, A, lda, strideA, info,
                                                       batch_count);
}

rocblas_status rocsolver_dtrtri_strided_batched(rocblas_handle handle,
                                                const rocblas_fill uplo,
                                                const rocblas_diagonal diag,
                                                const rocblas_int n,
                                                double* A,
                                                const rocblas_int lda,
                                                const rocblas_stride strideA,
                                                rocblas_int* info,
                                                const rocblas_int batch_count)
{
    return rocsolver_trtri_strided_batched_impl<double>(handle, uplo, diag, n, A, lda, strideA,
                                                        info, batch_count);
}

rocblas_status rocsolver_ctrtri_strided_batched(rocblas_handle handle,
                                                const rocblas_fill uplo,
                                                const rocblas_diagonal diag,
                                                const rocblas_int n,
                                                rocblas_float_complex* A,
                                                const rocblas_int lda,
                                                const rocblas_stride strideA,
                                                rocblas_int* info,
                                                const rocblas_int batch_count)
{
    return rocsolver_trtri_strided_batched_impl<rocblas_float_complex>(handle, uplo, diag, n, A, lda,
                                                                       strideA, info, batch_count);
}

rocblas_status rocsolver_ztrtri_strided_batched(rocblas_handle handle,
                                                const rocblas_fill uplo,
                                                const rocblas_diagonal diag,
                                                const rocblas_int n,
                                                rocblas_double_complex* A,
                                                const rocblas_int lda,
                                                const rocblas_stride strideA,
                                                rocblas_int* info,
                                                const rocblas_int batch_count)
{
    return rocsolver_trtri_strided_batched_impl<rocblas_double_complex>(
        handle, uplo, diag, n, A, lda, strideA, info, batch_count);
}

} // extern C
