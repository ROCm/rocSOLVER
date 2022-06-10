/* ************************************************************************
 * Copyright (c) 2019-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_getri_outofplace.hpp"

template <typename T, typename U>
rocblas_status rocsolver_getri_outofplace_batched_impl(rocblas_handle handle,
                                                       const rocblas_int n,
                                                       U A,
                                                       const rocblas_int lda,
                                                       rocblas_int* ipiv,
                                                       const rocblas_stride strideP,
                                                       U C,
                                                       const rocblas_int ldc,
                                                       rocblas_int* info,
                                                       const bool pivot,
                                                       const rocblas_int batch_count)
{
    const char* name = (pivot ? "getri_outofplace_batched" : "getri_npvt_outofplace_batched");
    ROCSOLVER_ENTER_TOP(name, "-n", n, "--lda", lda, "--strideP", strideP, "--ldc", ldc,
                        "--batch_count", batch_count);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_getri_outofplace_argCheck(handle, n, lda, ldc, A, C, ipiv, info,
                                                            pivot, batch_count);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;
    rocblas_int shiftP = 0;
    rocblas_int shiftC = 0;

    // batched execution
    rocblas_stride strideA = 0;
    rocblas_stride strideC = 0;

    // memory workspace sizes:
    // size of reusable workspace (for calling GETRS)
    bool optim_mem;
    size_t size_work1, size_work2, size_work3, size_work4;

    rocsolver_getri_outofplace_getMemorySize<true, false, T>(
        n, batch_count, &size_work1, &size_work2, &size_work3, &size_work4, &optim_mem);

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

    // Execution
    return rocsolver_getri_outofplace_template<true, false, T>(
        handle, n, A, shiftA, lda, strideA, ipiv, shiftP, strideP, C, shiftC, ldc, strideC, info,
        batch_count, work1, work2, work3, work4, optim_mem, pivot);
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_sgetri_outofplace_batched(rocblas_handle handle,
                                                   const rocblas_int n,
                                                   float* const A[],
                                                   const rocblas_int lda,
                                                   rocblas_int* ipiv,
                                                   const rocblas_stride strideP,
                                                   float* const C[],
                                                   const rocblas_int ldc,
                                                   rocblas_int* info,
                                                   const rocblas_int batch_count)
{
    return rocsolver_getri_outofplace_batched_impl<float>(handle, n, A, lda, ipiv, strideP, C, ldc,
                                                          info, true, batch_count);
}

rocblas_status rocsolver_dgetri_outofplace_batched(rocblas_handle handle,
                                                   const rocblas_int n,
                                                   double* const A[],
                                                   const rocblas_int lda,
                                                   rocblas_int* ipiv,
                                                   const rocblas_stride strideP,
                                                   double* const C[],
                                                   const rocblas_int ldc,
                                                   rocblas_int* info,
                                                   const rocblas_int batch_count)
{
    return rocsolver_getri_outofplace_batched_impl<double>(handle, n, A, lda, ipiv, strideP, C, ldc,
                                                           info, true, batch_count);
}

rocblas_status rocsolver_cgetri_outofplace_batched(rocblas_handle handle,
                                                   const rocblas_int n,
                                                   rocblas_float_complex* const A[],
                                                   const rocblas_int lda,
                                                   rocblas_int* ipiv,
                                                   const rocblas_stride strideP,
                                                   rocblas_float_complex* const C[],
                                                   const rocblas_int ldc,
                                                   rocblas_int* info,
                                                   const rocblas_int batch_count)
{
    return rocsolver_getri_outofplace_batched_impl<rocblas_float_complex>(
        handle, n, A, lda, ipiv, strideP, C, ldc, info, true, batch_count);
}

rocblas_status rocsolver_zgetri_outofplace_batched(rocblas_handle handle,
                                                   const rocblas_int n,
                                                   rocblas_double_complex* const A[],
                                                   const rocblas_int lda,
                                                   rocblas_int* ipiv,
                                                   const rocblas_stride strideP,
                                                   rocblas_double_complex* const C[],
                                                   const rocblas_int ldc,
                                                   rocblas_int* info,
                                                   const rocblas_int batch_count)
{
    return rocsolver_getri_outofplace_batched_impl<rocblas_double_complex>(
        handle, n, A, lda, ipiv, strideP, C, ldc, info, true, batch_count);
}

rocblas_status rocsolver_sgetri_npvt_outofplace_batched(rocblas_handle handle,
                                                        const rocblas_int n,
                                                        float* const A[],
                                                        const rocblas_int lda,
                                                        float* const C[],
                                                        const rocblas_int ldc,
                                                        rocblas_int* info,
                                                        const rocblas_int batch_count)
{
    rocblas_int* ipiv = nullptr;
    return rocsolver_getri_outofplace_batched_impl<float>(handle, n, A, lda, ipiv, 0, C, ldc, info,
                                                          false, batch_count);
}

rocblas_status rocsolver_dgetri_npvt_outofplace_batched(rocblas_handle handle,
                                                        const rocblas_int n,
                                                        double* const A[],
                                                        const rocblas_int lda,
                                                        double* const C[],
                                                        const rocblas_int ldc,
                                                        rocblas_int* info,
                                                        const rocblas_int batch_count)
{
    rocblas_int* ipiv = nullptr;
    return rocsolver_getri_outofplace_batched_impl<double>(handle, n, A, lda, ipiv, 0, C, ldc, info,
                                                           false, batch_count);
}

rocblas_status rocsolver_cgetri_npvt_outofplace_batched(rocblas_handle handle,
                                                        const rocblas_int n,
                                                        rocblas_float_complex* const A[],
                                                        const rocblas_int lda,
                                                        rocblas_float_complex* const C[],
                                                        const rocblas_int ldc,
                                                        rocblas_int* info,
                                                        const rocblas_int batch_count)
{
    rocblas_int* ipiv = nullptr;
    return rocsolver_getri_outofplace_batched_impl<rocblas_float_complex>(
        handle, n, A, lda, ipiv, 0, C, ldc, info, false, batch_count);
}

rocblas_status rocsolver_zgetri_npvt_outofplace_batched(rocblas_handle handle,
                                                        const rocblas_int n,
                                                        rocblas_double_complex* const A[],
                                                        const rocblas_int lda,
                                                        rocblas_double_complex* const C[],
                                                        const rocblas_int ldc,
                                                        rocblas_int* info,
                                                        const rocblas_int batch_count)
{
    rocblas_int* ipiv = nullptr;
    return rocsolver_getri_outofplace_batched_impl<rocblas_double_complex>(
        handle, n, A, lda, ipiv, 0, C, ldc, info, false, batch_count);
}

} // extern C
