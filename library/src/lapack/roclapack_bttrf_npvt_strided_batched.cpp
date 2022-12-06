/* ************************************************************************
 * Copyright (c) 2021-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_bttrf_npvt.hpp"

template <typename T, typename U>
rocblas_status rocsolver_bttrf_npvt_strided_batched_impl(rocblas_handle handle,
                                                         const rocblas_int nb,
                                                         const rocblas_int nblocks,
                                                         U A,
                                                         const rocblas_int lda,
                                                         const rocblas_stride strideA,
                                                         U B,
                                                         const rocblas_int ldb,
                                                         const rocblas_stride strideB,
                                                         U C,
                                                         const rocblas_int ldc,
                                                         const rocblas_stride strideC,
                                                         rocblas_int* info,
                                                         const rocblas_int batch_count)
{
    ROCSOLVER_ENTER_TOP("bttrf_npvt_strided_batched", "--nb", nb, "--nblocks", nblocks, "--lda",
                        lda, "--strideA", strideA, "--ldb", ldb, "--strideB", strideB, "--ldc", ldc,
                        "--strideC", strideC, "--batch_count", batch_count);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_bttrf_npvt_argCheck(handle, nb, nblocks, lda, ldb, ldc, A, B, C,
                                                      info, batch_count);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;
    rocblas_int shiftB = 0;
    rocblas_int shiftC = 0;

    // memory workspace sizes:
    // size of reusable workspace
    size_t size_work;

    rocsolver_bttrf_npvt_getMemorySize<false, true, T>(nb, nblocks, batch_count, &size_work);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_work);

    // memory workspace allocation
    void* work;
    rocblas_device_malloc mem(handle, size_work);

    if(!mem)
        return rocblas_status_memory_error;
    work = mem[0];

    // Execution
    return rocsolver_bttrf_npvt_template<false, true, T>(handle, nb, nblocks, A, shiftA, lda,
                                                         strideA, B, shiftB, ldb, strideB, C, shiftC,
                                                         ldc, strideC, info, batch_count, work);
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_sbttrf_npvt_strided_batched(rocblas_handle handle,
                                                     const rocblas_int nb,
                                                     const rocblas_int nblocks,
                                                     float* A,
                                                     const rocblas_int lda,
                                                     const rocblas_stride strideA,
                                                     float* B,
                                                     const rocblas_int ldb,
                                                     const rocblas_stride strideB,
                                                     float* C,
                                                     const rocblas_int ldc,
                                                     const rocblas_stride strideC,
                                                     rocblas_int* info,
                                                     const rocblas_int batch_count)
{
    return rocsolver_bttrf_npvt_strided_batched_impl<float>(
        handle, nb, nblocks, A, lda, strideA, B, ldb, strideB, C, ldc, strideC, info, batch_count);
}

rocblas_status rocsolver_dbttrf_npvt_strided_batched(rocblas_handle handle,
                                                     const rocblas_int nb,
                                                     const rocblas_int nblocks,
                                                     double* A,
                                                     const rocblas_int lda,
                                                     const rocblas_stride strideA,
                                                     double* B,
                                                     const rocblas_int ldb,
                                                     const rocblas_stride strideB,
                                                     double* C,
                                                     const rocblas_int ldc,
                                                     const rocblas_stride strideC,
                                                     rocblas_int* info,
                                                     const rocblas_int batch_count)
{
    return rocsolver_bttrf_npvt_strided_batched_impl<double>(
        handle, nb, nblocks, A, lda, strideA, B, ldb, strideB, C, ldc, strideC, info, batch_count);
}

rocblas_status rocsolver_cbttrf_npvt_strided_batched(rocblas_handle handle,
                                                     const rocblas_int nb,
                                                     const rocblas_int nblocks,
                                                     rocblas_float_complex* A,
                                                     const rocblas_int lda,
                                                     const rocblas_stride strideA,
                                                     rocblas_float_complex* B,
                                                     const rocblas_int ldb,
                                                     const rocblas_stride strideB,
                                                     rocblas_float_complex* C,
                                                     const rocblas_int ldc,
                                                     const rocblas_stride strideC,
                                                     rocblas_int* info,
                                                     const rocblas_int batch_count)
{
    return rocsolver_bttrf_npvt_strided_batched_impl<rocblas_float_complex>(
        handle, nb, nblocks, A, lda, strideA, B, ldb, strideB, C, ldc, strideC, info, batch_count);
}

rocblas_status rocsolver_zbttrf_npvt_strided_batched(rocblas_handle handle,
                                                     const rocblas_int nb,
                                                     const rocblas_int nblocks,
                                                     rocblas_double_complex* A,
                                                     const rocblas_int lda,
                                                     const rocblas_stride strideA,
                                                     rocblas_double_complex* B,
                                                     const rocblas_int ldb,
                                                     const rocblas_stride strideB,
                                                     rocblas_double_complex* C,
                                                     const rocblas_int ldc,
                                                     const rocblas_stride strideC,
                                                     rocblas_int* info,
                                                     const rocblas_int batch_count)
{
    return rocsolver_bttrf_npvt_strided_batched_impl<rocblas_double_complex>(
        handle, nb, nblocks, A, lda, strideA, B, ldb, strideB, C, ldc, strideC, info, batch_count);
}

} // extern C
