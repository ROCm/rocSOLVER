/* ************************************************************************
 * Copyright (c) 2021-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_bttrf_npvt_interleaved.hpp"

template <typename T, typename U>
rocblas_status rocsolver_bttrf_npvt_interleaved_batched_impl(rocblas_handle handle,
                                                             const rocblas_int nb,
                                                             const rocblas_int nblocks,
                                                             U A,
                                                             const rocblas_int lda,
                                                             U B,
                                                             const rocblas_int ldb,
                                                             U C,
                                                             const rocblas_int ldc,
                                                             rocblas_int* info,
                                                             const rocblas_int batch_count)
{
    ROCSOLVER_ENTER_TOP("bttrf_npvt_interleaved_batched", "--nb", nb, "--nblocks", nblocks, "--lda",
                        lda, "--ldb", ldb, "--ldc", ldc, "--batch_count", batch_count);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_bttrf_npvt_interleaved_argCheck(handle, nb, nblocks, lda, ldb,
                                                                  ldc, A, B, C, info, batch_count);
    if(st != rocblas_status_continue)
        return st;

    // memory workspace sizes:
    // size of reusable workspace
    size_t size_work;

    rocsolver_bttrf_npvt_interleaved_getMemorySize<T>(nb, nblocks, batch_count, &size_work);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_work);

    // memory workspace allocation
    void* work;
    rocblas_device_malloc mem(handle, size_work);

    if(!mem)
        return rocblas_status_memory_error;
    work = mem[0];

    // Execution
    return rocsolver_bttrf_npvt_interleaved_template<T>(handle, nb, nblocks, A, lda, B, ldb, C, ldc,
                                                        info, batch_count, work);
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_sbttrf_npvt_interleaved_batched(rocblas_handle handle,
                                                         const rocblas_int nb,
                                                         const rocblas_int nblocks,
                                                         float* A,
                                                         const rocblas_int lda,
                                                         float* B,
                                                         const rocblas_int ldb,
                                                         float* C,
                                                         const rocblas_int ldc,
                                                         rocblas_int* info,
                                                         const rocblas_int batch_count)
{
    return rocsolver_bttrf_npvt_interleaved_batched_impl<float>(handle, nb, nblocks, A, lda, B, ldb,
                                                                C, ldc, info, batch_count);
}

rocblas_status rocsolver_dbttrf_npvt_interleaved_batched(rocblas_handle handle,
                                                         const rocblas_int nb,
                                                         const rocblas_int nblocks,
                                                         double* A,
                                                         const rocblas_int lda,
                                                         double* B,
                                                         const rocblas_int ldb,
                                                         double* C,
                                                         const rocblas_int ldc,
                                                         rocblas_int* info,
                                                         const rocblas_int batch_count)
{
    return rocsolver_bttrf_npvt_interleaved_batched_impl<double>(handle, nb, nblocks, A, lda, B,
                                                                 ldb, C, ldc, info, batch_count);
}

rocblas_status rocsolver_cbttrf_npvt_interleaved_batched(rocblas_handle handle,
                                                         const rocblas_int nb,
                                                         const rocblas_int nblocks,
                                                         rocblas_float_complex* A,
                                                         const rocblas_int lda,
                                                         rocblas_float_complex* B,
                                                         const rocblas_int ldb,
                                                         rocblas_float_complex* C,
                                                         const rocblas_int ldc,
                                                         rocblas_int* info,
                                                         const rocblas_int batch_count)
{
    return rocsolver_bttrf_npvt_interleaved_batched_impl<rocblas_float_complex>(
        handle, nb, nblocks, A, lda, B, ldb, C, ldc, info, batch_count);
}

rocblas_status rocsolver_zbttrf_npvt_interleaved_batched(rocblas_handle handle,
                                                         const rocblas_int nb,
                                                         const rocblas_int nblocks,
                                                         rocblas_double_complex* A,
                                                         const rocblas_int lda,
                                                         rocblas_double_complex* B,
                                                         const rocblas_int ldb,
                                                         rocblas_double_complex* C,
                                                         const rocblas_int ldc,
                                                         rocblas_int* info,
                                                         const rocblas_int batch_count)
{
    return rocsolver_bttrf_npvt_interleaved_batched_impl<rocblas_double_complex>(
        handle, nb, nblocks, A, lda, B, ldb, C, ldc, info, batch_count);
}

} // extern C
