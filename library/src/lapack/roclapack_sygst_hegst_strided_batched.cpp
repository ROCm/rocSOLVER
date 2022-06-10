/* ************************************************************************
 * Copyright (c) 2021-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_sygst_hegst.hpp"

template <typename T, typename U>
rocblas_status rocsolver_sygst_hegst_strided_batched_impl(rocblas_handle handle,
                                                          const rocblas_eform itype,
                                                          const rocblas_fill uplo,
                                                          const rocblas_int n,
                                                          U A,
                                                          const rocblas_int lda,
                                                          const rocblas_stride strideA,
                                                          U B,
                                                          const rocblas_int ldb,
                                                          const rocblas_stride strideB,
                                                          const rocblas_int batch_count)
{
    const char* name = (!rocblas_is_complex<T> ? "sygst_strided_batched" : "hegst_strided_batched");
    ROCSOLVER_ENTER_TOP(name, "--itype", itype, "--uplo", uplo, "-n", n, "--lda", lda, "--strideA",
                        strideA, "--ldb", ldb, "--strideB", strideB, "--batch_count", batch_count);

    using S = decltype(std::real(T{}));

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st
        = rocsolver_sygs2_hegs2_argCheck(handle, itype, uplo, n, lda, ldb, A, B, batch_count);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;
    rocblas_int shiftB = 0;

    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // size of reusable workspace (and for calling SYGS2/HEGS2 and TRSM)
    bool optim_mem;
    size_t size_work_x_temp, size_workArr_temp_arr, size_store_wcs_invA, size_invA_arr;
    rocsolver_sygst_hegst_getMemorySize<false, true, T>(
        uplo, itype, n, batch_count, &size_scalars, &size_work_x_temp, &size_workArr_temp_arr,
        &size_store_wcs_invA, &size_invA_arr, &optim_mem);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_scalars, size_work_x_temp,
                                                      size_workArr_temp_arr, size_store_wcs_invA,
                                                      size_invA_arr);

    // memory workspace allocation
    void *scalars, *work_x_temp, *workArr_temp_arr, *store_wcs_invA, *invA_arr;
    rocblas_device_malloc mem(handle, size_scalars, size_work_x_temp, size_workArr_temp_arr,
                              size_store_wcs_invA, size_invA_arr);

    if(!mem)
        return rocblas_status_memory_error;

    scalars = mem[0];
    work_x_temp = mem[1];
    workArr_temp_arr = mem[2];
    store_wcs_invA = mem[3];
    invA_arr = mem[4];
    if(size_scalars > 0)
        init_scalars(handle, (T*)scalars);

    // execution
    return rocsolver_sygst_hegst_template<false, true, T, S>(
        handle, itype, uplo, n, A, shiftA, lda, strideA, B, shiftB, ldb, strideB, batch_count,
        (T*)scalars, work_x_temp, workArr_temp_arr, store_wcs_invA, invA_arr, optim_mem);
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_ssygst_strided_batched(rocblas_handle handle,
                                                const rocblas_eform itype,
                                                const rocblas_fill uplo,
                                                const rocblas_int n,
                                                float* A,
                                                const rocblas_int lda,
                                                const rocblas_stride strideA,
                                                float* B,
                                                const rocblas_int ldb,
                                                const rocblas_stride strideB,
                                                const rocblas_int batch_count)
{
    return rocsolver_sygst_hegst_strided_batched_impl<float>(handle, itype, uplo, n, A, lda,
                                                             strideA, B, ldb, strideB, batch_count);
}

rocblas_status rocsolver_dsygst_strided_batched(rocblas_handle handle,
                                                const rocblas_eform itype,
                                                const rocblas_fill uplo,
                                                const rocblas_int n,
                                                double* A,
                                                const rocblas_int lda,
                                                const rocblas_stride strideA,
                                                double* B,
                                                const rocblas_int ldb,
                                                const rocblas_stride strideB,
                                                const rocblas_int batch_count)
{
    return rocsolver_sygst_hegst_strided_batched_impl<double>(handle, itype, uplo, n, A, lda,
                                                              strideA, B, ldb, strideB, batch_count);
}

rocblas_status rocsolver_chegst_strided_batched(rocblas_handle handle,
                                                const rocblas_eform itype,
                                                const rocblas_fill uplo,
                                                const rocblas_int n,
                                                rocblas_float_complex* A,
                                                const rocblas_int lda,
                                                const rocblas_stride strideA,
                                                rocblas_float_complex* B,
                                                const rocblas_int ldb,
                                                const rocblas_stride strideB,
                                                const rocblas_int batch_count)
{
    return rocsolver_sygst_hegst_strided_batched_impl<rocblas_float_complex>(
        handle, itype, uplo, n, A, lda, strideA, B, ldb, strideB, batch_count);
}

rocblas_status rocsolver_zhegst_strided_batched(rocblas_handle handle,
                                                const rocblas_eform itype,
                                                const rocblas_fill uplo,
                                                const rocblas_int n,
                                                rocblas_double_complex* A,
                                                const rocblas_int lda,
                                                const rocblas_stride strideA,
                                                rocblas_double_complex* B,
                                                const rocblas_int ldb,
                                                const rocblas_stride strideB,
                                                const rocblas_int batch_count)
{
    return rocsolver_sygst_hegst_strided_batched_impl<rocblas_double_complex>(
        handle, itype, uplo, n, A, lda, strideA, B, ldb, strideB, batch_count);
}

} // extern C
