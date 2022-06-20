/* ************************************************************************
 * Copyright (c) 2021-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_sygst_hegst.hpp"

template <typename T, typename U>
rocblas_status rocsolver_sygst_hegst_batched_impl(rocblas_handle handle,
                                                  const rocblas_eform itype,
                                                  const rocblas_fill uplo,
                                                  const rocblas_int n,
                                                  U A,
                                                  const rocblas_int lda,
                                                  U B,
                                                  const rocblas_int ldb,
                                                  const rocblas_int batch_count)
{
    const char* name = (!rocblas_is_complex<T> ? "sygst_batched" : "hegst_batched");
    ROCSOLVER_ENTER_TOP(name, "--itype", itype, "--uplo", uplo, "-n", n, "--lda", lda, "--ldb", ldb,
                        "--batch_count", batch_count);

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

    // batched execution
    rocblas_stride strideA = 0;
    rocblas_stride strideB = 0;

    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // size of reusable workspace (and for calling SYGS2/HEGS2 and TRSM)
    bool optim_mem;
    size_t size_work_x_temp, size_workArr_temp_arr, size_store_wcs_invA, size_invA_arr;
    rocsolver_sygst_hegst_getMemorySize<true, false, T>(
        uplo, itype, n, batch_count, &size_scalars, &size_work_x_temp, &size_workArr_temp_arr,
        &size_store_wcs_invA, &size_invA_arr, &optim_mem);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_scalars, size_work_x_temp,
                                                      size_workArr_temp_arr, size_store_wcs_invA,
                                                      size_invA_arr);

    // memory workspace allocation
    void *scalars, *work_x_temp, *workArr_temp_arr, *store_invA, *invA_arr;
    rocblas_device_malloc mem(handle, size_scalars, size_work_x_temp, size_workArr_temp_arr,
                              size_store_wcs_invA, size_invA_arr);

    if(!mem)
        return rocblas_status_memory_error;

    scalars = mem[0];
    work_x_temp = mem[1];
    workArr_temp_arr = mem[2];
    store_invA = mem[3];
    invA_arr = mem[4];
    if(size_scalars > 0)
        init_scalars(handle, (T*)scalars);

    // execution
    return rocsolver_sygst_hegst_template<true, false, T, S>(
        handle, itype, uplo, n, A, shiftA, lda, strideA, B, shiftB, ldb, strideB, batch_count,
        (T*)scalars, work_x_temp, workArr_temp_arr, store_invA, invA_arr, optim_mem);
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_ssygst_batched(rocblas_handle handle,
                                        const rocblas_eform itype,
                                        const rocblas_fill uplo,
                                        const rocblas_int n,
                                        float* const A[],
                                        const rocblas_int lda,
                                        float* const B[],
                                        const rocblas_int ldb,
                                        const rocblas_int batch_count)
{
    return rocsolver_sygst_hegst_batched_impl<float>(handle, itype, uplo, n, A, lda, B, ldb,
                                                     batch_count);
}

rocblas_status rocsolver_dsygst_batched(rocblas_handle handle,
                                        const rocblas_eform itype,
                                        const rocblas_fill uplo,
                                        const rocblas_int n,
                                        double* const A[],
                                        const rocblas_int lda,
                                        double* const B[],
                                        const rocblas_int ldb,
                                        const rocblas_int batch_count)
{
    return rocsolver_sygst_hegst_batched_impl<double>(handle, itype, uplo, n, A, lda, B, ldb,
                                                      batch_count);
}

rocblas_status rocsolver_chegst_batched(rocblas_handle handle,
                                        const rocblas_eform itype,
                                        const rocblas_fill uplo,
                                        const rocblas_int n,
                                        rocblas_float_complex* const A[],
                                        const rocblas_int lda,
                                        rocblas_float_complex* const B[],
                                        const rocblas_int ldb,
                                        const rocblas_int batch_count)
{
    return rocsolver_sygst_hegst_batched_impl<rocblas_float_complex>(handle, itype, uplo, n, A, lda,
                                                                     B, ldb, batch_count);
}

rocblas_status rocsolver_zhegst_batched(rocblas_handle handle,
                                        const rocblas_eform itype,
                                        const rocblas_fill uplo,
                                        const rocblas_int n,
                                        rocblas_double_complex* const A[],
                                        const rocblas_int lda,
                                        rocblas_double_complex* const B[],
                                        const rocblas_int ldb,
                                        const rocblas_int batch_count)
{
    return rocsolver_sygst_hegst_batched_impl<rocblas_double_complex>(handle, itype, uplo, n, A,
                                                                      lda, B, ldb, batch_count);
}

} // extern C
