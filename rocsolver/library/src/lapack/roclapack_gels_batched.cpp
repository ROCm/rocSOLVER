/* ************************************************************************
 * Copyright (c) 2020-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_gels.hpp"

template <typename T, typename U>
rocblas_status rocsolver_gels_batched_impl(rocblas_handle handle,
                                           rocblas_operation trans,
                                           const rocblas_int m,
                                           const rocblas_int n,
                                           const rocblas_int nrhs,
                                           U A,
                                           const rocblas_int lda,
                                           U B,
                                           const rocblas_int ldb,
                                           rocblas_int* info,
                                           const rocblas_int batch_count)
{
    ROCSOLVER_ENTER_TOP("gels_batched", "--transposeA", trans, "-m", m, "-n", n, "-k", nrhs,
                        "--lda", lda, "--ldb:", ldb, "--batch", batch_count);

    if(!handle)
        ROCSOLVER_RETURN_TOP("gels_batched", rocblas_status_invalid_handle);

    // argument checking
    rocblas_status st
        = rocsolver_gels_argCheck(handle, trans, m, n, nrhs, A, lda, B, ldb, info, batch_count);
    if(st != rocblas_status_continue)
        ROCSOLVER_RETURN_TOP("gels_batched", st);

    // working with unshifted arrays
    const rocblas_int shiftA = 0;
    const rocblas_int shiftB = 0;

    // batched execution
    const rocblas_stride strideA = 0;
    const rocblas_stride strideB = 0;

    size_t size_scalars, size_work_x_temp, size_workArr_temp_arr, size_diag_trfac_invA,
        size_trfact_workTrmm_invA_arr, size_ipiv;
    rocsolver_gels_getMemorySize<true, false, T>(
        m, n, nrhs, batch_count, &size_scalars, &size_work_x_temp, &size_workArr_temp_arr,
        &size_diag_trfac_invA, &size_trfact_workTrmm_invA_arr, &size_ipiv);

    if(rocblas_is_device_memory_size_query(handle))
        ROCSOLVER_RETURN_TOP("gels_batched",
                             rocblas_set_optimal_device_memory_size(
                                 handle, size_scalars, size_work_x_temp, size_workArr_temp_arr,
                                 size_diag_trfac_invA, size_trfact_workTrmm_invA_arr, size_ipiv));

    // always allocate all required memory for TRSM optimal performance
    bool optim_mem = true;

    // memory workspace allocation
    void *scalars, *work, *workArr, *diag_trfac_invA, *trfact_workTrmm_invA, *ipiv;
    rocblas_device_malloc mem(handle, size_scalars, size_work_x_temp, size_workArr_temp_arr,
                              size_diag_trfac_invA, size_trfact_workTrmm_invA_arr, size_ipiv);
    if(!mem)
        ROCSOLVER_RETURN_TOP("gels_batched", rocblas_status_memory_error);

    scalars = mem[0];
    work = mem[1];
    workArr = mem[2];
    diag_trfac_invA = mem[3];
    trfact_workTrmm_invA = mem[4];
    ipiv = mem[5];
    if(size_scalars > 0)
        init_scalars(handle, (T*)scalars);

    // execution
    ROCSOLVER_RETURN_TOP("gels_batched",
                         rocsolver_gels_template<true, false, T>(
                             handle, trans, m, n, nrhs, A, shiftA, lda, strideA, B, shiftB, ldb,
                             strideB, info, batch_count, (T*)scalars, (T*)work, (T*)workArr,
                             (T*)diag_trfac_invA, (T**)trfact_workTrmm_invA, (T*)ipiv, optim_mem));
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_sgels_batched(rocblas_handle handle,
                                       rocblas_operation trans,
                                       const rocblas_int m,
                                       const rocblas_int n,
                                       const rocblas_int nrhs,
                                       float* const A[],
                                       const rocblas_int lda,
                                       float* const B[],
                                       const rocblas_int ldb,
                                       rocblas_int* info,
                                       const rocblas_int batch_count)
{
    return rocsolver_gels_batched_impl<float>(handle, trans, m, n, nrhs, A, lda, B, ldb, info,
                                              batch_count);
}

rocblas_status rocsolver_dgels_batched(rocblas_handle handle,
                                       rocblas_operation trans,
                                       const rocblas_int m,
                                       const rocblas_int n,
                                       const rocblas_int nrhs,
                                       double* const A[],
                                       const rocblas_int lda,
                                       double* const B[],
                                       const rocblas_int ldb,
                                       rocblas_int* info,
                                       const rocblas_int batch_count)
{
    return rocsolver_gels_batched_impl<double>(handle, trans, m, n, nrhs, A, lda, B, ldb, info,
                                               batch_count);
}

rocblas_status rocsolver_cgels_batched(rocblas_handle handle,
                                       rocblas_operation trans,
                                       const rocblas_int m,
                                       const rocblas_int n,
                                       const rocblas_int nrhs,
                                       rocblas_float_complex* const A[],
                                       const rocblas_int lda,
                                       rocblas_float_complex* const B[],
                                       const rocblas_int ldb,
                                       rocblas_int* info,
                                       const rocblas_int batch_count)
{
    return rocsolver_gels_batched_impl<rocblas_float_complex>(handle, trans, m, n, nrhs, A, lda, B,
                                                              ldb, info, batch_count);
}

rocblas_status rocsolver_zgels_batched(rocblas_handle handle,
                                       rocblas_operation trans,
                                       const rocblas_int m,
                                       const rocblas_int n,
                                       const rocblas_int nrhs,
                                       rocblas_double_complex* const A[],
                                       const rocblas_int lda,
                                       rocblas_double_complex* const B[],
                                       const rocblas_int ldb,
                                       rocblas_int* info,
                                       const rocblas_int batch_count)
{
    return rocsolver_gels_batched_impl<rocblas_double_complex>(handle, trans, m, n, nrhs, A, lda, B,
                                                               ldb, info, batch_count);
}

} // extern C
