/* ************************************************************************
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_gels.hpp"

template <typename T, typename U>
rocblas_status rocsolver_gels_strided_batched_impl(rocblas_handle handle,
                                                   rocblas_operation trans,
                                                   const rocblas_int m,
                                                   const rocblas_int n,
                                                   const rocblas_int nrhs,
                                                   U A,
                                                   const rocblas_int lda,
                                                   const rocblas_stride strideA,
                                                   U C,
                                                   const rocblas_int ldc,
                                                   const rocblas_stride strideC,
                                                   rocblas_int* info,
                                                   const rocblas_int batch_count)
{
    if(!handle)
        return rocblas_status_invalid_handle;

    // logging is missing ???

    // argument checking
    rocblas_status st = rocsolver_gels_argCheck(trans, m, n, nrhs, A, lda, C, ldc, info, batch_count);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    const rocblas_int shiftA = 0;
    const rocblas_int shiftC = 0;

    size_t size_scalars, size_work_x_temp, size_workArr_temp_arr, size_diag_trfac_invA,
        size_trfact_workTrmm_invA_arr, size_ipiv;
    rocsolver_gels_getMemorySize<false, true, T>(
        m, n, nrhs, batch_count, &size_scalars, &size_work_x_temp, &size_workArr_temp_arr,
        &size_diag_trfac_invA, &size_trfact_workTrmm_invA_arr, &size_ipiv);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_scalars, size_work_x_temp,
                                                      size_workArr_temp_arr, size_diag_trfac_invA,
                                                      size_trfact_workTrmm_invA_arr, size_ipiv);

    // always allocate all required memory for TRSM optimal performance
    bool optim_mem = true;

    // memory workspace allocation
    void *scalars, *work_x_temp, *workArr_temp_arr, *diag_trfac_invA, *trfact_workTrmm_invA_arr,
        *ipiv;
    rocblas_device_malloc mem(handle, size_scalars, size_work_x_temp, size_workArr_temp_arr,
                              size_diag_trfac_invA, size_trfact_workTrmm_invA_arr, size_ipiv);

    if(!mem)
        return rocblas_status_memory_error;

    scalars = mem[0];
    work_x_temp = mem[1];
    workArr_temp_arr = mem[2];
    diag_trfac_invA = mem[3];
    trfact_workTrmm_invA_arr = mem[4];
    ipiv = mem[5];
    T sca[] = {-1, 0, 1};
    RETURN_IF_HIP_ERROR(hipMemcpy(scalars, sca, size_scalars, hipMemcpyHostToDevice));

    return rocsolver_gels_template<false, true, T>(
        handle, trans, m, n, nrhs, A, shiftA, lda, strideA, C, shiftC, ldc, strideC, info,
        batch_count, (T*)scalars, (T*)work_x_temp, (T*)workArr_temp_arr, (T*)diag_trfac_invA,
        (T**)trfact_workTrmm_invA_arr, (T*)ipiv, optim_mem);
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_sgels_strided_batched(rocblas_handle handle,
                                               rocblas_operation trans,
                                               const rocblas_int m,
                                               const rocblas_int n,
                                               const rocblas_int nrhs,
                                               float* A,
                                               const rocblas_int lda,
                                               const rocblas_stride strideA,
                                               float* C,
                                               const rocblas_int ldc,
                                               const rocblas_stride strideC,
                                               rocblas_int* info,
                                               const rocblas_int batch_count)
{
    return rocsolver_gels_strided_batched_impl<float>(handle, trans, m, n, nrhs, A, lda, strideA, C,
                                                      ldc, strideC, info, batch_count);
}

rocblas_status rocsolver_dgels_strided_batched(rocblas_handle handle,
                                               rocblas_operation trans,
                                               const rocblas_int m,
                                               const rocblas_int n,
                                               const rocblas_int nrhs,
                                               double* A,
                                               const rocblas_int lda,
                                               const rocblas_stride strideA,
                                               double* C,
                                               const rocblas_int ldc,
                                               const rocblas_stride strideC,
                                               rocblas_int* info,
                                               const rocblas_int batch_count)
{
    return rocsolver_gels_strided_batched_impl<double>(handle, trans, m, n, nrhs, A, lda, strideA,
                                                       C, ldc, strideC, info, batch_count);
}

rocblas_status rocsolver_cgels_strided_batched(rocblas_handle handle,
                                               rocblas_operation trans,
                                               const rocblas_int m,
                                               const rocblas_int n,
                                               const rocblas_int nrhs,
                                               rocblas_float_complex* A,
                                               const rocblas_int lda,
                                               const rocblas_stride strideA,
                                               rocblas_float_complex* C,
                                               const rocblas_int ldc,
                                               const rocblas_stride strideC,
                                               rocblas_int* info,
                                               const rocblas_int batch_count)
{
    return rocsolver_gels_strided_batched_impl<rocblas_float_complex>(
        handle, trans, m, n, nrhs, A, lda, strideA, C, ldc, strideC, info, batch_count);
}

rocblas_status rocsolver_zgels_strided_batched(rocblas_handle handle,
                                               rocblas_operation trans,
                                               const rocblas_int m,
                                               const rocblas_int n,
                                               const rocblas_int nrhs,
                                               rocblas_double_complex* A,
                                               const rocblas_int lda,
                                               const rocblas_stride strideA,
                                               rocblas_double_complex* C,
                                               const rocblas_int ldc,
                                               const rocblas_stride strideC,
                                               rocblas_int* info,
                                               const rocblas_int batch_count)
{
    return rocsolver_gels_strided_batched_impl<rocblas_double_complex>(
        handle, trans, m, n, nrhs, A, lda, strideA, C, ldc, strideC, info, batch_count);
}

} // extern C
