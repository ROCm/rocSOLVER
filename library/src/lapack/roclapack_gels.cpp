/* ************************************************************************
 * Copyright (c) 2020-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_gels.hpp"

template <typename T, typename U, bool COMPLEX = rocblas_is_complex<T>>
rocblas_status rocsolver_gels_impl(rocblas_handle handle,
                                   rocblas_operation trans,
                                   const rocblas_int m,
                                   const rocblas_int n,
                                   const rocblas_int nrhs,
                                   U A,
                                   const rocblas_int lda,
                                   U B,
                                   const rocblas_int ldb,
                                   rocblas_int* info)
{
    ROCSOLVER_ENTER_TOP("gels", "--trans", trans, "-m", m, "-n", n, "--nrhs", nrhs, "--lda", lda,
                        "--ldb", ldb);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st
        = rocsolver_gels_argCheck<COMPLEX>(handle, trans, m, n, nrhs, A, lda, B, ldb, info);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    const rocblas_int shiftA = 0;
    const rocblas_int shiftB = 0;

    // normal (non-batched non-strided) execution
    const rocblas_stride strideA = 0;
    const rocblas_stride strideB = 0;
    const rocblas_int batch_count = 1;

    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // size of workspace (for calling GEQRF/GELQF, ORMQR/ORMLQ, and TRSM)
    bool optim_mem;
    size_t size_work_x_temp, size_workArr_temp_arr, size_diag_trfac_invA,
        size_trfact_workTrmm_invA_arr;
    // extra requirements for calling ORMQR/ORMLQ and to copy B
    size_t size_ipiv_savedB;
    rocsolver_gels_getMemorySize<false, false, T>(
        trans, m, n, nrhs, batch_count, &size_scalars, &size_work_x_temp, &size_workArr_temp_arr,
        &size_diag_trfac_invA, &size_trfact_workTrmm_invA_arr, &size_ipiv_savedB, &optim_mem);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(
            handle, size_scalars, size_work_x_temp, size_workArr_temp_arr, size_diag_trfac_invA,
            size_trfact_workTrmm_invA_arr, size_ipiv_savedB);

    // memory workspace allocation
    void *scalars, *work_x_temp, *workArr_temp_arr, *diag_trfac_invA, *trfact_workTrmm_invA_arr,
        *ipiv_savedB;
    rocblas_device_malloc mem(handle, size_scalars, size_work_x_temp, size_workArr_temp_arr,
                              size_diag_trfac_invA, size_trfact_workTrmm_invA_arr, size_ipiv_savedB);

    if(!mem)
        return rocblas_status_memory_error;

    scalars = mem[0];
    work_x_temp = mem[1];
    workArr_temp_arr = mem[2];
    diag_trfac_invA = mem[3];
    trfact_workTrmm_invA_arr = mem[4];
    ipiv_savedB = mem[5];
    if(size_scalars > 0)
        init_scalars(handle, (T*)scalars);

    // execution
    return rocsolver_gels_template<false, false, T>(
        handle, trans, m, n, nrhs, A, shiftA, lda, strideA, B, shiftB, ldb, strideB, info,
        batch_count, (T*)scalars, (T*)work_x_temp, (T*)workArr_temp_arr, (T*)diag_trfac_invA,
        (T**)trfact_workTrmm_invA_arr, (T*)ipiv_savedB, optim_mem);
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_sgels(rocblas_handle handle,
                               rocblas_operation trans,
                               const rocblas_int m,
                               const rocblas_int n,
                               const rocblas_int nrhs,
                               float* A,
                               const rocblas_int lda,
                               float* B,
                               const rocblas_int ldb,
                               rocblas_int* info)
{
    return rocsolver_gels_impl<float>(handle, trans, m, n, nrhs, A, lda, B, ldb, info);
}

rocblas_status rocsolver_dgels(rocblas_handle handle,
                               rocblas_operation trans,
                               const rocblas_int m,
                               const rocblas_int n,
                               const rocblas_int nrhs,
                               double* A,
                               const rocblas_int lda,
                               double* B,
                               const rocblas_int ldb,
                               rocblas_int* info)
{
    return rocsolver_gels_impl<double>(handle, trans, m, n, nrhs, A, lda, B, ldb, info);
}

rocblas_status rocsolver_cgels(rocblas_handle handle,
                               rocblas_operation trans,
                               const rocblas_int m,
                               const rocblas_int n,
                               const rocblas_int nrhs,
                               rocblas_float_complex* A,
                               const rocblas_int lda,
                               rocblas_float_complex* B,
                               const rocblas_int ldb,
                               rocblas_int* info)
{
    return rocsolver_gels_impl<rocblas_float_complex>(handle, trans, m, n, nrhs, A, lda, B, ldb,
                                                      info);
}

rocblas_status rocsolver_zgels(rocblas_handle handle,
                               rocblas_operation trans,
                               const rocblas_int m,
                               const rocblas_int n,
                               const rocblas_int nrhs,
                               rocblas_double_complex* A,
                               const rocblas_int lda,
                               rocblas_double_complex* B,
                               const rocblas_int ldb,
                               rocblas_int* info)
{
    return rocsolver_gels_impl<rocblas_double_complex>(handle, trans, m, n, nrhs, A, lda, B, ldb,
                                                       info);
}

} // extern C
