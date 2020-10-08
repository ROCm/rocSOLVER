/* ************************************************************************
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_gesvd.hpp"

template <typename T, typename TT, typename W>
rocblas_status rocsolver_gesvd_impl(rocblas_handle handle,
                                    const rocblas_svect left_svect,
                                    const rocblas_svect right_svect,
                                    const rocblas_int m,
                                    const rocblas_int n,
                                    W A,
                                    const rocblas_int lda,
                                    TT* S,
                                    T* U,
                                    const rocblas_int ldu,
                                    T* V,
                                    const rocblas_int ldv,
                                    TT* E,
                                    const rocblas_workmode fast_alg,
                                    rocblas_int* info)
{
    if(!handle)
        return rocblas_status_invalid_handle;

    // logging is missing ???

    // argument checking
    rocblas_status st = rocsolver_gesvd_argCheck(left_svect, right_svect, m, n, A, lda, S, U, ldu,
                                                 V, ldv, E, info);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideA = 0;
    rocblas_stride strideS = 0;
    rocblas_stride strideU = 0;
    rocblas_stride strideV = 0;
    rocblas_stride strideE = 0;
    rocblas_int batch_count = 1;

    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // size of reusable workspace and array of pointers (batched case)
    size_t size_work_workArr;
    // size of array of pointers (only for batched case)
    size_t size_workArr;
    // extra requirements for calling GEBRD and ORGBR
    size_t size_Abyx_norms_tmptr, size_X_trfact, size_Y;
    //size of array tau to store householder scalars
    size_t size_tau;
    rocsolver_gesvd_getMemorySize<false, T, TT>(
        left_svect, right_svect, m, n, batch_count, &size_scalars, &size_work_workArr,
        &size_Abyx_norms_tmptr, &size_X_trfact, &size_Y, &size_tau, &size_workArr);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_scalars, size_work_workArr,
                                                      size_Abyx_norms_tmptr, size_X_trfact, size_Y,
                                                      size_tau, size_workArr);

    // memory workspace allocation
    void *scalars, *work_workArr, *Abyx_norms_tmptr, *X_trfact, *Y, *tau, *workArr;
    rocblas_device_malloc mem(handle, size_scalars, size_work_workArr, size_Abyx_norms_tmptr,
                              size_X_trfact, size_Y, size_tau, size_workArr);

    if(!mem)
        return rocblas_status_memory_error;

    scalars = mem[0];
    work_workArr = mem[1];
    Abyx_norms_tmptr = mem[2];
    X_trfact = mem[3];
    Y = mem[4];
    tau = mem[5];
    workArr = mem[6];
    T sca[] = {-1, 0, 1};
    RETURN_IF_HIP_ERROR(hipMemcpy((T*)scalars, sca, size_scalars, hipMemcpyHostToDevice));

    // execution
    return rocsolver_gesvd_template<false, false, T>(
        handle, left_svect, right_svect, m, n, A, shiftA, lda, strideA, S, strideS, U, ldu, strideU,
        V, ldv, strideV, E, strideE, fast_alg, info, batch_count, (T*)scalars, work_workArr,
        (T*)Abyx_norms_tmptr, (T*)X_trfact, (T*)Y, (T*)tau, (T**)workArr);
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_sgesvd(rocblas_handle handle,
                                const rocblas_svect left_svect,
                                const rocblas_svect right_svect,
                                const rocblas_int m,
                                const rocblas_int n,
                                float* A,
                                const rocblas_int lda,
                                float* S,
                                float* U,
                                const rocblas_int ldu,
                                float* V,
                                const rocblas_int ldv,
                                float* E,
                                const rocblas_workmode fast_alg,
                                rocblas_int* info)
{
    return rocsolver_gesvd_impl<float>(handle, left_svect, right_svect, m, n, A, lda, S, U, ldu, V,
                                       ldv, E, fast_alg, info);
}

rocblas_status rocsolver_dgesvd(rocblas_handle handle,
                                const rocblas_svect left_svect,
                                const rocblas_svect right_svect,
                                const rocblas_int m,
                                const rocblas_int n,
                                double* A,
                                const rocblas_int lda,
                                double* S,
                                double* U,
                                const rocblas_int ldu,
                                double* V,
                                const rocblas_int ldv,
                                double* E,
                                const rocblas_workmode fast_alg,
                                rocblas_int* info)
{
    return rocsolver_gesvd_impl<double>(handle, left_svect, right_svect, m, n, A, lda, S, U, ldu, V,
                                        ldv, E, fast_alg, info);
}

rocblas_status rocsolver_cgesvd(rocblas_handle handle,
                                const rocblas_svect left_svect,
                                const rocblas_svect right_svect,
                                const rocblas_int m,
                                const rocblas_int n,
                                rocblas_float_complex* A,
                                const rocblas_int lda,
                                float* S,
                                rocblas_float_complex* U,
                                const rocblas_int ldu,
                                rocblas_float_complex* V,
                                const rocblas_int ldv,
                                float* E,
                                const rocblas_workmode fast_alg,
                                rocblas_int* info)
{
    return rocsolver_gesvd_impl<rocblas_float_complex>(handle, left_svect, right_svect, m, n, A,
                                                       lda, S, U, ldu, V, ldv, E, fast_alg, info);
}

rocblas_status rocsolver_zgesvd(rocblas_handle handle,
                                const rocblas_svect left_svect,
                                const rocblas_svect right_svect,
                                const rocblas_int m,
                                const rocblas_int n,
                                rocblas_double_complex* A,
                                const rocblas_int lda,
                                double* S,
                                rocblas_double_complex* U,
                                const rocblas_int ldu,
                                rocblas_double_complex* V,
                                const rocblas_int ldv,
                                double* E,
                                const rocblas_workmode fast_alg,
                                rocblas_int* info)
{
    return rocsolver_gesvd_impl<rocblas_double_complex>(handle, left_svect, right_svect, m, n, A,
                                                        lda, S, U, ldu, V, ldv, E, fast_alg, info);
}

} // extern C
