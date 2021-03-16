/* ************************************************************************
 * Copyright (c) 2019-2021 Advanced Micro Devices, Inc.
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
    ROCSOLVER_ENTER_TOP("gesvd", "--left_svect", left_svect, "--right_svect", right_svect, "-m", m,
                        "-n", n, "--lda", lda, "--ldu", ldu, "--ldv", ldv, "--fast_alg", fast_alg);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_gesvd_argCheck(handle, left_svect, right_svect, m, n, A, lda, S,
                                                 U, ldu, V, ldv, E, info);
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
    // extra requirements for calling orthogonal/unitary matrix operations and factorizations
    size_t size_Abyx_norms_tmptr, size_Abyx_norms_trfact_X, size_diag_tmptr_Y;
    // size of array tau to store householder scalars
    size_t size_tau;
    // size of temporary arrays for copies
    size_t size_tempArrayT, size_tempArrayC;
    // size of array of pointers (only for batched case)
    size_t size_workArr;

    rocsolver_gesvd_getMemorySize<false, T, TT>(
        left_svect, right_svect, m, n, batch_count, fast_alg, &size_scalars, &size_work_workArr,
        &size_Abyx_norms_tmptr, &size_Abyx_norms_trfact_X, &size_diag_tmptr_Y, &size_tau,
        &size_tempArrayT, &size_tempArrayC, &size_workArr);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(
            handle, size_scalars, size_work_workArr, size_Abyx_norms_tmptr, size_Abyx_norms_trfact_X,
            size_diag_tmptr_Y, size_tau, size_tempArrayT, size_tempArrayC, size_workArr);

    // memory workspace allocation
    void *scalars, *work_workArr, *Abyx_norms_tmptr, *Abyx_norms_trfact_X, *diag_tmptr_Y, *tau;
    void *tempArrayT, *tempArrayC, *workArr;
    rocblas_device_malloc mem(handle, size_scalars, size_work_workArr, size_Abyx_norms_tmptr,
                              size_Abyx_norms_trfact_X, size_diag_tmptr_Y, size_tau,
                              size_tempArrayT, size_tempArrayC, size_workArr);

    if(!mem)
        return rocblas_status_memory_error;

    scalars = mem[0];
    work_workArr = mem[1];
    Abyx_norms_tmptr = mem[2];
    Abyx_norms_trfact_X = mem[3];
    diag_tmptr_Y = mem[4];
    tau = mem[5];
    tempArrayT = mem[6];
    tempArrayC = mem[7];
    workArr = mem[8];
    if(size_scalars > 0)
        init_scalars(handle, (T*)scalars);

    // execution
    return rocsolver_gesvd_template<false, false, T>(
        handle, left_svect, right_svect, m, n, A, shiftA, lda, strideA, S, strideS, U, ldu, strideU,
        V, ldv, strideV, E, strideE, fast_alg, info, batch_count, (T*)scalars, work_workArr,
        (T*)Abyx_norms_tmptr, (T*)Abyx_norms_trfact_X, (T*)diag_tmptr_Y, (T*)tau, (T*)tempArrayT,
        (T*)tempArrayC, (T**)workArr);
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
