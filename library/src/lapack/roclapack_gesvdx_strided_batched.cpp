/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_gesvdx.hpp"

template <typename T, typename TT, typename W>
rocblas_status rocsolver_gesvdx_strided_batched_impl(rocblas_handle handle,
                                    const rocblas_svect left_svect,
                                    const rocblas_svect right_svect,
                                    const rocblas_srange srange,
                                    const rocblas_int m,
                                    const rocblas_int n,
                                    W A,
                                    const rocblas_int lda,
                                    const rocblas_stride strideA,
                                    const TT vl,
                                    const TT vu,
                                    const rocblas_int il,
                                    const rocblas_int iu,
                                    rocblas_int* nsv,
                                    TT* S,
                                    const rocblas_stride strideS,
                                    T* U,
                                    const rocblas_int ldu,
                                    const rocblas_stride strideU,
                                    T* V,
                                    const rocblas_int ldv,
                                    const rocblas_stride strideV,
                                    rocblas_int* ifail,
                                    const rocblas_stride strideF,
                                    rocblas_int* info,
                                    const rocblas_int batch_count)
{
    ROCSOLVER_ENTER_TOP("gesvdx_strided_batched", "--left_svect", left_svect, "--right_svect", right_svect, "--srange", srange, "-m", m,
                        "-n", n, "--lda", lda, "--strideA", strideA, "--vl", vl, "--vu", vu, "--il", il, "--iu", iu, "--strideS", strideS, "--ldu", ldu, "--strideU", strideU, 
                        "--ldv", ldv, "--strideV", strideV, "--strideF", strideF, "--batch_count", batch_count);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_gesvdx_argCheck(handle, left_svect, right_svect, srange, m, n, A, lda, 
                                                 vl, vu, il, iu, nsv, S, U, ldu, V, ldv, ifail, info, batch_count);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;

    // memory workspace sizes:
//    size_t size_workArr;
//    rocsolver_gesvdx_getMemorySize<false, T, TT>(&size_workArr);
//
//    if(rocblas_is_device_memory_size_query(handle))
//        return rocblas_set_optimal_device_memory_size(handle, size_workArr);

    // memory workspace allocation
//    void *work_workArr;
//    rocblas_device_malloc mem(handle, size_workArr);

//    if(!mem)
//        return rocblas_status_memory_error;

//    work_workArr = mem[1];
//    if(size_scalars > 0)
//        init_scalars(handle, (T*)scalars);

    // execution
    return rocsolver_gesvdx_template<false, true, T>(
        handle, left_svect, right_svect, srange, m, n, A, shiftA, lda, strideA, 
        vl, vu, il, iu, nsv, S, strideS, U, ldu, strideU,
        V, ldv, strideV, ifail, strideF, info, batch_count);
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_sgesvdx_strided_batched(rocblas_handle handle,
                                const rocblas_svect left_svect,
                                const rocblas_svect right_svect,
                                const rocblas_srange srange,
                                const rocblas_int m,
                                const rocblas_int n,
                                float* A,
                                const rocblas_int lda,
                                const rocblas_stride strideA,
                                const float vl,
                                const float vu,
                                const rocblas_int il,
                                const rocblas_int iu,
                                rocblas_int* nsv,
                                float* S,
                                const rocblas_stride strideS,
                                float* U,
                                const rocblas_int ldu,
                                const rocblas_stride strideU,
                                float* V,
                                const rocblas_int ldv,
                                const rocblas_stride strideV,
                                rocblas_int* ifail,
                                const rocblas_stride strideF,
                                rocblas_int* info,
                                const rocblas_int batch_count)
{
    return rocsolver_gesvdx_strided_batched_impl<float>(handle, left_svect, right_svect, srange, m, n, A, lda, strideA, 
                                        vl, vu, il, iu, nsv, S, strideS, U, ldu, strideU, V, ldv, strideV, ifail, strideF, info, batch_count);
}

rocblas_status rocsolver_dgesvdx_strided_batched(rocblas_handle handle,
                                const rocblas_svect left_svect,
                                const rocblas_svect right_svect,
                                const rocblas_srange srange,
                                const rocblas_int m,
                                const rocblas_int n,
                                double* A,
                                const rocblas_int lda,
                                const rocblas_stride strideA,
                                const double vl,
                                const double vu,
                                const rocblas_int il,
                                const rocblas_int iu,
                                rocblas_int* nsv,
                                double* S,
                                const rocblas_stride strideS,
                                double* U,
                                const rocblas_int ldu,
                                const rocblas_stride strideU,
                                double* V,
                                const rocblas_int ldv,
                                const rocblas_stride strideV,
                                rocblas_int* ifail,
                                const rocblas_stride strideF,
                                rocblas_int* info,
                                const rocblas_int batch_count)
{
    return rocsolver_gesvdx_strided_batched_impl<double>(handle, left_svect, right_svect, srange, m, n, A, lda, strideA,
                                        vl, vu, il, iu, nsv, S, strideS, U, ldu, strideU, V, ldv, strideV, ifail, strideF, info, batch_count);
}

rocblas_status rocsolver_cgesvdx_strided_batched(rocblas_handle handle,
                                const rocblas_svect left_svect,
                                const rocblas_svect right_svect,
                                const rocblas_srange srange,
                                const rocblas_int m,
                                const rocblas_int n,
                                rocblas_float_complex* A,
                                const rocblas_int lda,
                                const rocblas_stride strideA,
                                const float vl,
                                const float vu,
                                const rocblas_int il,
                                const rocblas_int iu,
                                rocblas_int* nsv,
                                float* S,
                                const rocblas_stride strideS,
                                rocblas_float_complex* U,
                                const rocblas_int ldu,
                                const rocblas_stride strideU,
                                rocblas_float_complex* V,
                                const rocblas_int ldv,
                                const rocblas_stride strideV,
                                rocblas_int* ifail,
                                const rocblas_stride strideF,
                                rocblas_int* info,
                                const rocblas_int batch_count)
{
    return rocsolver_gesvdx_strided_batched_impl<rocblas_float_complex>(handle, left_svect, right_svect, srange, m, n, A, lda, strideA,
                                                        vl, vu, il, iu, nsv, S, strideS, U, ldu, strideU, V, ldv, strideV, ifail, strideF, info, batch_count);
}

rocblas_status rocsolver_zgesvdx_strided_batched(rocblas_handle handle,
                                const rocblas_svect left_svect,
                                const rocblas_svect right_svect,
                                const rocblas_srange srange,
                                const rocblas_int m,
                                const rocblas_int n,
                                rocblas_double_complex* A,
                                const rocblas_int lda,
                                const rocblas_stride strideA,
                                const double vl,
                                const double vu,
                                const rocblas_int il,
                                const rocblas_int iu,
                                rocblas_int* nsv,
                                double* S,
                                const rocblas_stride strideS,
                                rocblas_double_complex* U,
                                const rocblas_int ldu,
                                const rocblas_stride strideU,
                                rocblas_double_complex* V,
                                const rocblas_int ldv,
                                const rocblas_stride strideV,
                                rocblas_int* ifail,
                                const rocblas_stride strideF,
                                rocblas_int* info,
                                const rocblas_int batch_count)
{
    return rocsolver_gesvdx_strided_batched_impl<rocblas_double_complex>(handle, left_svect, right_svect, srange, m, n, A, lda, strideA,
                                                        vl, vu, il, iu, nsv, S, strideS, U, ldu, strideU, V, ldv, strideV, ifail, strideF, info, batch_count);
}

} // extern C
