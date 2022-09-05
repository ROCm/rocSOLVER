/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_gesvdx.hpp"

template <typename T, typename TT, typename W>
rocblas_status rocsolver_gesvdx_impl(rocblas_handle handle,
                                    const rocblas_svect left_svect,
                                    const rocblas_svect right_svect,
                                    const rocblas_srange srange,
                                    const rocblas_int m,
                                    const rocblas_int n,
                                    W A,
                                    const rocblas_int lda,
                                    const TT vl,
                                    const TT vu,
                                    const rocblas_int il,
                                    const rocblas_int iu,
                                    rocblas_int* nsv,
                                    TT* S,
                                    T* U,
                                    const rocblas_int ldu,
                                    T* V,
                                    const rocblas_int ldv,
                                    rocblas_int* ifail,
                                    rocblas_int* info)
{
    ROCSOLVER_ENTER_TOP("gesvdx", "--left_svect", left_svect, "--right_svect", right_svect, "--srange", srange, "-m", m,
                        "-n", n, "--lda", lda, "--vl", vl, "--vu", vu, "--il", il, "--iu", iu, "--ldu", ldu, "--ldv", ldv);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_gesvdx_argCheck(handle, left_svect, right_svect, srange, m, n, A, lda, 
                                                 vl, vu, il, iu, nsv, S, U, ldu, V, ldv, ifail, info);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideA = 0;
    rocblas_stride strideS = 0;
    rocblas_stride strideU = 0;
    rocblas_stride strideV = 0;
    rocblas_stride strideF = 0;
    rocblas_int batch_count = 1;

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
    return rocsolver_gesvdx_template<false, false, T>(
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

rocblas_status rocsolver_sgesvdx(rocblas_handle handle,
                                const rocblas_svect left_svect,
                                const rocblas_svect right_svect,
                                const rocblas_srange srange,
                                const rocblas_int m,
                                const rocblas_int n,
                                float* A,
                                const rocblas_int lda,
                                const float vl,
                                const float vu,
                                const rocblas_int il,
                                const rocblas_int iu,
                                rocblas_int* nsv,
                                float* S,
                                float* U,
                                const rocblas_int ldu,
                                float* V,
                                const rocblas_int ldv,
                                rocblas_int* ifail,
                                rocblas_int* info)
{
    return rocsolver_gesvdx_impl<float>(handle, left_svect, right_svect, srange, m, n, A, lda, 
                                        vl, vu, il, iu, nsv, S, U, ldu, V, ldv, ifail, info);
}

rocblas_status rocsolver_dgesvdx(rocblas_handle handle,
                                const rocblas_svect left_svect,
                                const rocblas_svect right_svect,
                                const rocblas_srange srange,
                                const rocblas_int m,
                                const rocblas_int n,
                                double* A,
                                const rocblas_int lda,
                                const double vl,
                                const double vu,
                                const rocblas_int il,
                                const rocblas_int iu,
                                rocblas_int* nsv,
                                double* S,
                                double* U,
                                const rocblas_int ldu,
                                double* V,
                                const rocblas_int ldv,
                                rocblas_int* ifail,
                                rocblas_int* info)
{
    return rocsolver_gesvdx_impl<double>(handle, left_svect, right_svect, srange, m, n, A, lda, 
                                        vl, vu, il, iu, nsv, S, U, ldu, V, ldv, ifail, info);
}

rocblas_status rocsolver_cgesvdx(rocblas_handle handle,
                                const rocblas_svect left_svect,
                                const rocblas_svect right_svect,
                                const rocblas_srange srange,
                                const rocblas_int m,
                                const rocblas_int n,
                                rocblas_float_complex* A,
                                const rocblas_int lda,
                                const float vl,
                                const float vu,
                                const rocblas_int il,
                                const rocblas_int iu,
                                rocblas_int* nsv,
                                float* S,
                                rocblas_float_complex* U,
                                const rocblas_int ldu,
                                rocblas_float_complex* V,
                                const rocblas_int ldv,
                                rocblas_int* ifail,
                                rocblas_int* info)
{
    return rocsolver_gesvdx_impl<rocblas_float_complex>(handle, left_svect, right_svect, srange, m, n, A, lda,
                                                        vl, vu, il, iu, nsv, S, U, ldu, V, ldv, ifail, info);
}

rocblas_status rocsolver_zgesvdx(rocblas_handle handle,
                                const rocblas_svect left_svect,
                                const rocblas_svect right_svect,
                                const rocblas_srange srange,
                                const rocblas_int m,
                                const rocblas_int n,
                                rocblas_double_complex* A,
                                const rocblas_int lda,
                                const double vl,
                                const double vu,
                                const rocblas_int il,
                                const rocblas_int iu,
                                rocblas_int* nsv,
                                double* S,
                                rocblas_double_complex* U,
                                const rocblas_int ldu,
                                rocblas_double_complex* V,
                                const rocblas_int ldv,
                                rocblas_int* ifail,
                                rocblas_int* info)
{
    return rocsolver_gesvdx_impl<rocblas_double_complex>(handle, left_svect, right_svect, srange, m, n, A, lda,
                                                        vl, vu, il, iu, nsv, S, U, ldu, V, ldv, ifail, info);
}

} // extern C
