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
    ROCSOLVER_ENTER_TOP("gesvdx", "--left_svect", left_svect, "--right_svect", right_svect,
                        "--srange", srange, "-m", m, "-n", n, "--lda", lda, "--vl", vl, "--vu", vu,
                        "--il", il, "--iu", iu, "--ldu", ldu, "--ldv", ldv);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st
        = rocsolver_gesvdx_argCheck(handle, left_svect, right_svect, srange, m, n, A, lda, vl, vu,
                                    il, iu, nsv, S, U, ldu, V, ldv, ifail, info);
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
    // reusable workspace (for calls to gebrd, dbsvdx, ormqr, geqrf, etc)
    size_t size_scalars;
    size_t size_WS_svdx1;
    size_t size_WS_svdx2_lqrf1_brd1;
    size_t size_WS_svdx3_lqrf2_brd2;
    size_t size_WS_svdx4_lqrf3_brd3;
    size_t size_WS_svdx5_brd4;
    size_t size_WS_svdx6;
    size_t size_WS_svdx7;
    size_t size_WS_svdx8;
    size_t size_WS_svdx9;
    size_t size_WS_svdx10_mlqr1_mbr1;
    size_t size_WS_svdx11_mlqr2_mbr2;
    size_t size_WS_svdx12_mlqr3_mbr3;
    // temporary arrays for internal computations
    // (contain the bidiagonal form and the householder scalars)
    size_t size_tmpDE;
    size_t size_tauqp;
    size_t size_tmpZ;
    size_t size_tau;
    size_t size_tmpT;
    // size of array of pointers (only for batched case)
    size_t size_workArr;
    size_t size_workArr2;

    rocsolver_gesvdx_getMemorySize<false, T, TT>(
        left_svect, right_svect, srange, m, n, batch_count, &size_scalars, &size_WS_svdx1,
        &size_WS_svdx2_lqrf1_brd1, &size_WS_svdx3_lqrf2_brd2, &size_WS_svdx4_lqrf3_brd3,
        &size_WS_svdx5_brd4, &size_WS_svdx6, &size_WS_svdx7, &size_WS_svdx8, &size_WS_svdx9,
        &size_WS_svdx10_mlqr1_mbr1, &size_WS_svdx11_mlqr2_mbr2, &size_WS_svdx12_mlqr3_mbr3,
        &size_tmpDE, &size_tauqp, &size_tmpZ, &size_tau, &size_tmpT, &size_workArr, &size_workArr2);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(
            handle, size_scalars, size_WS_svdx1, size_WS_svdx2_lqrf1_brd1, size_WS_svdx3_lqrf2_brd2,
            size_WS_svdx4_lqrf3_brd3, size_WS_svdx5_brd4, size_WS_svdx6, size_WS_svdx7,
            size_WS_svdx8, size_WS_svdx9, size_WS_svdx10_mlqr1_mbr1, size_WS_svdx11_mlqr2_mbr2,
            size_WS_svdx12_mlqr3_mbr3, size_tmpDE, size_tauqp, size_tmpZ, size_tau, size_tmpT,
            size_workArr, size_workArr2);

    // memory workspace allocation
    void* scalars;
    void* WS_svdx1;
    void* WS_svdx2_lqrf1_brd1;
    void* WS_svdx3_lqrf2_brd2;
    void* WS_svdx4_lqrf3_brd3;
    void* WS_svdx5_brd4;
    void* WS_svdx6;
    void* WS_svdx7;
    void* WS_svdx8;
    void* WS_svdx9;
    void* WS_svdx10_mlqr1_mbr1;
    void* WS_svdx11_mlqr2_mbr2;
    void* WS_svdx12_mlqr3_mbr3;
    void* tmpDE;
    void* tauqp;
    void* tmpZ;
    void* tau;
    void* tmpT;
    void* workArr;
    void* workArr2;

    rocblas_device_malloc mem(handle, size_scalars, size_WS_svdx1, size_WS_svdx2_lqrf1_brd1,
                              size_WS_svdx3_lqrf2_brd2, size_WS_svdx4_lqrf3_brd3,
                              size_WS_svdx5_brd4, size_WS_svdx6, size_WS_svdx7, size_WS_svdx8,
                              size_WS_svdx9, size_WS_svdx10_mlqr1_mbr1, size_WS_svdx11_mlqr2_mbr2,
                              size_WS_svdx12_mlqr3_mbr3, size_tmpDE, size_tauqp, size_tmpZ,
                              size_tau, size_tmpT, size_workArr, size_workArr2);

    if(!mem)
        return rocblas_status_memory_error;

    scalars = mem[0];
    WS_svdx1 = mem[1];
    WS_svdx2_lqrf1_brd1 = mem[2];
    WS_svdx3_lqrf2_brd2 = mem[3];
    WS_svdx4_lqrf3_brd3 = mem[4];
    WS_svdx5_brd4 = mem[5];
    WS_svdx6 = mem[6];
    WS_svdx7 = mem[7];
    WS_svdx8 = mem[8];
    WS_svdx9 = mem[9];
    WS_svdx10_mlqr1_mbr1 = mem[10];
    WS_svdx11_mlqr2_mbr2 = mem[11];
    WS_svdx12_mlqr3_mbr3 = mem[12];
    tmpDE = mem[13];
    tauqp = mem[14];
    tmpZ = mem[15];
    tau = mem[16];
    tmpT = mem[17];
    workArr = mem[18];
    workArr2 = mem[19];
    if(size_scalars > 0)
        init_scalars(handle, (T*)scalars);

    // execution
    return rocsolver_gesvdx_template<false, false, T>(
        handle, left_svect, right_svect, srange, m, n, A, shiftA, lda, strideA, vl, vu, il, iu, nsv,
        S, strideS, U, ldu, strideU, V, ldv, strideV, ifail, strideF, info, batch_count, (T*)scalars,
        (rocblas_int*)WS_svdx1, WS_svdx2_lqrf1_brd1, WS_svdx3_lqrf2_brd2, WS_svdx4_lqrf3_brd3,
        WS_svdx5_brd4, (rocblas_int*)WS_svdx6, (rocblas_int*)WS_svdx7, (rocblas_int*)WS_svdx8,
        (rocblas_int*)WS_svdx9, WS_svdx10_mlqr1_mbr1, WS_svdx11_mlqr2_mbr2, WS_svdx12_mlqr3_mbr3,
        (TT*)tmpDE, (T*)tauqp, (TT*)tmpZ, (T*)tau, (T*)tmpT, (T**)workArr, (T**)workArr2);
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
    return rocsolver_gesvdx_impl<float>(handle, left_svect, right_svect, srange, m, n, A, lda, vl,
                                        vu, il, iu, nsv, S, U, ldu, V, ldv, ifail, info);
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
    return rocsolver_gesvdx_impl<double>(handle, left_svect, right_svect, srange, m, n, A, lda, vl,
                                         vu, il, iu, nsv, S, U, ldu, V, ldv, ifail, info);
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
    return rocsolver_gesvdx_impl<rocblas_float_complex>(handle, left_svect, right_svect, srange, m,
                                                        n, A, lda, vl, vu, il, iu, nsv, S, U, ldu,
                                                        V, ldv, ifail, info);
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
    return rocsolver_gesvdx_impl<rocblas_double_complex>(handle, left_svect, right_svect, srange, m,
                                                         n, A, lda, vl, vu, il, iu, nsv, S, U, ldu,
                                                         V, ldv, ifail, info);
}

} // extern C
