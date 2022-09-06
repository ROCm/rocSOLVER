/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocauxiliary_bdsvdx.hpp"

template <typename T>
rocblas_status rocsolver_bdsvdx_impl(rocblas_handle handle,
                                     const rocblas_fill uplo,
                                     const rocblas_svect svect,
                                     const rocblas_srange srange,
                                     const rocblas_int n,
                                     T* D,
                                     T* E,
                                     const T vl,
                                     const T vu,
                                     const rocblas_int il,
                                     const rocblas_int iu,
                                     rocblas_int* nsv,
                                     T* S,
                                     T* Z,
                                     const rocblas_int ldz,
                                     rocblas_int* ifail,
                                     rocblas_int* info)
{
    ROCSOLVER_ENTER_TOP("bdsvdx", "--uplo", uplo, "--svect", svect, "--srange", srange, "-n", n,
                        "--vl", vl, "--vu", vu, "--il", il, "--iu", iu, "--ldz", ldz);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_bdsvdx_argCheck(handle, uplo, svect, srange, n, D, E, vl, vu, il,
                                                  iu, nsv, S, Z, ldz, ifail, info);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftZ = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideD = 0;
    rocblas_stride strideE = 0;
    rocblas_stride strideS = 0;
    rocblas_stride strideZ = 0;
    rocblas_stride strideIfail = 0;
    rocblas_int batch_count = 1;

    // memory workspace sizes:
    // size of reusable workspaces (for calling STEBZ and STEIN)
    size_t size_work1_iwork, size_work2_pivmin, size_Esqr, size_bounds, size_inter, size_ninter;
    // size for temporary arrays
    size_t size_nsplit, size_iblock, size_isplit, size_Dtgk, size_Etgk, size_Stmp;
    rocsolver_bdsvdx_getMemorySize<T>(n, batch_count, &size_work1_iwork, &size_work2_pivmin,
                                      &size_Esqr, &size_bounds, &size_inter, &size_ninter,
                                      &size_nsplit, &size_iblock, &size_isplit, &size_Dtgk,
                                      &size_Etgk, &size_Stmp);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(
            handle, size_work1_iwork, size_work2_pivmin, size_Esqr, size_bounds, size_inter,
            size_ninter, size_nsplit, size_iblock, size_isplit, size_Dtgk, size_Etgk, size_Stmp);

    // memory workspace allocation
    void *work1_iwork, *work2_pivmin, *Esqr, *bounds, *inter, *ninter, *nsplit, *iblock, *isplit,
        *Stmp, *Dtgk, *Etgk;
    rocblas_device_malloc mem(handle, size_work1_iwork, size_work2_pivmin, size_Esqr, size_bounds,
                              size_inter, size_ninter, size_nsplit, size_iblock, size_isplit,
                              size_Dtgk, size_Etgk, size_Stmp);
    if(!mem)
        return rocblas_status_memory_error;

    work1_iwork = mem[0];
    work2_pivmin = mem[1];
    Esqr = mem[2];
    bounds = mem[3];
    inter = mem[4];
    ninter = mem[5];
    nsplit = mem[6];
    iblock = mem[7];
    isplit = mem[8];
    Dtgk = mem[9];
    Etgk = mem[10];
    Stmp = mem[11];

    // execution
    return rocsolver_bdsvdx_template<T>(
        handle, uplo, svect, srange, n, D, strideD, E, strideE, vl, vu, il, iu, nsv, S, strideS, Z,
        shiftZ, ldz, strideZ, ifail, strideIfail, info, batch_count, (rocblas_int*)work1_iwork,
        (T*)work2_pivmin, (T*)Esqr, (T*)bounds, (T*)inter, (rocblas_int*)ninter, (rocblas_int*)nsplit,
        (rocblas_int*)iblock, (rocblas_int*)isplit, (T*)Dtgk, (T*)Etgk, (T*)Stmp);
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_sbdsvdx(rocblas_handle handle,
                                 const rocblas_fill uplo,
                                 const rocblas_svect svect,
                                 const rocblas_srange srange,
                                 const rocblas_int n,
                                 float* D,
                                 float* E,
                                 const float vl,
                                 const float vu,
                                 const rocblas_int il,
                                 const rocblas_int iu,
                                 rocblas_int* nsv,
                                 float* S,
                                 float* Z,
                                 const rocblas_int ldz,
                                 rocblas_int* ifail,
                                 rocblas_int* info)
{
    return rocsolver_bdsvdx_impl<float>(handle, uplo, svect, srange, n, D, E, vl, vu, il, iu, nsv,
                                        S, Z, ldz, ifail, info);
}

rocblas_status rocsolver_dbdsvdx(rocblas_handle handle,
                                 const rocblas_fill uplo,
                                 const rocblas_svect svect,
                                 const rocblas_srange srange,
                                 const rocblas_int n,
                                 double* D,
                                 double* E,
                                 const double vl,
                                 const double vu,
                                 const rocblas_int il,
                                 const rocblas_int iu,
                                 rocblas_int* nsv,
                                 double* S,
                                 double* Z,
                                 const rocblas_int ldz,
                                 rocblas_int* ifail,
                                 rocblas_int* info)
{
    return rocsolver_bdsvdx_impl<double>(handle, uplo, svect, srange, n, D, E, vl, vu, il, iu, nsv,
                                         S, Z, ldz, ifail, info);
}

} // extern C
