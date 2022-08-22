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
                                     const T abstol,
                                     rocblas_int* nsv,
                                     T* S,
                                     T* Z,
                                     const rocblas_int ldz,
                                     rocblas_int* ifail,
                                     rocblas_int* info)
{
    ROCSOLVER_ENTER_TOP("bdsvdx", "--uplo", uplo, "--svect", svect, "--srange", srange, "-n", n,
                        "--vl", vl, "--vu", vu, "--il", il, "--iu", iu, "--abstol", abstol, "--ldz",
                        ldz);

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
    size_t size_work;
    rocsolver_bdsvdx_getMemorySize<T>(n, batch_count, &size_work);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_work);

    // memory workspace allocation
    void* work;
    rocblas_device_malloc mem(handle, size_work);
    if(!mem)
        return rocblas_status_memory_error;

    work = mem[0];

    // execution
    return rocsolver_bdsvdx_template<T>(handle, uplo, svect, srange, n, D, strideD, E, strideE, vl,
                                        vu, il, iu, abstol, nsv, S, strideS, Z, shiftZ, ldz,
                                        strideZ, ifail, strideIfail, info, batch_count, work);
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
                                 const float abstol,
                                 rocblas_int* nsv,
                                 float* S,
                                 float* Z,
                                 const rocblas_int ldz,
                                 rocblas_int* ifail,
                                 rocblas_int* info)
{
    return rocsolver_bdsvdx_impl<float>(handle, uplo, svect, srange, n, D, E, vl, vu, il, iu,
                                        abstol, nsv, S, Z, ldz, ifail, info);
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
                                 const double abstol,
                                 rocblas_int* nsv,
                                 double* S,
                                 double* Z,
                                 const rocblas_int ldz,
                                 rocblas_int* ifail,
                                 rocblas_int* info)
{
    return rocsolver_bdsvdx_impl<double>(handle, uplo, svect, srange, n, D, E, vl, vu, il, iu,
                                         abstol, nsv, S, Z, ldz, ifail, info);
}

} // extern C
