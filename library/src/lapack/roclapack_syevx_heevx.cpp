/* ************************************************************************
 * Copyright (c) 2021-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_syevx_heevx.hpp"

template <typename T, typename S, typename U>
rocblas_status rocsolver_syevx_heevx_impl(rocblas_handle handle,
                                          const rocblas_evect evect,
                                          const rocblas_erange erange,
                                          const rocblas_fill uplo,
                                          const rocblas_int n,
                                          U A,
                                          const rocblas_int lda,
                                          const S vl,
                                          const S vu,
                                          const rocblas_int il,
                                          const rocblas_int iu,
                                          rocblas_int* nev,
                                          S* W,
                                          U Z,
                                          const rocblas_int ldz,
                                          rocblas_int* ifail,
                                          rocblas_int* info)
{
    const char* name = (!is_complex<T> ? "syevx" : "heevx");
    ROCSOLVER_ENTER_TOP(name, "--evect", evect, "--erange", erange, "--uplo", uplo, "-n", n,
                        "--lda", lda, "--vl", vl, "--vu", vu, "--il", il, "--iu", iu, "--ldz", ldz);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_syevx_heevx_argCheck(handle, evect, erange, uplo, n, A, lda, vl,
                                                       vu, il, iu, nev, W, Z, ldz, ifail, info);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;
    rocblas_int shiftZ = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideA = 0;
    rocblas_stride strideW = 0;
    rocblas_stride strideZ = 0;
    rocblas_stride strideF = 0;
    rocblas_int batch_count = 1;

    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // size of array of pointers (only for batched case)
    size_t size_workArr;

    rocsolver_syevx_heevx_getMemorySize<false, T, S>(evect, erange, uplo, n, batch_count,
                                                     &size_scalars, &size_workArr);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_scalars, size_workArr);

    // memory workspace allocation
    void *scalars, *workArr;
    rocblas_device_malloc mem(handle, size_scalars, size_workArr);

    if(!mem)
        return rocblas_status_memory_error;

    scalars = mem[0];
    workArr = mem[1];
    if(size_scalars > 0)
        init_scalars(handle, (T*)scalars);

    // execution
    return rocsolver_syevx_heevx_template<false, false, T>(
        handle, evect, erange, uplo, n, A, shiftA, lda, strideA, vl, vu, il, iu, nev, W, strideW, Z,
        shiftZ, ldz, strideZ, ifail, strideF, info, batch_count, (T*)scalars, (T**)workArr);
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_ssyevx(rocblas_handle handle,
                                const rocblas_evect evect,
                                const rocblas_erange erange,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                float* A,
                                const rocblas_int lda,
                                const float vl,
                                const float vu,
                                const rocblas_int il,
                                const rocblas_int iu,
                                rocblas_int* nev,
                                float* W,
                                float* Z,
                                const rocblas_int ldz,
                                rocblas_int* ifail,
                                rocblas_int* info)
{
    return rocsolver_syevx_heevx_impl<float>(handle, evect, erange, uplo, n, A, lda, vl, vu, il, iu,
                                             nev, W, Z, ldz, ifail, info);
}

rocblas_status rocsolver_dsyevx(rocblas_handle handle,
                                const rocblas_evect evect,
                                const rocblas_erange erange,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                double* A,
                                const rocblas_int lda,
                                const double vl,
                                const double vu,
                                const rocblas_int il,
                                const rocblas_int iu,
                                rocblas_int* nev,
                                double* W,
                                double* Z,
                                const rocblas_int ldz,
                                rocblas_int* ifail,
                                rocblas_int* info)
{
    return rocsolver_syevx_heevx_impl<double>(handle, evect, erange, uplo, n, A, lda, vl, vu, il,
                                              iu, nev, W, Z, ldz, ifail, info);
}

rocblas_status rocsolver_cheevx(rocblas_handle handle,
                                const rocblas_evect evect,
                                const rocblas_erange erange,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                rocblas_float_complex* A,
                                const rocblas_int lda,
                                const float vl,
                                const float vu,
                                const rocblas_int il,
                                const rocblas_int iu,
                                rocblas_int* nev,
                                float* W,
                                rocblas_float_complex* Z,
                                const rocblas_int ldz,
                                rocblas_int* ifail,
                                rocblas_int* info)
{
    return rocsolver_syevx_heevx_impl<rocblas_float_complex>(
        handle, evect, erange, uplo, n, A, lda, vl, vu, il, iu, nev, W, Z, ldz, ifail, info);
}

rocblas_status rocsolver_zheevx(rocblas_handle handle,
                                const rocblas_evect evect,
                                const rocblas_erange erange,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                rocblas_double_complex* A,
                                const rocblas_int lda,
                                const double vl,
                                const double vu,
                                const rocblas_int il,
                                const rocblas_int iu,
                                rocblas_int* nev,
                                double* W,
                                rocblas_double_complex* Z,
                                const rocblas_int ldz,
                                rocblas_int* ifail,
                                rocblas_int* info)
{
    return rocsolver_syevx_heevx_impl<rocblas_double_complex>(
        handle, evect, erange, uplo, n, A, lda, vl, vu, il, iu, nev, W, Z, ldz, ifail, info);
}

} // extern C
