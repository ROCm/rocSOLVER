/* ************************************************************************
 * Copyright (c) 2021-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_sygvx_hegvx.hpp"

template <typename T, typename S, typename U>
rocblas_status rocsolver_sygvx_hegvx_impl(rocblas_handle handle,
                                          const rocblas_eform itype,
                                          const rocblas_evect evect,
                                          const rocblas_erange erange,
                                          const rocblas_fill uplo,
                                          const rocblas_int n,
                                          U A,
                                          const rocblas_int lda,
                                          U B,
                                          const rocblas_int ldb,
                                          const S vl,
                                          const S vu,
                                          const rocblas_int il,
                                          const rocblas_int iu,
                                          const S abstol,
                                          rocblas_int* nev,
                                          S* W,
                                          U Z,
                                          const rocblas_int ldz,
                                          rocblas_int* ifail,
                                          rocblas_int* info)
{
    const char* name = (!is_complex<T> ? "sygvx" : "hegvx");
    ROCSOLVER_ENTER_TOP(name, "--itype", itype, "--evect", evect, "--erange", erange, "--uplo",
                        uplo, "-n", n, "--lda", lda, "--ldb", ldb, "--vl", vl, "--vu", vu, "--il",
                        il, "--iu", iu, "--abstol", abstol, "--ldz", ldz);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st
        = rocsolver_sygvx_hegvx_argCheck(handle, itype, evect, erange, uplo, n, A, lda, B, ldb, vl,
                                         vu, il, iu, nev, W, Z, ldz, ifail, info);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;
    rocblas_int shiftB = 0;
    rocblas_int shiftZ = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideA = 0;
    rocblas_stride strideB = 0;
    rocblas_stride strideW = 0;
    rocblas_stride strideZ = 0;
    rocblas_stride strideF = 0;
    rocblas_int batch_count = 1;

    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // size of array of pointers (only for batched case)
    size_t size_workArr;
    rocsolver_sygvx_hegvx_getMemorySize<false, T, S>(itype, evect, erange, uplo, n, batch_count,
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
    return rocsolver_sygvx_hegvx_template<false, false, T>(
        handle, itype, evect, erange, uplo, n, A, shiftA, lda, strideA, B, shiftB, ldb, strideB, vl,
        vu, il, iu, abstol, nev, W, strideW, Z, shiftZ, ldz, strideZ, ifail, strideF, info,
        batch_count, (T*)scalars, (T**)workArr);
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_ssygvx(rocblas_handle handle,
                                const rocblas_eform itype,
                                const rocblas_evect evect,
                                const rocblas_erange erange,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                float* A,
                                const rocblas_int lda,
                                float* B,
                                const rocblas_int ldb,
                                const float vl,
                                const float vu,
                                const rocblas_int il,
                                const rocblas_int iu,
                                const float abstol,
                                rocblas_int* nev,
                                float* W,
                                float* Z,
                                const rocblas_int ldz,
                                rocblas_int* ifail,
                                rocblas_int* info)
{
    return rocsolver_sygvx_hegvx_impl<float>(handle, itype, evect, erange, uplo, n, A, lda, B, ldb,
                                             vl, vu, il, iu, abstol, nev, W, Z, ldz, ifail, info);
}

rocblas_status rocsolver_dsygvx(rocblas_handle handle,
                                const rocblas_eform itype,
                                const rocblas_evect evect,
                                const rocblas_erange erange,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                double* A,
                                const rocblas_int lda,
                                double* B,
                                const rocblas_int ldb,
                                const double vl,
                                const double vu,
                                const rocblas_int il,
                                const rocblas_int iu,
                                const double abstol,
                                rocblas_int* nev,
                                double* W,
                                double* Z,
                                const rocblas_int ldz,
                                rocblas_int* ifail,
                                rocblas_int* info)
{
    return rocsolver_sygvx_hegvx_impl<double>(handle, itype, evect, erange, uplo, n, A, lda, B, ldb,
                                              vl, vu, il, iu, abstol, nev, W, Z, ldz, ifail, info);
}

rocblas_status rocsolver_chegvx(rocblas_handle handle,
                                const rocblas_eform itype,
                                const rocblas_evect evect,
                                const rocblas_erange erange,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                rocblas_float_complex* A,
                                const rocblas_int lda,
                                rocblas_float_complex* B,
                                const rocblas_int ldb,
                                const float vl,
                                const float vu,
                                const rocblas_int il,
                                const rocblas_int iu,
                                const float abstol,
                                rocblas_int* nev,
                                float* W,
                                rocblas_float_complex* Z,
                                const rocblas_int ldz,
                                rocblas_int* ifail,
                                rocblas_int* info)
{
    return rocsolver_sygvx_hegvx_impl<rocblas_float_complex>(handle, itype, evect, erange, uplo, n,
                                                             A, lda, B, ldb, vl, vu, il, iu, abstol,
                                                             nev, W, Z, ldz, ifail, info);
}

rocblas_status rocsolver_zhegvx(rocblas_handle handle,
                                const rocblas_eform itype,
                                const rocblas_evect evect,
                                const rocblas_erange erange,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                rocblas_double_complex* A,
                                const rocblas_int lda,
                                rocblas_double_complex* B,
                                const rocblas_int ldb,
                                const double vl,
                                const double vu,
                                const rocblas_int il,
                                const rocblas_int iu,
                                const double abstol,
                                rocblas_int* nev,
                                double* W,
                                rocblas_double_complex* Z,
                                const rocblas_int ldz,
                                rocblas_int* ifail,
                                rocblas_int* info)
{
    return rocsolver_sygvx_hegvx_impl<rocblas_double_complex>(handle, itype, evect, erange, uplo, n,
                                                              A, lda, B, ldb, vl, vu, il, iu,
                                                              abstol, nev, W, Z, ldz, ifail, info);
}

} // extern C
