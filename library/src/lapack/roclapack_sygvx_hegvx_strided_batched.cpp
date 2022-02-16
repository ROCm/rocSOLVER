/* ************************************************************************
 * Copyright (c) 2021-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_sygvx_hegvx.hpp"

template <typename T, typename S, typename U>
rocblas_status rocsolver_sygvx_hegvx_strided_batched_impl(rocblas_handle handle,
                                                          const rocblas_eform itype,
                                                          const rocblas_evect evect,
                                                          const rocblas_erange erange,
                                                          const rocblas_fill uplo,
                                                          const rocblas_int n,
                                                          U A,
                                                          const rocblas_int lda,
                                                          const rocblas_stride strideA,
                                                          U B,
                                                          const rocblas_int ldb,
                                                          const rocblas_stride strideB,
                                                          const S vl,
                                                          const S vu,
                                                          const rocblas_int il,
                                                          const rocblas_int iu,
                                                          const S abstol,
                                                          rocblas_int* nev,
                                                          S* W,
                                                          const rocblas_stride strideW,
                                                          U Z,
                                                          const rocblas_int ldz,
                                                          const rocblas_stride strideZ,
                                                          rocblas_int* ifail,
                                                          const rocblas_stride strideF,
                                                          rocblas_int* info,
                                                          const rocblas_int batch_count)
{
    const char* name = (!is_complex<T> ? "sygvx_strided_batched" : "hegvx_strided_batched");
    ROCSOLVER_ENTER_TOP(name, "--itype", itype, "--evect", evect, "--erange", erange, "--uplo",
                        uplo, "-n", n, "--lda", lda, "--strideA", strideA, "--ldb", ldb,
                        "--strideB", strideB, "--vl", vl, "--vu", vu, "--il", il, "--iu", iu,
                        "--abstol", abstol, "--strideW", strideW, "--ldz", ldz, "--strideZ",
                        strideZ, "--strideF", strideF, "--batch_count", batch_count);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st
        = rocsolver_sygvx_hegvx_argCheck(handle, itype, evect, erange, uplo, n, A, lda, B, ldb, vl,
                                         vu, il, iu, nev, W, Z, ldz, ifail, info, batch_count);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;
    rocblas_int shiftB = 0;
    rocblas_int shiftZ = 0;

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
    return rocsolver_sygvx_hegvx_template<false, true, T>(
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

rocblas_status rocsolver_ssygvx_strided_batched(rocblas_handle handle,
                                                const rocblas_eform itype,
                                                const rocblas_evect evect,
                                                const rocblas_erange erange,
                                                const rocblas_fill uplo,
                                                const rocblas_int n,
                                                float* A,
                                                const rocblas_int lda,
                                                const rocblas_stride strideA,
                                                float* B,
                                                const rocblas_int ldb,
                                                const rocblas_stride strideB,
                                                const float vl,
                                                const float vu,
                                                const rocblas_int il,
                                                const rocblas_int iu,
                                                const float abstol,
                                                rocblas_int* nev,
                                                float* W,
                                                const rocblas_stride strideW,
                                                float* Z,
                                                const rocblas_int ldz,
                                                const rocblas_stride strideZ,
                                                rocblas_int* ifail,
                                                const rocblas_stride strideF,
                                                rocblas_int* info,
                                                const rocblas_int batch_count)
{
    return rocsolver_sygvx_hegvx_strided_batched_impl<float>(
        handle, itype, evect, erange, uplo, n, A, lda, strideA, B, ldb, strideB, vl, vu, il, iu,
        abstol, nev, W, strideW, Z, ldz, strideZ, ifail, strideF, info, batch_count);
}

rocblas_status rocsolver_dsygvx_strided_batched(rocblas_handle handle,
                                                const rocblas_eform itype,
                                                const rocblas_evect evect,
                                                const rocblas_erange erange,
                                                const rocblas_fill uplo,
                                                const rocblas_int n,
                                                double* A,
                                                const rocblas_int lda,
                                                const rocblas_stride strideA,
                                                double* B,
                                                const rocblas_int ldb,
                                                const rocblas_stride strideB,
                                                const double vl,
                                                const double vu,
                                                const rocblas_int il,
                                                const rocblas_int iu,
                                                const double abstol,
                                                rocblas_int* nev,
                                                double* W,
                                                const rocblas_stride strideW,
                                                double* Z,
                                                const rocblas_int ldz,
                                                const rocblas_stride strideZ,
                                                rocblas_int* ifail,
                                                const rocblas_stride strideF,
                                                rocblas_int* info,
                                                const rocblas_int batch_count)
{
    return rocsolver_sygvx_hegvx_strided_batched_impl<double>(
        handle, itype, evect, erange, uplo, n, A, lda, strideA, B, ldb, strideB, vl, vu, il, iu,
        abstol, nev, W, strideW, Z, ldz, strideZ, ifail, strideF, info, batch_count);
}

rocblas_status rocsolver_chegvx_strided_batched(rocblas_handle handle,
                                                const rocblas_eform itype,
                                                const rocblas_evect evect,
                                                const rocblas_erange erange,
                                                const rocblas_fill uplo,
                                                const rocblas_int n,
                                                rocblas_float_complex* A,
                                                const rocblas_int lda,
                                                const rocblas_stride strideA,
                                                rocblas_float_complex* B,
                                                const rocblas_int ldb,
                                                const rocblas_stride strideB,
                                                const float vl,
                                                const float vu,
                                                const rocblas_int il,
                                                const rocblas_int iu,
                                                const float abstol,
                                                rocblas_int* nev,
                                                float* W,
                                                const rocblas_stride strideW,
                                                rocblas_float_complex* Z,
                                                const rocblas_int ldz,
                                                const rocblas_stride strideZ,
                                                rocblas_int* ifail,
                                                const rocblas_stride strideF,
                                                rocblas_int* info,
                                                const rocblas_int batch_count)
{
    return rocsolver_sygvx_hegvx_strided_batched_impl<rocblas_float_complex>(
        handle, itype, evect, erange, uplo, n, A, lda, strideA, B, ldb, strideB, vl, vu, il, iu,
        abstol, nev, W, strideW, Z, ldz, strideZ, ifail, strideF, info, batch_count);
}

rocblas_status rocsolver_zhegvx_strided_batched(rocblas_handle handle,
                                                const rocblas_eform itype,
                                                const rocblas_evect evect,
                                                const rocblas_erange erange,
                                                const rocblas_fill uplo,
                                                const rocblas_int n,
                                                rocblas_double_complex* A,
                                                const rocblas_int lda,
                                                const rocblas_stride strideA,
                                                rocblas_double_complex* B,
                                                const rocblas_int ldb,
                                                const rocblas_stride strideB,
                                                const double vl,
                                                const double vu,
                                                const rocblas_int il,
                                                const rocblas_int iu,
                                                const double abstol,
                                                rocblas_int* nev,
                                                double* W,
                                                const rocblas_stride strideW,
                                                rocblas_double_complex* Z,
                                                const rocblas_int ldz,
                                                const rocblas_stride strideZ,
                                                rocblas_int* ifail,
                                                const rocblas_stride strideF,
                                                rocblas_int* info,
                                                const rocblas_int batch_count)
{
    return rocsolver_sygvx_hegvx_strided_batched_impl<rocblas_double_complex>(
        handle, itype, evect, erange, uplo, n, A, lda, strideA, B, ldb, strideB, vl, vu, il, iu,
        abstol, nev, W, strideW, Z, ldz, strideZ, ifail, strideF, info, batch_count);
}

} // extern C
