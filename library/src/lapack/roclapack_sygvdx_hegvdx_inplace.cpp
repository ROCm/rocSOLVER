/* ************************************************************************
 * Copyright (c) 2021-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_sygvdx_hegvdx_inplace.hpp"

/*
 * ===========================================================================
 *    sygvdx/hegvdx_inplace is not intended for inclusion in the public API. It
 *    exists to provide a sygvdx/hegvdx method with a signature identical to
 *    the cuSOLVER implementation, for use exclusively in hipSOLVER.
 * ===========================================================================
 */

template <typename T, typename S, typename U>
rocblas_status rocsolver_sygvdx_hegvdx_inplace_impl(rocblas_handle handle,
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
                                                    rocblas_int* h_nev,
                                                    S* W,
                                                    rocblas_int* info)
{
    const char* name = (!rocblas_is_complex<T> ? "sygvdx_inplace" : "hegvdx_inplace");
    ROCSOLVER_ENTER_TOP(name, "--itype", itype, "--evect", evect, "--erange", erange, "--uplo",
                        uplo, "-n", n, "--lda", lda, "--ldb", ldb, "--vl", vl, "--vu", vu, "--il",
                        il, "--iu", iu, "--abstol", abstol);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_sygvdx_hegvdx_inplace_argCheck(
        handle, itype, evect, erange, uplo, n, A, lda, B, ldb, vl, vu, il, iu, h_nev, W, info);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;
    rocblas_int shiftB = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideA = 0;
    rocblas_stride strideB = 0;
    rocblas_stride strideW = 0;
    rocblas_int batch_count = 1;

    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // size of reusable workspaces (and for calling TRSM, SYGST/HEGST, and SYEVDX/HEEVDX)
    bool optim_mem;
    size_t size_work1, size_work2, size_work3, size_work4, size_work5, size_work6;
    // extra requirements for calling SYEVDX/HEEVDX_INPLACE
    size_t size_D, size_E, size_iblock, size_isplit, size_tau, size_nev;
    // extra requirements for calling POTRF and SYEVDX/HEEVDX_INPLACE
    size_t size_work7_workArr;
    // size of temporary info array
    size_t size_iinfo;
    rocsolver_sygvdx_hegvdx_inplace_getMemorySize<false, false, T, S>(
        itype, evect, uplo, n, batch_count, &size_scalars, &size_work1, &size_work2, &size_work3,
        &size_work4, &size_work5, &size_work6, &size_D, &size_E, &size_iblock, &size_isplit,
        &size_tau, &size_nev, &size_work7_workArr, &size_iinfo, &optim_mem);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(
            handle, size_scalars, size_work1, size_work2, size_work3, size_work4, size_work5,
            size_work6, size_D, size_E, size_iblock, size_isplit, size_tau, size_nev,
            size_work7_workArr, size_iinfo);

    // memory workspace allocation
    void *scalars, *work1, *work2, *work3, *work4, *work5, *work6, *D, *E, *iblock, *isplit, *tau,
        *d_nev, *work7_workArr, *iinfo;
    rocblas_device_malloc mem(handle, size_scalars, size_work1, size_work2, size_work3, size_work4,
                              size_work5, size_work6, size_D, size_E, size_iblock, size_isplit,
                              size_tau, size_nev, size_work7_workArr, size_iinfo);

    if(!mem)
        return rocblas_status_memory_error;

    scalars = mem[0];
    work1 = mem[1];
    work2 = mem[2];
    work3 = mem[3];
    work4 = mem[4];
    work5 = mem[5], work6 = mem[6], D = mem[7];
    E = mem[8];
    iblock = mem[9];
    isplit = mem[10];
    tau = mem[11];
    d_nev = mem[12];
    work7_workArr = mem[13];
    iinfo = mem[14];
    if(size_scalars > 0)
        init_scalars(handle, (T*)scalars);

    // execution
    return rocsolver_sygvdx_hegvdx_inplace_template<false, false, T>(
        handle, itype, evect, erange, uplo, n, A, shiftA, lda, strideA, B, shiftB, ldb, strideB, vl,
        vu, il, iu, abstol, h_nev, W, strideW, info, batch_count, (T*)scalars, work1, work2, work3,
        work4, work5, work6, (S*)D, (S*)E, (rocblas_int*)iblock, (rocblas_int*)isplit, (T*)tau,
        (rocblas_int*)d_nev, work7_workArr, (rocblas_int*)iinfo, optim_mem);
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

ROCSOLVER_EXPORT rocblas_status rocsolver_ssygvdx_inplace(rocblas_handle handle,
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
                                                          rocblas_int* h_nev,
                                                          float* W,
                                                          rocblas_int* info)
{
    return rocsolver_sygvdx_hegvdx_inplace_impl<float>(handle, itype, evect, erange, uplo, n, A,
                                                       lda, B, ldb, vl, vu, il, iu, abstol, h_nev,
                                                       W, info);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_dsygvdx_inplace(rocblas_handle handle,
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
                                                          rocblas_int* h_nev,
                                                          double* W,
                                                          rocblas_int* info)
{
    return rocsolver_sygvdx_hegvdx_inplace_impl<double>(handle, itype, evect, erange, uplo, n, A,
                                                        lda, B, ldb, vl, vu, il, iu, abstol, h_nev,
                                                        W, info);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_chegvdx_inplace(rocblas_handle handle,
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
                                                          rocblas_int* h_nev,
                                                          float* W,
                                                          rocblas_int* info)
{
    return rocsolver_sygvdx_hegvdx_inplace_impl<rocblas_float_complex>(
        handle, itype, evect, erange, uplo, n, A, lda, B, ldb, vl, vu, il, iu, abstol, h_nev, W,
        info);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_zhegvdx_inplace(rocblas_handle handle,
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
                                                          rocblas_int* h_nev,
                                                          double* W,
                                                          rocblas_int* info)
{
    return rocsolver_sygvdx_hegvdx_inplace_impl<rocblas_double_complex>(
        handle, itype, evect, erange, uplo, n, A, lda, B, ldb, vl, vu, il, iu, abstol, h_nev, W,
        info);
}

} // extern C
