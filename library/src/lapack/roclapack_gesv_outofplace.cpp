/* ************************************************************************
 * Copyright (c) 2019-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_gesv_outofplace.hpp"

/*
 * ===========================================================================
 *    gesv_outofplace is not intended for inclusion in the public API. It
 *    exists to provide a gesv method with a signature identical to
 *    the cuSOLVER implementation, for use exclusively in hipSOLVER.
 * ===========================================================================
 */

template <typename T>
rocblas_status rocsolver_gesv_outofplace_impl(rocblas_handle handle,
                                              const rocblas_int n,
                                              const rocblas_int nrhs,
                                              T* A,
                                              const rocblas_int lda,
                                              rocblas_int* ipiv,
                                              T* B,
                                              const rocblas_int ldb,
                                              T* X,
                                              const rocblas_int ldx,
                                              rocblas_int* info)
{
    ROCSOLVER_ENTER_TOP("gesv_outofplace", "-n", n, "--nrhs", nrhs, "--lda", lda, "--ldb", ldb,
                        "--ldx", ldx);

    using S = decltype(std::real(T{}));

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st
        = rocsolver_gesv_outofplace_argCheck(handle, n, nrhs, lda, ldb, ldx, A, B, X, ipiv, info);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;
    rocblas_int shiftB = 0;
    rocblas_int shiftX = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideA = 0;
    rocblas_stride strideB = 0;
    rocblas_stride strideX = 0;
    rocblas_stride strideP = 0;
    rocblas_int batch_count = 1;

    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // size of reusable workspace for calling GETRF and GETRS
    bool optim_mem;
    size_t size_work1, size_work2, size_work3, size_work4;
    // extra requirements for calling GETRF
    size_t size_pivotval, size_pivotidx, size_iinfo;
    rocsolver_gesv_outofplace_getMemorySize<false, false, T>(
        n, nrhs, batch_count, &size_scalars, &size_work1, &size_work2, &size_work3, &size_work4,
        &size_pivotval, &size_pivotidx, &size_iinfo, &optim_mem);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_scalars, size_work1, size_work2,
                                                      size_work3, size_work4, size_pivotval,
                                                      size_pivotidx, size_iinfo);

    // memory workspace allocation
    void *scalars, *work1, *work2, *work3, *work4, *pivotval, *pivotidx, *iinfo;
    rocblas_device_malloc mem(handle, size_scalars, size_work1, size_work2, size_work3, size_work4,
                              size_pivotval, size_pivotidx, size_iinfo);

    if(!mem)
        return rocblas_status_memory_error;

    scalars = mem[0];
    work1 = mem[1];
    work2 = mem[2];
    work3 = mem[3];
    work4 = mem[4];
    pivotval = mem[5];
    pivotidx = mem[6];
    iinfo = mem[7];
    if(size_scalars > 0)
        init_scalars(handle, (T*)scalars);

    // execution
    return rocsolver_gesv_outofplace_template<false, false, T>(
        handle, n, nrhs, A, shiftA, lda, strideA, ipiv, strideP, B, shiftB, ldb, strideB, X, shiftX,
        ldx, strideX, info, batch_count, (T*)scalars, work1, work2, work3, work4, (T*)pivotval,
        (rocblas_int*)pivotidx, (rocblas_int*)iinfo, optim_mem);
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

ROCSOLVER_EXPORT rocblas_status rocsolver_sgesv_outofplace(rocblas_handle handle,
                                                           const rocblas_int n,
                                                           const rocblas_int nrhs,
                                                           float* A,
                                                           const rocblas_int lda,
                                                           rocblas_int* ipiv,
                                                           float* B,
                                                           const rocblas_int ldb,
                                                           float* X,
                                                           const rocblas_int ldx,
                                                           rocblas_int* info)
{
    return rocsolver_gesv_outofplace_impl<float>(handle, n, nrhs, A, lda, ipiv, B, ldb, X, ldx, info);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_dgesv_outofplace(rocblas_handle handle,
                                                           const rocblas_int n,
                                                           const rocblas_int nrhs,
                                                           double* A,
                                                           const rocblas_int lda,
                                                           rocblas_int* ipiv,
                                                           double* B,
                                                           const rocblas_int ldb,
                                                           double* X,
                                                           const rocblas_int ldx,
                                                           rocblas_int* info)
{
    return rocsolver_gesv_outofplace_impl<double>(handle, n, nrhs, A, lda, ipiv, B, ldb, X, ldx,
                                                  info);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_cgesv_outofplace(rocblas_handle handle,
                                                           const rocblas_int n,
                                                           const rocblas_int nrhs,
                                                           rocblas_float_complex* A,
                                                           const rocblas_int lda,
                                                           rocblas_int* ipiv,
                                                           rocblas_float_complex* B,
                                                           const rocblas_int ldb,
                                                           rocblas_float_complex* X,
                                                           const rocblas_int ldx,
                                                           rocblas_int* info)
{
    return rocsolver_gesv_outofplace_impl<rocblas_float_complex>(handle, n, nrhs, A, lda, ipiv, B,
                                                                 ldb, X, ldx, info);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_zgesv_outofplace(rocblas_handle handle,
                                                           const rocblas_int n,
                                                           const rocblas_int nrhs,
                                                           rocblas_double_complex* A,
                                                           const rocblas_int lda,
                                                           rocblas_int* ipiv,
                                                           rocblas_double_complex* B,
                                                           const rocblas_int ldb,
                                                           rocblas_double_complex* X,
                                                           const rocblas_int ldx,
                                                           rocblas_int* info)
{
    return rocsolver_gesv_outofplace_impl<rocblas_double_complex>(handle, n, nrhs, A, lda, ipiv, B,
                                                                  ldb, X, ldx, info);
}

} // extern C
