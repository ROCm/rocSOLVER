/* ************************************************************************
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_posv.hpp"

template <typename T>
rocblas_status rocsolver_posv_impl(rocblas_handle handle,
                                   const rocblas_fill uplo,
                                   const rocblas_int n,
                                   const rocblas_int nrhs,
                                   T* A,
                                   const rocblas_int lda,
                                   T* B,
                                   const rocblas_int ldb,
                                   rocblas_int* info)
{
    ROCSOLVER_ENTER_TOP("posv", "--uplo", uplo, "-n", n, "--nrhs", nrhs, "--lda", lda, "--ldb", ldb);

    using S = decltype(std::real(T{}));

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_posv_argCheck(handle, uplo, n, nrhs, lda, ldb, A, B, info);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;
    rocblas_int shiftB = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideA = 0;
    rocblas_stride strideB = 0;
    rocblas_int batch_count = 1;

    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // size of reusable workspace (and for calling TRSM)
    size_t size_work1, size_work2, size_work3, size_work4;
    // extra requirements for calling POTRF and to copy B
    size_t size_pivots_savedB, size_iinfo;
    rocsolver_posv_getMemorySize<false, T>(n, nrhs, uplo, batch_count, &size_scalars, &size_work1,
                                           &size_work2, &size_work3, &size_work4,
                                           &size_pivots_savedB, &size_iinfo);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_scalars, size_work1, size_work2,
                                                      size_work3, size_work4, size_pivots_savedB,
                                                      size_iinfo);

    // always allocate all required memory for TRSM optimal performance
    bool optim_mem = true;

    // memory workspace allocation
    void *scalars, *work1, *work2, *work3, *work4, *pivots_savedB, *iinfo;
    rocblas_device_malloc mem(handle, size_scalars, size_work1, size_work2, size_work3, size_work4,
                              size_pivots_savedB, size_iinfo);

    if(!mem)
        return rocblas_status_memory_error;

    scalars = mem[0];
    work1 = mem[1];
    work2 = mem[2];
    work3 = mem[3];
    work4 = mem[4];
    pivots_savedB = mem[5];
    iinfo = mem[6];
    if(size_scalars > 0)
        init_scalars(handle, (T*)scalars);

    // execution
    return rocsolver_posv_template<false, T, S>(
        handle, uplo, n, nrhs, A, shiftA, lda, strideA, B, shiftB, ldb, strideB, info, batch_count,
        (T*)scalars, work1, work2, work3, work4, (T*)pivots_savedB, (rocblas_int*)iinfo, optim_mem);
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocblas_status rocsolver_sposv(rocblas_handle handle,
                                          const rocblas_fill uplo,
                                          const rocblas_int n,
                                          const rocblas_int nrhs,
                                          float* A,
                                          const rocblas_int lda,
                                          float* B,
                                          const rocblas_int ldb,
                                          rocblas_int* info)
{
    return rocsolver_posv_impl<float>(handle, uplo, n, nrhs, A, lda, B, ldb, info);
}

extern "C" rocblas_status rocsolver_dposv(rocblas_handle handle,
                                          const rocblas_fill uplo,
                                          const rocblas_int n,
                                          const rocblas_int nrhs,
                                          double* A,
                                          const rocblas_int lda,
                                          double* B,
                                          const rocblas_int ldb,
                                          rocblas_int* info)
{
    return rocsolver_posv_impl<double>(handle, uplo, n, nrhs, A, lda, B, ldb, info);
}

extern "C" rocblas_status rocsolver_cposv(rocblas_handle handle,
                                          const rocblas_fill uplo,
                                          const rocblas_int n,
                                          const rocblas_int nrhs,
                                          rocblas_float_complex* A,
                                          const rocblas_int lda,
                                          rocblas_float_complex* B,
                                          const rocblas_int ldb,
                                          rocblas_int* info)
{
    return rocsolver_posv_impl<rocblas_float_complex>(handle, uplo, n, nrhs, A, lda, B, ldb, info);
}

extern "C" rocblas_status rocsolver_zposv(rocblas_handle handle,
                                          const rocblas_fill uplo,
                                          const rocblas_int n,
                                          const rocblas_int nrhs,
                                          rocblas_double_complex* A,
                                          const rocblas_int lda,
                                          rocblas_double_complex* B,
                                          const rocblas_int ldb,
                                          rocblas_int* info)
{
    return rocsolver_posv_impl<rocblas_double_complex>(handle, uplo, n, nrhs, A, lda, B, ldb, info);
}
