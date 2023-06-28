/* ************************************************************************
 * Copyright (C) 2019-2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, 
 * are permitted provided that the following conditions are met:
 * 1)Redistributions of source code must retain the above copyright notice, 
 * this list of conditions and the following disclaimer.
 * 2)Redistributions in binary form must reproduce the above copyright notice, 
 * this list of conditions and the following disclaimer in the documentation 
 * and/or other materials provided with the distribution.
 * ************************************************************************ */

#include "roclapack_posv.hpp"

template <typename T, typename U>
rocblas_status rocsolver_posv_batched_impl(rocblas_handle handle,
                                           const rocblas_fill uplo,
                                           const rocblas_int n,
                                           const rocblas_int nrhs,
                                           U A,
                                           const rocblas_int lda,
                                           U B,
                                           const rocblas_int ldb,
                                           rocblas_int* info,
                                           const rocblas_int batch_count)
{
    ROCSOLVER_ENTER_TOP("posv_batched", "--uplo", uplo, "-n", n, "--nrhs", nrhs, "--lda", lda,
                        "--ldb", ldb, "--batch_count", batch_count);

    using S = decltype(std::real(T{}));

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st
        = rocsolver_posv_argCheck(handle, uplo, n, nrhs, lda, ldb, A, B, info, batch_count);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;
    rocblas_int shiftB = 0;

    // batched execution
    rocblas_stride strideA = 0;
    rocblas_stride strideB = 0;

    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // size of workspace (for calling POTRF and POTRS)
    bool optim_mem;
    size_t size_work1, size_work2, size_work3, size_work4;
    // extra requirements for calling POTRF and to copy B
    size_t size_pivots_savedB, size_iinfo;
    rocsolver_posv_getMemorySize<true, false, T>(n, nrhs, uplo, batch_count, &size_scalars,
                                                 &size_work1, &size_work2, &size_work3, &size_work4,
                                                 &size_pivots_savedB, &size_iinfo, &optim_mem);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_scalars, size_work1, size_work2,
                                                      size_work3, size_work4, size_pivots_savedB,
                                                      size_iinfo);

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
    return rocsolver_posv_template<true, false, T, S>(
        handle, uplo, n, nrhs, A, shiftA, lda, strideA, B, shiftB, ldb, strideB, info, batch_count,
        (T*)scalars, work1, work2, work3, work4, (T*)pivots_savedB, (rocblas_int*)iinfo, optim_mem);
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocblas_status rocsolver_sposv_batched(rocblas_handle handle,
                                                  const rocblas_fill uplo,
                                                  const rocblas_int n,
                                                  const rocblas_int nrhs,
                                                  float* const A[],
                                                  const rocblas_int lda,
                                                  float* const B[],
                                                  const rocblas_int ldb,
                                                  rocblas_int* info,
                                                  const rocblas_int batch_count)
{
    return rocsolver_posv_batched_impl<float>(handle, uplo, n, nrhs, A, lda, B, ldb, info,
                                              batch_count);
}

extern "C" rocblas_status rocsolver_dposv_batched(rocblas_handle handle,
                                                  const rocblas_fill uplo,
                                                  const rocblas_int n,
                                                  const rocblas_int nrhs,
                                                  double* const A[],
                                                  const rocblas_int lda,
                                                  double* const B[],
                                                  const rocblas_int ldb,
                                                  rocblas_int* info,
                                                  const rocblas_int batch_count)
{
    return rocsolver_posv_batched_impl<double>(handle, uplo, n, nrhs, A, lda, B, ldb, info,
                                               batch_count);
}

extern "C" rocblas_status rocsolver_cposv_batched(rocblas_handle handle,
                                                  const rocblas_fill uplo,
                                                  const rocblas_int n,
                                                  const rocblas_int nrhs,
                                                  rocblas_float_complex* const A[],
                                                  const rocblas_int lda,
                                                  rocblas_float_complex* const B[],
                                                  const rocblas_int ldb,
                                                  rocblas_int* info,
                                                  const rocblas_int batch_count)
{
    return rocsolver_posv_batched_impl<rocblas_float_complex>(handle, uplo, n, nrhs, A, lda, B, ldb,
                                                              info, batch_count);
}

extern "C" rocblas_status rocsolver_zposv_batched(rocblas_handle handle,
                                                  const rocblas_fill uplo,
                                                  const rocblas_int n,
                                                  const rocblas_int nrhs,
                                                  rocblas_double_complex* const A[],
                                                  const rocblas_int lda,
                                                  rocblas_double_complex* const B[],
                                                  const rocblas_int ldb,
                                                  rocblas_int* info,
                                                  const rocblas_int batch_count)
{
    return rocsolver_posv_batched_impl<rocblas_double_complex>(handle, uplo, n, nrhs, A, lda, B,
                                                               ldb, info, batch_count);
}
