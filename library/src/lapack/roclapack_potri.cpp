/* ************************************************************************
 * Copyright (c) 2019-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_potri.hpp"

template <typename T, typename U>
rocblas_status rocsolver_potri_impl(rocblas_handle handle,
                                    const rocblas_fill uplo,
                                    const rocblas_int n,
                                    U A,
                                    const rocblas_int lda,
                                    rocblas_int* info)
{
    ROCSOLVER_ENTER_TOP("potri", "--uplo", uplo, "-n", n, "--lda", lda);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_potri_argCheck(handle, uplo, n, lda, A, info);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideA = 0;
    rocblas_int batch_count = 1;

    // memory workspace sizes:
    // size of workspace (for calling TRTRI)
    bool optim_mem;
    size_t size_work1, size_work2, size_work3, size_work4, size_tmpcopy;
    // size of arrays of pointers (for batched cases)
    size_t size_workArr;
    rocsolver_potri_getMemorySize<false, false, T>(n, batch_count, &size_work1, &size_work2,
                                                   &size_work3, &size_work4, &size_tmpcopy,
                                                   &size_workArr, &optim_mem);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_work1, size_work2, size_work3,
                                                      size_work4, size_tmpcopy, size_workArr);

    // memory workspace allocation
    void *work1, *work2, *work3, *work4, *tmpcopy, *workArr;
    rocblas_device_malloc mem(handle, size_work1, size_work2, size_work3, size_work4, size_tmpcopy,
                              size_workArr);

    if(!mem)
        return rocblas_status_memory_error;

    work1 = mem[0];
    work2 = mem[1];
    work3 = mem[2];
    work4 = mem[3];
    tmpcopy = mem[4];
    workArr = mem[5];

    // execution
    return rocsolver_potri_template<false, false, T>(handle, uplo, n, A, shiftA, lda, strideA, info,
                                                     batch_count, work1, work2, work3, work4,
                                                     (T*)tmpcopy, (T**)workArr, optim_mem);
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_spotri(rocblas_handle handle,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                float* A,
                                const rocblas_int lda,
                                rocblas_int* info)
{
    return rocsolver_potri_impl<float>(handle, uplo, n, A, lda, info);
}

rocblas_status rocsolver_dpotri(rocblas_handle handle,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                double* A,
                                const rocblas_int lda,
                                rocblas_int* info)
{
    return rocsolver_potri_impl<double>(handle, uplo, n, A, lda, info);
}

rocblas_status rocsolver_cpotri(rocblas_handle handle,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                rocblas_float_complex* A,
                                const rocblas_int lda,
                                rocblas_int* info)
{
    return rocsolver_potri_impl<rocblas_float_complex>(handle, uplo, n, A, lda, info);
}

rocblas_status rocsolver_zpotri(rocblas_handle handle,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                rocblas_double_complex* A,
                                const rocblas_int lda,
                                rocblas_int* info)
{
    return rocsolver_potri_impl<rocblas_double_complex>(handle, uplo, n, A, lda, info);
}
}
