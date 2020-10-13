/* ************************************************************************
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_potrf.hpp"

template <typename S, typename T, typename U>
rocblas_status rocsolver_potrf_strided_batched_impl(rocblas_handle handle,
                                                    const rocblas_fill uplo,
                                                    const rocblas_int n,
                                                    U A,
                                                    const rocblas_int lda,
                                                    const rocblas_stride strideA,
                                                    rocblas_int* info,
                                                    const rocblas_int batch_count)
{
    if(!handle)
        return rocblas_status_invalid_handle;

    // logging is missing ???

    // argument checking
    rocblas_status st = rocsolver_potf2_potrf_argCheck(uplo, n, lda, A, info, batch_count);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;

    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // size of reusable workspace (and for calling TRSM)
    size_t size_work1, size_work2, size_work3, size_work4;
    // extra requirements for calling POTF2
    size_t size_pivots;
    // size to store info about positiveness of each subblock
    size_t size_iinfo;
    rocsolver_potrf_getMemorySize<false, T>(n, uplo, batch_count, &size_scalars, &size_work1,
                                            &size_work2, &size_work3, &size_work4, &size_pivots,
                                            &size_iinfo);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_scalars, size_work1, size_work2,
                                                      size_work3, size_work4, size_pivots,
                                                      size_iinfo);

    // always allocate all required memory for TRSM optimal performance
    bool optim_mem = true;

    // memory workspace allocation
    void *scalars, *work1, *work2, *work3, *work4, *pivots, *iinfo;
    rocblas_device_malloc mem(handle, size_scalars, size_work1, size_work2, size_work3, size_work4,
                              size_pivots, size_iinfo);

    if(!mem)
        return rocblas_status_memory_error;

    scalars = mem[0];
    work1 = mem[1];
    work2 = mem[2];
    work3 = mem[3];
    work4 = mem[4];
    pivots = mem[5];
    iinfo = mem[6];
    T sca[] = {-1, 0, 1};
    RETURN_IF_HIP_ERROR(hipMemcpy((T*)scalars, sca, size_scalars, hipMemcpyHostToDevice));

    // execution
    return rocsolver_potrf_template<false, S, T>(handle, uplo, n, A, shiftA, lda, strideA, info,
                                                 batch_count, (T*)scalars, work1, work2, work3,
                                                 work4, (T*)pivots, (rocblas_int*)iinfo, optim_mem);
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_spotrf_strided_batched(rocblas_handle handle,
                                                const rocblas_fill uplo,
                                                const rocblas_int n,
                                                float* A,
                                                const rocblas_int lda,
                                                const rocblas_stride strideA,
                                                rocblas_int* info,
                                                const rocblas_int batch_count)
{
    return rocsolver_potrf_strided_batched_impl<float, float>(handle, uplo, n, A, lda, strideA,
                                                              info, batch_count);
}

rocblas_status rocsolver_dpotrf_strided_batched(rocblas_handle handle,
                                                const rocblas_fill uplo,
                                                const rocblas_int n,
                                                double* A,
                                                const rocblas_int lda,
                                                const rocblas_stride strideA,
                                                rocblas_int* info,
                                                const rocblas_int batch_count)
{
    return rocsolver_potrf_strided_batched_impl<double, double>(handle, uplo, n, A, lda, strideA,
                                                                info, batch_count);
}

rocblas_status rocsolver_cpotrf_strided_batched(rocblas_handle handle,
                                                const rocblas_fill uplo,
                                                const rocblas_int n,
                                                rocblas_float_complex* A,
                                                const rocblas_int lda,
                                                const rocblas_stride strideA,
                                                rocblas_int* info,
                                                const rocblas_int batch_count)
{
    return rocsolver_potrf_strided_batched_impl<float, rocblas_float_complex>(
        handle, uplo, n, A, lda, strideA, info, batch_count);
}

rocblas_status rocsolver_zpotrf_strided_batched(rocblas_handle handle,
                                                const rocblas_fill uplo,
                                                const rocblas_int n,
                                                rocblas_double_complex* A,
                                                const rocblas_int lda,
                                                const rocblas_stride strideA,
                                                rocblas_int* info,
                                                const rocblas_int batch_count)
{
    return rocsolver_potrf_strided_batched_impl<double, rocblas_double_complex>(
        handle, uplo, n, A, lda, strideA, info, batch_count);
}
}
