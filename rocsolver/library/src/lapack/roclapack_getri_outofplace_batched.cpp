/* ************************************************************************
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_getri.hpp"

/*
 * ===========================================================================
 *    getri_outofplace_batched is not intended for inclusion in the public API.
 *    It exists to provide a getri_batched method with a signature identical to
 *    the cuBLAS implementation, for use exclusively in hipBLAS.
 * ===========================================================================
 */

template <typename T, typename U>
rocblas_status rocsolver_getri_outofplace_batched_impl(rocblas_handle handle,
                                                       const rocblas_int n,
                                                       U A,
                                                       const rocblas_int lda,
                                                       rocblas_int* ipiv,
                                                       const rocblas_stride strideP,
                                                       U C,
                                                       const rocblas_int ldc,
                                                       rocblas_int* info,
                                                       const rocblas_int batch_count)
{
    if(!handle)
        return rocblas_status_invalid_handle;

    // logging is missing ???

    // argument checking
    rocblas_status st = rocsolver_getri_argCheck(n, lda, ldc, A, C, ipiv, info, batch_count);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;
    rocblas_int shiftP = 0;
    rocblas_int shiftC = 0;

    // batched execution
    rocblas_stride strideA = 0;
    rocblas_stride strideC = 0;

    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // size of reusable workspace (for calling TRSM and TRTRI)
    size_t size_work1, size_work2, size_work3, size_work4;
    // size of temporary array required for copies
    size_t size_tmpcopy;
    // size of arrays of pointers (for batched cases)
    size_t size_workArr;
    rocsolver_getri_getMemorySize<true, false, T>(n, batch_count, &size_scalars, &size_work1,
                                                  &size_work2, &size_work3, &size_work4,
                                                  &size_tmpcopy, &size_workArr);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_scalars, size_work1, size_work2,
                                                      size_work3, size_work4, size_tmpcopy,
                                                      size_workArr);

    // always allocate all required memory for TRSM optimal performance
    bool optim_mem = true;

    // memory workspace allocation
    void *scalars, *work1, *work2, *work3, *work4, *tmpcopy, *workArr;
    rocblas_device_malloc mem(handle, size_scalars, size_work1, size_work2, size_work3, size_work4,
                              size_tmpcopy, size_workArr);

    if(!mem)
        return rocblas_status_memory_error;

    scalars = mem[0];
    work1 = mem[1];
    work2 = mem[2];
    work3 = mem[3];
    work4 = mem[4];
    tmpcopy = mem[5];
    workArr = mem[6];
    T sca[] = {-1, 0, 1};
    RETURN_IF_HIP_ERROR(hipMemcpy((T*)scalars, sca, size_scalars, hipMemcpyHostToDevice));

    // out-of-place execution
    return rocsolver_getri_template<true, false, T>(
        handle, n, A, shiftA, lda, strideA, C, shiftC, ldc, strideC, ipiv, shiftP, strideP, info,
        batch_count, (T*)scalars, work1, work2, work3, work4, (T*)tmpcopy, (T**)workArr, optim_mem);
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

ROCSOLVER_EXPORT rocblas_status rocsolver_sgetri_outofplace_batched(rocblas_handle handle,
                                                                    const rocblas_int n,
                                                                    float* const A[],
                                                                    const rocblas_int lda,
                                                                    rocblas_int* ipiv,
                                                                    const rocblas_stride strideP,
                                                                    float* const C[],
                                                                    const rocblas_int ldc,
                                                                    rocblas_int* info,
                                                                    const rocblas_int batch_count)
{
    return rocsolver_getri_outofplace_batched_impl<float>(handle, n, A, lda, ipiv, strideP, C, ldc,
                                                          info, batch_count);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_dgetri_outofplace_batched(rocblas_handle handle,
                                                                    const rocblas_int n,
                                                                    double* const A[],
                                                                    const rocblas_int lda,
                                                                    rocblas_int* ipiv,
                                                                    const rocblas_stride strideP,
                                                                    double* const C[],
                                                                    const rocblas_int ldc,
                                                                    rocblas_int* info,
                                                                    const rocblas_int batch_count)
{
    return rocsolver_getri_outofplace_batched_impl<double>(handle, n, A, lda, ipiv, strideP, C, ldc,
                                                           info, batch_count);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_cgetri_outofplace_batched(rocblas_handle handle,
                                                                    const rocblas_int n,
                                                                    rocblas_float_complex* const A[],
                                                                    const rocblas_int lda,
                                                                    rocblas_int* ipiv,
                                                                    const rocblas_stride strideP,
                                                                    rocblas_float_complex* const C[],
                                                                    const rocblas_int ldc,
                                                                    rocblas_int* info,
                                                                    const rocblas_int batch_count)
{
    return rocsolver_getri_outofplace_batched_impl<rocblas_float_complex>(
        handle, n, A, lda, ipiv, strideP, C, ldc, info, batch_count);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_zgetri_outofplace_batched(rocblas_handle handle,
                                                                    const rocblas_int n,
                                                                    rocblas_double_complex* const A[],
                                                                    const rocblas_int lda,
                                                                    rocblas_int* ipiv,
                                                                    const rocblas_stride strideP,
                                                                    rocblas_double_complex* const C[],
                                                                    const rocblas_int ldc,
                                                                    rocblas_int* info,
                                                                    const rocblas_int batch_count)
{
    return rocsolver_getri_outofplace_batched_impl<rocblas_double_complex>(
        handle, n, A, lda, ipiv, strideP, C, ldc, info, batch_count);
}

} // extern C
