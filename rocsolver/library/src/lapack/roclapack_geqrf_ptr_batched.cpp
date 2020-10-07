/* ************************************************************************
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_geqrf.hpp"

/*
 * ===========================================================================
 *    geqrf_ptr_batched is not intended for inclusion in the public API. It
 *    exists to provide a geqrf_batched method with a signature identical to
 *    the cuBLAS implementation, for use exclusively in hipBLAS.
 * ===========================================================================
 */

template <typename T>
__global__ void copy_array_to_ptrs(rocblas_stride n, T* const ptrs[], T* array)
{
    int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int b = hipBlockIdx_y;

    if(i < n)
        ptrs[b][i] = array[i + b * n];
}

template <typename T, typename U>
rocblas_status rocsolver_geqrf_ptr_batched_impl(rocblas_handle handle,
                                                const rocblas_int m,
                                                const rocblas_int n,
                                                U A,
                                                const rocblas_int lda,
                                                U tau,
                                                const rocblas_int batch_count)
{
    if(!handle)
        return rocblas_status_invalid_handle;

    // logging is missing ???

    // argument checking
    rocblas_status st = rocsolver_geqr2_geqrf_argCheck(m, n, lda, A, tau, batch_count);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;

    // batched execution
    rocblas_stride strideA = 0;
    rocblas_stride strideP = min(m, n);

    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // size of arrays of pointers (for batched cases) and re-usable workspace
    size_t size_work_workArr, size_workArr;
    // extra requirements for calling GEQR2 and to store temporary triangular factor
    size_t size_Abyx_norms_trfact;
    // extra requirements for calling GEQR2 and LARFB
    size_t size_diag_tmptr;
    rocsolver_geqrf_getMemorySize<T, true>(m, n, batch_count, &size_scalars, &size_work_workArr,
                                           &size_Abyx_norms_trfact, &size_diag_tmptr, &size_workArr);

    // this is to mamange tau as a simple array ipiv
    size_t size_ipiv = sizeof(T) * strideP * batch_count;

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_scalars, size_work_workArr,
                                                      size_Abyx_norms_trfact, size_diag_tmptr,
                                                      size_workArr, size_ipiv);

    // memory workspace allocation
    void *scalars, *work_workArr, *Abyx_norms_trfact, *diag_tmptr, *workArr, *ipiv;
    rocblas_device_malloc mem(handle, size_scalars, size_work_workArr, size_Abyx_norms_trfact,
                              size_diag_tmptr, size_workArr, size_ipiv);

    if(!mem)
        return rocblas_status_memory_error;

    scalars = mem[0];
    work_workArr = mem[1];
    Abyx_norms_trfact = mem[2];
    diag_tmptr = mem[3];
    workArr = mem[4];
    ipiv = mem[5];
    T sca[] = {-1, 0, 1};
    RETURN_IF_HIP_ERROR(hipMemcpy((T*)scalars, sca, size_scalars, hipMemcpyHostToDevice));

    // execution
    rocblas_status status = rocsolver_geqrf_template<true, false, T>(
        handle, m, n, A, shiftA, lda, strideA, (T*)ipiv, strideP, batch_count, (T*)scalars,
        work_workArr, (T*)Abyx_norms_trfact, (T*)diag_tmptr, (T**)workArr);

    // copy ipiv into tau
    if(size_ipiv > 0)
    {
        hipStream_t stream;
        rocblas_get_stream(handle, &stream);

        rocblas_int blocks = (strideP - 1) / 32 + 1;
        hipLaunchKernelGGL(copy_array_to_ptrs, dim3(blocks, batch_count), dim3(32, 1), 0, stream,
                           strideP, tau, (T*)ipiv);
    }

    return status;
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

ROCSOLVER_EXPORT rocblas_status rocsolver_sgeqrf_ptr_batched(rocblas_handle handle,
                                                             const rocblas_int m,
                                                             const rocblas_int n,
                                                             float* const A[],
                                                             const rocblas_int lda,
                                                             float* const ipiv[],
                                                             const rocblas_int batch_count)
{
    return rocsolver_geqrf_ptr_batched_impl<float>(handle, m, n, A, lda, ipiv, batch_count);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_dgeqrf_ptr_batched(rocblas_handle handle,
                                                             const rocblas_int m,
                                                             const rocblas_int n,
                                                             double* const A[],
                                                             const rocblas_int lda,
                                                             double* const ipiv[],
                                                             const rocblas_int batch_count)
{
    return rocsolver_geqrf_ptr_batched_impl<double>(handle, m, n, A, lda, ipiv, batch_count);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_cgeqrf_ptr_batched(rocblas_handle handle,
                                                             const rocblas_int m,
                                                             const rocblas_int n,
                                                             rocblas_float_complex* const A[],
                                                             const rocblas_int lda,
                                                             rocblas_float_complex* const ipiv[],
                                                             const rocblas_int batch_count)
{
    return rocsolver_geqrf_ptr_batched_impl<rocblas_float_complex>(handle, m, n, A, lda, ipiv,
                                                                   batch_count);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_zgeqrf_ptr_batched(rocblas_handle handle,
                                                             const rocblas_int m,
                                                             const rocblas_int n,
                                                             rocblas_double_complex* const A[],
                                                             const rocblas_int lda,
                                                             rocblas_double_complex* const ipiv[],
                                                             const rocblas_int batch_count)
{
    return rocsolver_geqrf_ptr_batched_impl<rocblas_double_complex>(handle, m, n, A, lda, ipiv,
                                                                    batch_count);
}

} // extern C
