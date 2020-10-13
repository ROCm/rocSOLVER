/* ************************************************************************
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocauxiliary_larf.hpp"

template <typename T>
rocblas_status rocsolver_larf_impl(rocblas_handle handle,
                                   const rocblas_side side,
                                   const rocblas_int m,
                                   const rocblas_int n,
                                   T* x,
                                   const rocblas_int incx,
                                   const T* alpha,
                                   T* A,
                                   const rocblas_int lda)
{
    if(!handle)
        return rocblas_status_invalid_handle;

    // logging is missing ???

    // argument checking
    rocblas_status st = rocsolver_larf_argCheck(side, m, n, lda, incx, x, A, alpha);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;
    rocblas_int shiftx = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride stridex = 0;
    rocblas_stride stridea = 0;
    rocblas_stride stridep = 0;
    rocblas_int batch_count = 1;

    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // size for temporary results in generation of Householder matrix
    size_t size_Abyx;
    // size of arrays of pointers (for batched cases)
    size_t size_workArr;
    rocsolver_larf_getMemorySize<T, false>(side, m, n, batch_count, &size_scalars, &size_Abyx,
                                           &size_workArr);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_scalars, size_Abyx, size_workArr);

    // memory workspace allocation
    void *scalars, *Abyx, *workArr;
    rocblas_device_malloc mem(handle, size_scalars, size_Abyx, size_workArr);
    if(!mem)
        return rocblas_status_memory_error;

    scalars = mem[0];
    Abyx = mem[1];
    workArr = mem[2];
    T sca[] = {-1, 0, 1};
    RETURN_IF_HIP_ERROR(hipMemcpy((T*)scalars, sca, size_scalars, hipMemcpyHostToDevice));

    // execution
    return rocsolver_larf_template<T>(handle, side, m, n, x, shiftx, incx, stridex, alpha, stridep,
                                      A, shiftA, lda, stridea, batch_count, (T*)scalars, (T*)Abyx,
                                      (T**)workArr);
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_slarf(rocblas_handle handle,
                               const rocblas_side side,
                               const rocblas_int m,
                               const rocblas_int n,
                               float* x,
                               const rocblas_int incx,
                               const float* alpha,
                               float* A,
                               const rocblas_int lda)
{
    return rocsolver_larf_impl<float>(handle, side, m, n, x, incx, alpha, A, lda);
}

rocblas_status rocsolver_dlarf(rocblas_handle handle,
                               const rocblas_side side,
                               const rocblas_int m,
                               const rocblas_int n,
                               double* x,
                               const rocblas_int incx,
                               const double* alpha,
                               double* A,
                               const rocblas_int lda)
{
    return rocsolver_larf_impl<double>(handle, side, m, n, x, incx, alpha, A, lda);
}

rocblas_status rocsolver_clarf(rocblas_handle handle,
                               const rocblas_side side,
                               const rocblas_int m,
                               const rocblas_int n,
                               rocblas_float_complex* x,
                               const rocblas_int incx,
                               const rocblas_float_complex* alpha,
                               rocblas_float_complex* A,
                               const rocblas_int lda)
{
    return rocsolver_larf_impl<rocblas_float_complex>(handle, side, m, n, x, incx, alpha, A, lda);
}

rocblas_status rocsolver_zlarf(rocblas_handle handle,
                               const rocblas_side side,
                               const rocblas_int m,
                               const rocblas_int n,
                               rocblas_double_complex* x,
                               const rocblas_int incx,
                               const rocblas_double_complex* alpha,
                               rocblas_double_complex* A,
                               const rocblas_int lda)
{
    return rocsolver_larf_impl<rocblas_double_complex>(handle, side, m, n, x, incx, alpha, A, lda);
}

} // extern C
