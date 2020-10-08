/* ************************************************************************
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocauxiliary_orgbr_ungbr.hpp"

template <typename T>
rocblas_status rocsolver_orgbr_ungbr_impl(rocblas_handle handle,
                                          const rocblas_storev storev,
                                          const rocblas_int m,
                                          const rocblas_int n,
                                          const rocblas_int k,
                                          T* A,
                                          const rocblas_int lda,
                                          T* ipiv)
{
    if(!handle)
        return rocblas_status_invalid_handle;

    // logging is missing ???

    // argument checking
    rocblas_status st = rocsolver_orgbr_argCheck(storev, m, n, k, lda, A, ipiv);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideA = 0;
    rocblas_stride strideP = 0;
    rocblas_int batch_count = 1;

    // memory workspace sizes:
    // requirements for calling ORGQR/UNGQR or ORGLQ/UNGLQ
    size_t size_scalars;
    size_t size_workArr;
    size_t size_work;
    size_t size_Abyx_tmptr;
    size_t size_trfact;
    rocsolver_orgbr_ungbr_getMemorySize<T, false>(storev, m, n, k, batch_count, &size_scalars,
                                                  &size_work, &size_Abyx_tmptr, &size_trfact,
                                                  &size_workArr);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_scalars, size_work,
                                                      size_Abyx_tmptr, size_trfact, size_workArr);

    // memory workspace allocation
    void *scalars, *work, *Abyx_tmptr, *trfact, *workArr;
    rocblas_device_malloc mem(handle, size_scalars, size_work, size_Abyx_tmptr, size_trfact,
                              size_workArr);
    if(!mem)
        return rocblas_status_memory_error;

    scalars = mem[0];
    work = mem[1];
    Abyx_tmptr = mem[2];
    trfact = mem[3];
    workArr = mem[4];
    T sca[] = {-1, 0, 1};
    RETURN_IF_HIP_ERROR(hipMemcpy((T*)scalars, sca, size_scalars, hipMemcpyHostToDevice));

    // execution
    return rocsolver_orgbr_ungbr_template<false, false, T>(
        handle, storev, m, n, k, A, shiftA, lda, strideA, ipiv, strideP, batch_count, (T*)scalars,
        (T*)work, (T*)Abyx_tmptr, (T*)trfact, (T**)workArr);
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_sorgbr(rocblas_handle handle,
                                const rocblas_storev storev,
                                const rocblas_int m,
                                const rocblas_int n,
                                const rocblas_int k,
                                float* A,
                                const rocblas_int lda,
                                float* ipiv)
{
    return rocsolver_orgbr_ungbr_impl<float>(handle, storev, m, n, k, A, lda, ipiv);
}

rocblas_status rocsolver_dorgbr(rocblas_handle handle,
                                const rocblas_storev storev,
                                const rocblas_int m,
                                const rocblas_int n,
                                const rocblas_int k,
                                double* A,
                                const rocblas_int lda,
                                double* ipiv)
{
    return rocsolver_orgbr_ungbr_impl<double>(handle, storev, m, n, k, A, lda, ipiv);
}

rocblas_status rocsolver_cungbr(rocblas_handle handle,
                                const rocblas_storev storev,
                                const rocblas_int m,
                                const rocblas_int n,
                                const rocblas_int k,
                                rocblas_float_complex* A,
                                const rocblas_int lda,
                                rocblas_float_complex* ipiv)
{
    return rocsolver_orgbr_ungbr_impl<rocblas_float_complex>(handle, storev, m, n, k, A, lda, ipiv);
}

rocblas_status rocsolver_zungbr(rocblas_handle handle,
                                const rocblas_storev storev,
                                const rocblas_int m,
                                const rocblas_int n,
                                const rocblas_int k,
                                rocblas_double_complex* A,
                                const rocblas_int lda,
                                rocblas_double_complex* ipiv)
{
    return rocsolver_orgbr_ungbr_impl<rocblas_double_complex>(handle, storev, m, n, k, A, lda, ipiv);
}

} // extern C
