/* ************************************************************************
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocauxiliary_ormbr_unmbr.hpp"

template <typename T, bool COMPLEX = is_complex<T>>
rocblas_status rocsolver_ormbr_unmbr_impl(rocblas_handle handle,
                                          const rocblas_storev storev,
                                          const rocblas_side side,
                                          const rocblas_operation trans,
                                          const rocblas_int m,
                                          const rocblas_int n,
                                          const rocblas_int k,
                                          T* A,
                                          const rocblas_int lda,
                                          T* ipiv,
                                          T* C,
                                          const rocblas_int ldc)
{
    if(!handle)
        return rocblas_status_invalid_handle;

    // logging is missing ???

    // argument checking
    rocblas_status st
        = rocsolver_ormbr_argCheck<COMPLEX>(storev, side, trans, m, n, k, lda, ldc, A, C, ipiv);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;
    rocblas_int shiftC = 0;

    // normal (non-batched non-strided) execution
    rocblas_int strideA = 0;
    rocblas_int strideP = 0;
    rocblas_int strideC = 0;
    rocblas_int batch_count = 1;

    // memory workspace sizes:
    // requirements for calling ORMQR/UNMQR or ORMLQ/UNMLQ
    size_t size_scalars;
    size_t size_AbyxORwork, size_diagORtmptr;
    size_t size_trfact;
    size_t size_workArr;
    rocsolver_ormbr_unmbr_getMemorySize<T, false>(storev, side, m, n, k, batch_count, &size_scalars,
                                                  &size_AbyxORwork, &size_diagORtmptr, &size_trfact,
                                                  &size_workArr);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_scalars, size_AbyxORwork,
                                                      size_diagORtmptr, size_trfact, size_workArr);

    // memory workspace allocation
    void *scalars, *AbyxORwork, *diagORtmptr, *trfact, *workArr;
    rocblas_device_malloc mem(handle, size_scalars, size_AbyxORwork, size_diagORtmptr, size_trfact,
                              size_workArr);
    if(!mem)
        return rocblas_status_memory_error;

    scalars = mem[0];
    AbyxORwork = mem[1];
    diagORtmptr = mem[2];
    trfact = mem[3];
    workArr = mem[4];
    T sca[] = {-1, 0, 1};
    RETURN_IF_HIP_ERROR(hipMemcpy((T*)scalars, sca, size_scalars, hipMemcpyHostToDevice));

    // execution
    return rocsolver_ormbr_unmbr_template<false, false, T>(
        handle, storev, side, trans, m, n, k, A, shiftA, lda, strideA, ipiv, strideP, C, shiftC, ldc,
        strideC, batch_count, (T*)scalars, (T*)AbyxORwork, (T*)diagORtmptr, (T*)trfact, (T**)workArr);
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_sormbr(rocblas_handle handle,
                                const rocblas_storev storev,
                                const rocblas_side side,
                                const rocblas_operation trans,
                                const rocblas_int m,
                                const rocblas_int n,
                                const rocblas_int k,
                                float* A,
                                const rocblas_int lda,
                                float* ipiv,
                                float* C,
                                const rocblas_int ldc)
{
    return rocsolver_ormbr_unmbr_impl<float>(handle, storev, side, trans, m, n, k, A, lda, ipiv, C,
                                             ldc);
}

rocblas_status rocsolver_dormbr(rocblas_handle handle,
                                const rocblas_storev storev,
                                const rocblas_side side,
                                const rocblas_operation trans,
                                const rocblas_int m,
                                const rocblas_int n,
                                const rocblas_int k,
                                double* A,
                                const rocblas_int lda,
                                double* ipiv,
                                double* C,
                                const rocblas_int ldc)
{
    return rocsolver_ormbr_unmbr_impl<double>(handle, storev, side, trans, m, n, k, A, lda, ipiv, C,
                                              ldc);
}

rocblas_status rocsolver_cunmbr(rocblas_handle handle,
                                const rocblas_storev storev,
                                const rocblas_side side,
                                const rocblas_operation trans,
                                const rocblas_int m,
                                const rocblas_int n,
                                const rocblas_int k,
                                rocblas_float_complex* A,
                                const rocblas_int lda,
                                rocblas_float_complex* ipiv,
                                rocblas_float_complex* C,
                                const rocblas_int ldc)
{
    return rocsolver_ormbr_unmbr_impl<rocblas_float_complex>(handle, storev, side, trans, m, n, k,
                                                             A, lda, ipiv, C, ldc);
}

rocblas_status rocsolver_zunmbr(rocblas_handle handle,
                                const rocblas_storev storev,
                                const rocblas_side side,
                                const rocblas_operation trans,
                                const rocblas_int m,
                                const rocblas_int n,
                                const rocblas_int k,
                                rocblas_double_complex* A,
                                const rocblas_int lda,
                                rocblas_double_complex* ipiv,
                                rocblas_double_complex* C,
                                const rocblas_int ldc)
{
    return rocsolver_ormbr_unmbr_impl<rocblas_double_complex>(handle, storev, side, trans, m, n, k,
                                                              A, lda, ipiv, C, ldc);
}

} // extern C
