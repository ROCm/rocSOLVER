/* ************************************************************************
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocauxiliary_ormtr_unmtr.hpp"

template <typename T, bool COMPLEX = is_complex<T>>
rocblas_status rocsolver_ormtr_unmtr_impl(rocblas_handle handle,
                                          const rocblas_side side,
                                          const rocblas_fill uplo,
                                          const rocblas_operation trans,
                                          const rocblas_int m,
                                          const rocblas_int n,
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
        = rocsolver_ormtr_argCheck<COMPLEX>(side, uplo, trans, m, n, lda, ldc, A, C, ipiv);
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
    // requirements for calling ORMQL/UNMQL or ORMQR/UNMQR
    size_t size_scalars;
    size_t size_AbyxORwork, size_diagORtmptr;
    size_t size_trfact;
    size_t size_workArr;
    rocsolver_ormtr_unmtr_getMemorySize<T, false>(side, uplo, m, n, batch_count, &size_scalars,
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
    return rocsolver_ormtr_unmtr_template<false, false, T>(
        handle, side, uplo, trans, m, n, A, shiftA, lda, strideA, ipiv, strideP, C, shiftC, ldc,
        strideC, batch_count, (T*)scalars, (T*)AbyxORwork, (T*)diagORtmptr, (T*)trfact, (T**)workArr);
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_sormtr(rocblas_handle handle,
                                const rocblas_side side,
                                const rocblas_fill uplo,
                                const rocblas_operation trans,
                                const rocblas_int m,
                                const rocblas_int n,
                                float* A,
                                const rocblas_int lda,
                                float* ipiv,
                                float* C,
                                const rocblas_int ldc)
{
    return rocsolver_ormtr_unmtr_impl<float>(handle, side, uplo, trans, m, n, A, lda, ipiv, C, ldc);
}

rocblas_status rocsolver_dormtr(rocblas_handle handle,
                                const rocblas_side side,
                                const rocblas_fill uplo,
                                const rocblas_operation trans,
                                const rocblas_int m,
                                const rocblas_int n,
                                double* A,
                                const rocblas_int lda,
                                double* ipiv,
                                double* C,
                                const rocblas_int ldc)
{
    return rocsolver_ormtr_unmtr_impl<double>(handle, side, uplo, trans, m, n, A, lda, ipiv, C, ldc);
}

rocblas_status rocsolver_cunmtr(rocblas_handle handle,
                                const rocblas_side side,
                                const rocblas_fill uplo,
                                const rocblas_operation trans,
                                const rocblas_int m,
                                const rocblas_int n,
                                rocblas_float_complex* A,
                                const rocblas_int lda,
                                rocblas_float_complex* ipiv,
                                rocblas_float_complex* C,
                                const rocblas_int ldc)
{
    return rocsolver_ormtr_unmtr_impl<rocblas_float_complex>(handle, side, uplo, trans, m, n, A,
                                                             lda, ipiv, C, ldc);
}

rocblas_status rocsolver_zunmtr(rocblas_handle handle,
                                const rocblas_side side,
                                const rocblas_fill uplo,
                                const rocblas_operation trans,
                                const rocblas_int m,
                                const rocblas_int n,
                                rocblas_double_complex* A,
                                const rocblas_int lda,
                                rocblas_double_complex* ipiv,
                                rocblas_double_complex* C,
                                const rocblas_int ldc)
{
    return rocsolver_ormtr_unmtr_impl<rocblas_double_complex>(handle, side, uplo, trans, m, n, A,
                                                              lda, ipiv, C, ldc);
}

} // extern C
