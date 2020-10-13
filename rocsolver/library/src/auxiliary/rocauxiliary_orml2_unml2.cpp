/* ************************************************************************
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocauxiliary_orml2_unml2.hpp"

template <typename T, bool COMPLEX = is_complex<T>>
rocblas_status rocsolver_orml2_unml2_impl(rocblas_handle handle,
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
        = rocsolver_orml2_ormlq_argCheck<COMPLEX>(side, trans, m, n, k, lda, ldc, A, C, ipiv);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;
    rocblas_int shiftC = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideA = 0;
    rocblas_stride strideP = 0;
    rocblas_stride strideC = 0;
    rocblas_int batch_count = 1;

    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // extra requirements for calling LARF
    size_t size_Abyx;
    // size of temporary array for diagonal elements
    size_t size_diag;
    // size of arrays of pointers (for batched cases)
    size_t size_workArr;
    rocsolver_orml2_unml2_getMemorySize<T, false>(side, m, n, k, batch_count, &size_scalars,
                                                  &size_Abyx, &size_diag, &size_workArr);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_scalars, size_Abyx, size_diag,
                                                      size_workArr);

    // memory workspace allocation
    void *scalars, *Abyx, *diag, *workArr;
    rocblas_device_malloc mem(handle, size_scalars, size_Abyx, size_diag, size_workArr);
    if(!mem)
        return rocblas_status_memory_error;

    scalars = mem[0];
    Abyx = mem[1];
    diag = mem[2];
    workArr = mem[3];
    T sca[] = {-1, 0, 1};
    RETURN_IF_HIP_ERROR(hipMemcpy((T*)scalars, sca, size_scalars, hipMemcpyHostToDevice));

    // execution
    return rocsolver_orml2_unml2_template<T>(handle, side, trans, m, n, k, A, shiftA, lda, strideA,
                                             ipiv, strideP, C, shiftC, ldc, strideC, batch_count,
                                             (T*)scalars, (T*)Abyx, (T*)diag, (T**)workArr);
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_sorml2(rocblas_handle handle,
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
    return rocsolver_orml2_unml2_impl<float>(handle, side, trans, m, n, k, A, lda, ipiv, C, ldc);
}

rocblas_status rocsolver_dorml2(rocblas_handle handle,
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
    return rocsolver_orml2_unml2_impl<double>(handle, side, trans, m, n, k, A, lda, ipiv, C, ldc);
}

rocblas_status rocsolver_cunml2(rocblas_handle handle,
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
    return rocsolver_orml2_unml2_impl<rocblas_float_complex>(handle, side, trans, m, n, k, A, lda,
                                                             ipiv, C, ldc);
}

rocblas_status rocsolver_zunml2(rocblas_handle handle,
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
    return rocsolver_orml2_unml2_impl<rocblas_double_complex>(handle, side, trans, m, n, k, A, lda,
                                                              ipiv, C, ldc);
}

} // extern C
