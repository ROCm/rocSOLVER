/* ************************************************************************
 * Copyright (c) 2019-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_getf2.hpp"

template <bool PIVOT, typename T, typename U>
rocblas_status rocsolver_getf2_strided_batched_impl(rocblas_handle handle,
                                                    const rocblas_int m,
                                                    const rocblas_int n,
                                                    U A,
                                                    const rocblas_int lda,
                                                    const rocblas_stride strideA,
                                                    rocblas_int* ipiv,
                                                    const rocblas_stride strideP,
                                                    rocblas_int* info,
                                                    const rocblas_int batch_count)
{
    const char* name = (PIVOT ? "getf2_strided_batched" : "getf2_npvt_strided_batched");
    ROCSOLVER_ENTER_TOP(name, "-m", m, "-n", n, "--lda", lda, "--bsa", strideA, "--bsp", strideP,
                        "--batch", batch_count);

    using S = decltype(std::real(T{}));

    if(!handle)
        ROCSOLVER_RETURN_TOP(name, rocblas_status_invalid_handle);

    // logging is missing ???

    // argument checking
    rocblas_status st
        = rocsolver_getf2_getrf_argCheck(handle, m, n, lda, A, ipiv, info, PIVOT, batch_count);
    if(st != rocblas_status_continue)
        ROCSOLVER_RETURN_TOP(name, st);

    // using unshifted arrays
    rocblas_int shiftA = 0;
    rocblas_int shiftP = 0;

    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // size of reusable workspace
    size_t size_work;
    // sizes to store pivots in intermediate computations
    size_t size_pivotval;
    size_t size_pivotidx;
    rocsolver_getf2_getMemorySize<true, T, S>(m, n, batch_count, &size_scalars, &size_work,
                                              &size_pivotval, &size_pivotidx);

    if(rocblas_is_device_memory_size_query(handle))
        ROCSOLVER_RETURN_TOP(name,
                             rocblas_set_optimal_device_memory_size(handle, size_scalars, size_work,
                                                                    size_pivotval, size_pivotidx));

    // memory workspace allocation
    void *scalars, *pivotidx, *pivotval, *work;
    rocblas_device_malloc mem(handle, size_scalars, size_work, size_pivotval, size_pivotidx);

    if(!mem)
        ROCSOLVER_RETURN_TOP(name, rocblas_status_memory_error);

    scalars = mem[0];
    work = mem[1];
    pivotval = mem[2];
    pivotidx = mem[3];
    if(size_scalars > 0)
        init_scalars(handle, (T*)scalars);

    // execution
    ROCSOLVER_RETURN_TOP(name,
                         rocsolver_getf2_template<true, PIVOT, T, S>(
                             handle, m, n, A, shiftA, lda, strideA, ipiv, shiftP, strideP, info,
                             batch_count, (T*)scalars, (rocblas_index_value_t<S>*)work,
                             (T*)pivotval, (rocblas_int*)pivotidx));
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_sgetf2_strided_batched(rocblas_handle handle,
                                                const rocblas_int m,
                                                const rocblas_int n,
                                                float* A,
                                                const rocblas_int lda,
                                                const rocblas_stride strideA,
                                                rocblas_int* ipiv,
                                                const rocblas_stride strideP,
                                                rocblas_int* info,
                                                const rocblas_int batch_count)
{
    return rocsolver_getf2_strided_batched_impl<true, float>(handle, m, n, A, lda, strideA, ipiv,
                                                             strideP, info, batch_count);
}

rocblas_status rocsolver_dgetf2_strided_batched(rocblas_handle handle,
                                                const rocblas_int m,
                                                const rocblas_int n,
                                                double* A,
                                                const rocblas_int lda,
                                                const rocblas_stride strideA,
                                                rocblas_int* ipiv,
                                                const rocblas_stride strideP,
                                                rocblas_int* info,
                                                const rocblas_int batch_count)
{
    return rocsolver_getf2_strided_batched_impl<true, double>(handle, m, n, A, lda, strideA, ipiv,
                                                              strideP, info, batch_count);
}

rocblas_status rocsolver_cgetf2_strided_batched(rocblas_handle handle,
                                                const rocblas_int m,
                                                const rocblas_int n,
                                                rocblas_float_complex* A,
                                                const rocblas_int lda,
                                                const rocblas_stride strideA,
                                                rocblas_int* ipiv,
                                                const rocblas_stride strideP,
                                                rocblas_int* info,
                                                const rocblas_int batch_count)
{
    return rocsolver_getf2_strided_batched_impl<true, rocblas_float_complex>(
        handle, m, n, A, lda, strideA, ipiv, strideP, info, batch_count);
}

rocblas_status rocsolver_zgetf2_strided_batched(rocblas_handle handle,
                                                const rocblas_int m,
                                                const rocblas_int n,
                                                rocblas_double_complex* A,
                                                const rocblas_int lda,
                                                const rocblas_stride strideA,
                                                rocblas_int* ipiv,
                                                const rocblas_stride strideP,
                                                rocblas_int* info,
                                                const rocblas_int batch_count)
{
    return rocsolver_getf2_strided_batched_impl<true, rocblas_double_complex>(
        handle, m, n, A, lda, strideA, ipiv, strideP, info, batch_count);
}

rocblas_status rocsolver_sgetf2_npvt_strided_batched(rocblas_handle handle,
                                                     const rocblas_int m,
                                                     const rocblas_int n,
                                                     float* A,
                                                     const rocblas_int lda,
                                                     const rocblas_stride strideA,
                                                     rocblas_int* info,
                                                     const rocblas_int batch_count)
{
    rocblas_int* ipiv = nullptr;
    return rocsolver_getf2_strided_batched_impl<false, float>(handle, m, n, A, lda, strideA, ipiv,
                                                              0, info, batch_count);
}

rocblas_status rocsolver_dgetf2_npvt_strided_batched(rocblas_handle handle,
                                                     const rocblas_int m,
                                                     const rocblas_int n,
                                                     double* A,
                                                     const rocblas_int lda,
                                                     const rocblas_stride strideA,
                                                     rocblas_int* info,
                                                     const rocblas_int batch_count)
{
    rocblas_int* ipiv = nullptr;
    return rocsolver_getf2_strided_batched_impl<false, double>(handle, m, n, A, lda, strideA, ipiv,
                                                               0, info, batch_count);
}

rocblas_status rocsolver_cgetf2_npvt_strided_batched(rocblas_handle handle,
                                                     const rocblas_int m,
                                                     const rocblas_int n,
                                                     rocblas_float_complex* A,
                                                     const rocblas_int lda,
                                                     const rocblas_stride strideA,
                                                     rocblas_int* info,
                                                     const rocblas_int batch_count)
{
    rocblas_int* ipiv = nullptr;
    return rocsolver_getf2_strided_batched_impl<false, rocblas_float_complex>(
        handle, m, n, A, lda, strideA, ipiv, 0, info, batch_count);
}

rocblas_status rocsolver_zgetf2_npvt_strided_batched(rocblas_handle handle,
                                                     const rocblas_int m,
                                                     const rocblas_int n,
                                                     rocblas_double_complex* A,
                                                     const rocblas_int lda,
                                                     const rocblas_stride strideA,
                                                     rocblas_int* info,
                                                     const rocblas_int batch_count)
{
    rocblas_int* ipiv = nullptr;
    return rocsolver_getf2_strided_batched_impl<false, rocblas_double_complex>(
        handle, m, n, A, lda, strideA, ipiv, 0, info, batch_count);
}

} // extern C
