/* ************************************************************************
 * Copyright (c) 2019-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_getf2.hpp"

template <typename T, typename U>
rocblas_status rocsolver_getf2_batched_impl(rocblas_handle handle,
                                            const rocblas_int m,
                                            const rocblas_int n,
                                            U A,
                                            const rocblas_int lda,
                                            rocblas_int* ipiv,
                                            const rocblas_stride strideP,
                                            rocblas_int* info,
                                            const bool pivot,
                                            const rocblas_int batch_count)
{
    const char* name = (pivot ? "getf2_batched" : "getf2_npvt_batched");
    ROCSOLVER_ENTER_TOP(name, "-m", m, "-n", n, "--lda", lda, "--strideP", strideP, "--batch_count",
                        batch_count);

    using S = decltype(std::real(T{}));

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st
        = rocsolver_getf2_getrf_argCheck(handle, m, n, lda, A, ipiv, info, pivot, batch_count);
    if(st != rocblas_status_continue)
        return st;

    // using unshifted arrays
    rocblas_int shiftA = 0;
    rocblas_int shiftP = 0;

    // batched execution
    rocblas_stride strideA = 0;

    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // sizes to store pivots in intermediate computations
    size_t size_pivotval;
    size_t size_pivotidx;
    rocsolver_getf2_getMemorySize<true, T>(m, n, pivot, batch_count, &size_scalars, &size_pivotval,
                                           &size_pivotidx);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_scalars, size_pivotval,
                                                      size_pivotidx);

    // memory workspace allocation
    void *scalars, *pivotidx, *pivotval;
    rocblas_device_malloc mem(handle, size_scalars, size_pivotval, size_pivotidx);

    if(!mem)
        return rocblas_status_memory_error;

    scalars = mem[0];
    pivotval = mem[1];
    pivotidx = mem[2];
    if(size_scalars > 0)
        init_scalars(handle, (T*)scalars);

    // execution
    return rocsolver_getf2_template<true, T>(handle, m, n, A, shiftA, lda, strideA, ipiv, shiftP,
                                             strideP, info, batch_count, (T*)scalars, (T*)pivotval,
                                             (rocblas_int*)pivotidx, pivot);
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_sgetf2_batched(rocblas_handle handle,
                                        const rocblas_int m,
                                        const rocblas_int n,
                                        float* const A[],
                                        const rocblas_int lda,
                                        rocblas_int* ipiv,
                                        const rocblas_stride strideP,
                                        rocblas_int* info,
                                        const rocblas_int batch_count)
{
    return rocsolver_getf2_batched_impl<float>(handle, m, n, A, lda, ipiv, strideP, info, true,
                                               batch_count);
}

rocblas_status rocsolver_dgetf2_batched(rocblas_handle handle,
                                        const rocblas_int m,
                                        const rocblas_int n,
                                        double* const A[],
                                        const rocblas_int lda,
                                        rocblas_int* ipiv,
                                        const rocblas_stride strideP,
                                        rocblas_int* info,
                                        const rocblas_int batch_count)
{
    return rocsolver_getf2_batched_impl<double>(handle, m, n, A, lda, ipiv, strideP, info, true,
                                                batch_count);
}

rocblas_status rocsolver_cgetf2_batched(rocblas_handle handle,
                                        const rocblas_int m,
                                        const rocblas_int n,
                                        rocblas_float_complex* const A[],
                                        const rocblas_int lda,
                                        rocblas_int* ipiv,
                                        const rocblas_stride strideP,
                                        rocblas_int* info,
                                        const rocblas_int batch_count)
{
    return rocsolver_getf2_batched_impl<rocblas_float_complex>(handle, m, n, A, lda, ipiv, strideP,
                                                               info, true, batch_count);
}

rocblas_status rocsolver_zgetf2_batched(rocblas_handle handle,
                                        const rocblas_int m,
                                        const rocblas_int n,
                                        rocblas_double_complex* const A[],
                                        const rocblas_int lda,
                                        rocblas_int* ipiv,
                                        const rocblas_stride strideP,
                                        rocblas_int* info,
                                        const rocblas_int batch_count)
{
    return rocsolver_getf2_batched_impl<rocblas_double_complex>(handle, m, n, A, lda, ipiv, strideP,
                                                                info, true, batch_count);
}

rocblas_status rocsolver_sgetf2_npvt_batched(rocblas_handle handle,
                                             const rocblas_int m,
                                             const rocblas_int n,
                                             float* const A[],
                                             const rocblas_int lda,
                                             rocblas_int* info,
                                             const rocblas_int batch_count)
{
    rocblas_int* ipiv = nullptr;
    return rocsolver_getf2_batched_impl<float>(handle, m, n, A, lda, ipiv, 0, info, false,
                                               batch_count);
}

rocblas_status rocsolver_dgetf2_npvt_batched(rocblas_handle handle,
                                             const rocblas_int m,
                                             const rocblas_int n,
                                             double* const A[],
                                             const rocblas_int lda,
                                             rocblas_int* info,
                                             const rocblas_int batch_count)
{
    rocblas_int* ipiv = nullptr;
    return rocsolver_getf2_batched_impl<double>(handle, m, n, A, lda, ipiv, 0, info, false,
                                                batch_count);
}

rocblas_status rocsolver_cgetf2_npvt_batched(rocblas_handle handle,
                                             const rocblas_int m,
                                             const rocblas_int n,
                                             rocblas_float_complex* const A[],
                                             const rocblas_int lda,
                                             rocblas_int* info,
                                             const rocblas_int batch_count)
{
    rocblas_int* ipiv = nullptr;
    return rocsolver_getf2_batched_impl<rocblas_float_complex>(handle, m, n, A, lda, ipiv, 0, info,
                                                               false, batch_count);
}

rocblas_status rocsolver_zgetf2_npvt_batched(rocblas_handle handle,
                                             const rocblas_int m,
                                             const rocblas_int n,
                                             rocblas_double_complex* const A[],
                                             const rocblas_int lda,
                                             rocblas_int* info,
                                             const rocblas_int batch_count)
{
    rocblas_int* ipiv = nullptr;
    return rocsolver_getf2_batched_impl<rocblas_double_complex>(handle, m, n, A, lda, ipiv, 0, info,
                                                                false, batch_count);
}

} // extern C
