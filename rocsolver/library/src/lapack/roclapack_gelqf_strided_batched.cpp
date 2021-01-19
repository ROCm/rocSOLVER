/* ************************************************************************
 * Copyright (c) 2019-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_gelqf.hpp"

template <typename T, typename U>
rocblas_status rocsolver_gelqf_strided_batched_impl(rocblas_handle handle,
                                                    const rocblas_int m,
                                                    const rocblas_int n,
                                                    U A,
                                                    const rocblas_int lda,
                                                    const rocblas_stride strideA,
                                                    T* ipiv,
                                                    const rocblas_stride stridep,
                                                    const rocblas_int batch_count)
{
    ROCSOLVER_ENTER_TOP("gelqf_strided_batched", "-m", m, "-n", n, "--lda", lda, "--bsa", strideA,
                        "--bsp", stridep, "--batch", batch_count);

    if(!handle)
        ROCSOLVER_RETURN_TOP("gelqf_strided_batched", rocblas_status_invalid_handle);

    // argument checking
    rocblas_status st = rocsolver_gelq2_gelqf_argCheck(handle, m, n, lda, A, ipiv, batch_count);
    if(st != rocblas_status_continue)
        ROCSOLVER_RETURN_TOP("gelqf_strided_batched", st);

    // working with unshifted arrays
    rocblas_int shiftA = 0;

    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // size of arrays of pointers (for batched cases) and re-usable workspace
    size_t size_work_workArr, size_workArr;
    // extra requirements for calling GEQR2 and to store temporary triangular factor
    size_t size_Abyx_norms_trfact;
    // extra requirements for calling GEQR2 and LARFB
    size_t size_diag_tmptr;
    rocsolver_gelqf_getMemorySize<T, false>(m, n, batch_count, &size_scalars, &size_work_workArr,
                                            &size_Abyx_norms_trfact, &size_diag_tmptr, &size_workArr);

    if(rocblas_is_device_memory_size_query(handle))
        ROCSOLVER_RETURN_TOP("gelqf_strided_batched",
                             rocblas_set_optimal_device_memory_size(
                                 handle, size_scalars, size_work_workArr, size_Abyx_norms_trfact,
                                 size_diag_tmptr, size_workArr));

    // memory workspace allocation
    void *scalars, *work_workArr, *Abyx_norms_trfact, *diag_tmptr, *workArr;
    rocblas_device_malloc mem(handle, size_scalars, size_work_workArr, size_Abyx_norms_trfact,
                              size_diag_tmptr, size_workArr);

    if(!mem)
        ROCSOLVER_RETURN_TOP("gelqf_strided_batched", rocblas_status_memory_error);

    scalars = mem[0];
    work_workArr = mem[1];
    Abyx_norms_trfact = mem[2];
    diag_tmptr = mem[3];
    workArr = mem[4];
    if(size_scalars > 0)
        init_scalars(handle, (T*)scalars);

    // execution
    ROCSOLVER_RETURN_TOP("gelqf_strided_batched",
                         rocsolver_gelqf_template<false, true, T>(
                             handle, m, n, A, shiftA, lda, strideA, ipiv, stridep, batch_count,
                             (T*)scalars, work_workArr, (T*)Abyx_norms_trfact, (T*)diag_tmptr,
                             (T**)workArr));
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_sgelqf_strided_batched(rocblas_handle handle,
                                                const rocblas_int m,
                                                const rocblas_int n,
                                                float* A,
                                                const rocblas_int lda,
                                                const rocblas_stride strideA,
                                                float* ipiv,
                                                const rocblas_stride stridep,
                                                const rocblas_int batch_count)
{
    return rocsolver_gelqf_strided_batched_impl<float>(handle, m, n, A, lda, strideA, ipiv, stridep,
                                                       batch_count);
}

rocblas_status rocsolver_dgelqf_strided_batched(rocblas_handle handle,
                                                const rocblas_int m,
                                                const rocblas_int n,
                                                double* A,
                                                const rocblas_int lda,
                                                const rocblas_stride strideA,
                                                double* ipiv,
                                                const rocblas_stride stridep,
                                                const rocblas_int batch_count)
{
    return rocsolver_gelqf_strided_batched_impl<double>(handle, m, n, A, lda, strideA, ipiv,
                                                        stridep, batch_count);
}

rocblas_status rocsolver_cgelqf_strided_batched(rocblas_handle handle,
                                                const rocblas_int m,
                                                const rocblas_int n,
                                                rocblas_float_complex* A,
                                                const rocblas_int lda,
                                                const rocblas_stride strideA,
                                                rocblas_float_complex* ipiv,
                                                const rocblas_stride stridep,
                                                const rocblas_int batch_count)
{
    return rocsolver_gelqf_strided_batched_impl<rocblas_float_complex>(handle, m, n, A, lda, strideA,
                                                                       ipiv, stridep, batch_count);
}

rocblas_status rocsolver_zgelqf_strided_batched(rocblas_handle handle,
                                                const rocblas_int m,
                                                const rocblas_int n,
                                                rocblas_double_complex* A,
                                                const rocblas_int lda,
                                                const rocblas_stride strideA,
                                                rocblas_double_complex* ipiv,
                                                const rocblas_stride stridep,
                                                const rocblas_int batch_count)
{
    return rocsolver_gelqf_strided_batched_impl<rocblas_double_complex>(
        handle, m, n, A, lda, strideA, ipiv, stridep, batch_count);
}

} // extern C
