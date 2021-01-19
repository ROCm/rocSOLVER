/* ************************************************************************
 * Copyright (c) 2019-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_getrf.hpp"

template <bool PIVOT, typename T, typename U>
rocblas_status rocsolver_getrf_impl(rocblas_handle handle,
                                    const rocblas_int m,
                                    const rocblas_int n,
                                    U A,
                                    const rocblas_int lda,
                                    rocblas_int* ipiv,
                                    rocblas_int* info)
{
    const char* name = (PIVOT ? "getrf" : "getrf_npvt");
    ROCSOLVER_ENTER_TOP(name, "-m", m, "-n", n, "--lda", lda);

    using S = decltype(std::real(T{}));

    if(!handle)
        ROCSOLVER_RETURN_TOP(name, rocblas_status_invalid_handle);

    // argument checking
    rocblas_status st = rocsolver_getf2_getrf_argCheck(handle, m, n, lda, A, ipiv, info, PIVOT);
    if(st != rocblas_status_continue)
        ROCSOLVER_RETURN_TOP(name, st);

    // working with unshifted arrays
    rocblas_int shiftA = 0;
    rocblas_int shiftP = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideA = 0;
    rocblas_stride strideP = 0;
    rocblas_int batch_count = 1;

    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // size of reusable workspace (and for calling TRSM)
    size_t size_work, size_work1, size_work2, size_work3, size_work4;
    // extra requirements for calling GETF2
    size_t size_pivotval, size_pivotidx;
    // size to store info about singularity of each subblock
    size_t size_iinfo;
    rocsolver_getrf_getMemorySize<false, false, PIVOT, T, S>(
        m, n, batch_count, &size_scalars, &size_work, &size_work1, &size_work2, &size_work3,
        &size_work4, &size_pivotval, &size_pivotidx, &size_iinfo);

    if(rocblas_is_device_memory_size_query(handle))
        ROCSOLVER_RETURN_TOP(name,
                             rocblas_set_optimal_device_memory_size(
                                 handle, size_scalars, size_work, size_work1, size_work2,
                                 size_work3, size_work4, size_pivotval, size_pivotidx, size_iinfo));

    // always allocate all required memory for TRSM optimal performance
    bool optim_mem = true;

    // memory workspace allocation
    void *scalars, *work, *work1, *work2, *work3, *work4, *pivotval, *pivotidx, *iinfo;
    rocblas_device_malloc mem(handle, size_scalars, size_work, size_work1, size_work2, size_work3,
                              size_work4, size_pivotval, size_pivotidx, size_iinfo);

    if(!mem)
        ROCSOLVER_RETURN_TOP(name, rocblas_status_memory_error);

    scalars = mem[0];
    work = mem[1];
    work1 = mem[2];
    work2 = mem[3];
    work3 = mem[4];
    work4 = mem[5];
    pivotval = mem[6];
    pivotidx = mem[7];
    iinfo = mem[8];
    if(size_scalars > 0)
        init_scalars(handle, (T*)scalars);

    // execution
    ROCSOLVER_RETURN_TOP(name,
                         rocsolver_getrf_template<false, false, PIVOT, T, S>(
                             handle, m, n, A, shiftA, lda, strideA, ipiv, shiftP, strideP, info,
                             batch_count, (T*)scalars, (rocblas_index_value_t<S>*)work, work1,
                             work2, work3, work4, (T*)pivotval, (rocblas_int*)pivotidx,
                             (rocblas_int*)iinfo, optim_mem));
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_sgetrf(rocblas_handle handle,
                                const rocblas_int m,
                                const rocblas_int n,
                                float* A,
                                const rocblas_int lda,
                                rocblas_int* ipiv,
                                rocblas_int* info)
{
    return rocsolver_getrf_impl<true, float>(handle, m, n, A, lda, ipiv, info);
}

rocblas_status rocsolver_dgetrf(rocblas_handle handle,
                                const rocblas_int m,
                                const rocblas_int n,
                                double* A,
                                const rocblas_int lda,
                                rocblas_int* ipiv,
                                rocblas_int* info)
{
    return rocsolver_getrf_impl<true, double>(handle, m, n, A, lda, ipiv, info);
}

rocblas_status rocsolver_cgetrf(rocblas_handle handle,
                                const rocblas_int m,
                                const rocblas_int n,
                                rocblas_float_complex* A,
                                const rocblas_int lda,
                                rocblas_int* ipiv,
                                rocblas_int* info)
{
    return rocsolver_getrf_impl<true, rocblas_float_complex>(handle, m, n, A, lda, ipiv, info);
}

rocblas_status rocsolver_zgetrf(rocblas_handle handle,
                                const rocblas_int m,
                                const rocblas_int n,
                                rocblas_double_complex* A,
                                const rocblas_int lda,
                                rocblas_int* ipiv,
                                rocblas_int* info)
{
    return rocsolver_getrf_impl<true, rocblas_double_complex>(handle, m, n, A, lda, ipiv, info);
}

rocblas_status rocsolver_sgetrf_npvt(rocblas_handle handle,
                                     const rocblas_int m,
                                     const rocblas_int n,
                                     float* A,
                                     const rocblas_int lda,
                                     rocblas_int* info)
{
    rocblas_int* ipiv = nullptr;
    return rocsolver_getrf_impl<false, float>(handle, m, n, A, lda, ipiv, info);
}

rocblas_status rocsolver_dgetrf_npvt(rocblas_handle handle,
                                     const rocblas_int m,
                                     const rocblas_int n,
                                     double* A,
                                     const rocblas_int lda,
                                     rocblas_int* info)
{
    rocblas_int* ipiv = nullptr;
    return rocsolver_getrf_impl<false, double>(handle, m, n, A, lda, ipiv, info);
}

rocblas_status rocsolver_cgetrf_npvt(rocblas_handle handle,
                                     const rocblas_int m,
                                     const rocblas_int n,
                                     rocblas_float_complex* A,
                                     const rocblas_int lda,
                                     rocblas_int* info)
{
    rocblas_int* ipiv = nullptr;
    return rocsolver_getrf_impl<false, rocblas_float_complex>(handle, m, n, A, lda, ipiv, info);
}

rocblas_status rocsolver_zgetrf_npvt(rocblas_handle handle,
                                     const rocblas_int m,
                                     const rocblas_int n,
                                     rocblas_double_complex* A,
                                     const rocblas_int lda,
                                     rocblas_int* info)
{
    rocblas_int* ipiv = nullptr;
    return rocsolver_getrf_impl<false, rocblas_double_complex>(handle, m, n, A, lda, ipiv, info);
}

} // extern C
