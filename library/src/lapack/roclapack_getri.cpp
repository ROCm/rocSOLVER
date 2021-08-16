/* ************************************************************************
 * Copyright (c) 2019-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_getri.hpp"

template <typename T, typename U>
rocblas_status rocsolver_getri_impl(rocblas_handle handle,
                                    const rocblas_int n,
                                    U A,
                                    const rocblas_int lda,
                                    rocblas_int* ipiv,
                                    rocblas_int* info,
                                    const bool pivot)
{
    const char* name = (pivot ? "getri" : "getri_npvt");
    ROCSOLVER_ENTER_TOP(name, "-n", n, "--lda", lda);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_getri_argCheck(handle, n, lda, A, ipiv, info, pivot);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;
    rocblas_int shiftP = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideA = 0;
    rocblas_stride strideP = 0;
    rocblas_int batch_count = 1;

    // memory workspace sizes:
    // size of reusable workspace (for calling TRSM and TRTRI)
    bool optim_mem;
    size_t size_work1, size_work2, size_work3, size_work4;
    // size of temporary array required for copies
    size_t size_tmpcopy;
    // size of arrays of pointers (for batched cases)
    size_t size_workArr;
    rocsolver_getri_getMemorySize<false, false, T>(n, batch_count, &size_work1, &size_work2,
                                                   &size_work3, &size_work4, &size_tmpcopy,
                                                   &size_workArr, &optim_mem);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_work1, size_work2, size_work3,
                                                      size_work4, size_tmpcopy, size_workArr);

    // memory workspace allocation
    void *work1, *work2, *work3, *work4, *tmpcopy, *workArr;
    rocblas_device_malloc mem(handle, size_work1, size_work2, size_work3, size_work4, size_tmpcopy,
                              size_workArr);

    if(!mem)
        return rocblas_status_memory_error;

    work1 = mem[0];
    work2 = mem[1];
    work3 = mem[2];
    work4 = mem[3];
    tmpcopy = mem[4];
    workArr = mem[5];

    // execution
    return rocsolver_getri_template<false, false, T>(
        handle, n, A, shiftA, lda, strideA, ipiv, shiftP, strideP, info, batch_count, work1, work2,
        work3, work4, (T*)tmpcopy, (T**)workArr, optim_mem);
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_sgetri(rocblas_handle handle,
                                const rocblas_int n,
                                float* A,
                                const rocblas_int lda,
                                rocblas_int* ipiv,
                                rocblas_int* info)
{
    return rocsolver_getri_impl<float>(handle, n, A, lda, ipiv, info, true);
}

rocblas_status rocsolver_dgetri(rocblas_handle handle,
                                const rocblas_int n,
                                double* A,
                                const rocblas_int lda,
                                rocblas_int* ipiv,
                                rocblas_int* info)
{
    return rocsolver_getri_impl<double>(handle, n, A, lda, ipiv, info, true);
}

rocblas_status rocsolver_cgetri(rocblas_handle handle,
                                const rocblas_int n,
                                rocblas_float_complex* A,
                                const rocblas_int lda,
                                rocblas_int* ipiv,
                                rocblas_int* info)
{
    return rocsolver_getri_impl<rocblas_float_complex>(handle, n, A, lda, ipiv, info, true);
}

rocblas_status rocsolver_zgetri(rocblas_handle handle,
                                const rocblas_int n,
                                rocblas_double_complex* A,
                                const rocblas_int lda,
                                rocblas_int* ipiv,
                                rocblas_int* info)
{
    return rocsolver_getri_impl<rocblas_double_complex>(handle, n, A, lda, ipiv, info, true);
}

rocblas_status rocsolver_sgetri_npvt(rocblas_handle handle,
                                     const rocblas_int n,
                                     float* A,
                                     const rocblas_int lda,
                                     rocblas_int* info)
{
    rocblas_int* ipiv = nullptr;
    return rocsolver_getri_impl<float>(handle, n, A, lda, ipiv, info, false);
}

rocblas_status rocsolver_dgetri_npvt(rocblas_handle handle,
                                     const rocblas_int n,
                                     double* A,
                                     const rocblas_int lda,
                                     rocblas_int* info)
{
    rocblas_int* ipiv = nullptr;
    return rocsolver_getri_impl<double>(handle, n, A, lda, ipiv, info, false);
}

rocblas_status rocsolver_cgetri_npvt(rocblas_handle handle,
                                     const rocblas_int n,
                                     rocblas_float_complex* A,
                                     const rocblas_int lda,
                                     rocblas_int* info)
{
    rocblas_int* ipiv = nullptr;
    return rocsolver_getri_impl<rocblas_float_complex>(handle, n, A, lda, ipiv, info, false);
}

rocblas_status rocsolver_zgetri_npvt(rocblas_handle handle,
                                     const rocblas_int n,
                                     rocblas_double_complex* A,
                                     const rocblas_int lda,
                                     rocblas_int* info)
{
    rocblas_int* ipiv = nullptr;
    return rocsolver_getri_impl<rocblas_double_complex>(handle, n, A, lda, ipiv, info, false);
}

} // extern C
