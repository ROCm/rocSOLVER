/* ************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 * *************************************************************************/

#include "roclapack_syevdj_heevdj.hpp"

template <typename T, typename S, typename W>
rocblas_status rocsolver_syevdj_heevdj_impl(rocblas_handle handle,
                                            const rocblas_evect evect,
                                            const rocblas_fill uplo,
                                            const rocblas_int n,
                                            W A,
                                            const rocblas_int lda,
                                            S* D,
                                            rocblas_int* info)
{
    const char* name = (!rocblas_is_complex<T> ? "syevdj" : "heevdj");
    ROCSOLVER_ENTER_TOP(name, "--evect", evect, "--uplo", uplo, "-n", n, "--lda", lda);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_syevdj_heevdj_argCheck(handle, evect, uplo, n, A, lda, D, info);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideA = 0;
    rocblas_stride strideD = 0;
    rocblas_int batch_count = 1;

    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // size of reusable workspaces
    size_t size_work1;
    size_t size_work2;
    size_t size_work3;
    // extra space for call stedc
    size_t size_workSplits, size_work4;
    // size of array of pointers (only for batched case)
    size_t size_workArr;
    // size for temporary householder scalars
    size_t size_workTau;
    // size for temporary vectors
    size_t size_workVec;
    // size for temporary superdiagonal of tridiag form
    size_t size_workE;

    rocsolver_syevdj_heevdj_getMemorySize<false, T, S>(
        evect, uplo, n, batch_count, &size_scalars, &size_workE, &size_workTau, &size_workVec,
        &size_workSplits, &size_work1, &size_work2, &size_work3, &size_work4, &size_workArr);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(
            handle, size_scalars, size_workE, size_workTau, size_workVec, size_workSplits,
            size_work1, size_work2, size_work3, size_work4, size_workArr);

    // memory workspace allocation
    void *scalars, *work1, *work2, *work3, *work4, *workE, *workVec, *workSplits, *workTau, *workArr;
    rocblas_device_malloc mem(handle, size_scalars, size_workE, size_workTau, size_workVec,
                              size_workSplits, size_work1, size_work2, size_work3, size_work4,
                              size_workArr);

    if(!mem)
        return rocblas_status_memory_error;

    scalars = mem[0];
    workE = mem[1];
    workTau = mem[2];
    workVec = mem[3];
    workSplits = mem[4];
    work1 = mem[5];
    work2 = mem[6];
    work3 = mem[7];
    work4 = mem[8];
    workArr = mem[9];
    if(size_scalars > 0)
        init_scalars(handle, (T*)scalars);

    // execution
    return rocsolver_syevdj_heevdj_template<false, false, T>(
        handle, evect, uplo, n, A, shiftA, lda, strideA, D, strideD, info, batch_count, (T*)scalars,
        (S*)workE, (T*)workTau, (T*)workVec, (rocblas_int*)workSplits, work1, work2, work3, work4,
        workArr);
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_ssyevdj(rocblas_handle handle,
                                 const rocblas_evect evect,
                                 const rocblas_fill uplo,
                                 const rocblas_int n,
                                 float* A,
                                 const rocblas_int lda,
                                 float* D,
                                 rocblas_int* info)
{
    return rocsolver_syevdj_heevdj_impl<float>(handle, evect, uplo, n, A, lda, D, info);
}

rocblas_status rocsolver_dsyevdj(rocblas_handle handle,
                                 const rocblas_evect evect,
                                 const rocblas_fill uplo,
                                 const rocblas_int n,
                                 double* A,
                                 const rocblas_int lda,
                                 double* D,
                                 rocblas_int* info)
{
    return rocsolver_syevdj_heevdj_impl<double>(handle, evect, uplo, n, A, lda, D, info);
}

rocblas_status rocsolver_cheevdj(rocblas_handle handle,
                                 const rocblas_evect evect,
                                 const rocblas_fill uplo,
                                 const rocblas_int n,
                                 rocblas_float_complex* A,
                                 const rocblas_int lda,
                                 float* D,
                                 rocblas_int* info)
{
    return rocsolver_syevdj_heevdj_impl<rocblas_float_complex>(handle, evect, uplo, n, A, lda, D,
                                                               info);
}

rocblas_status rocsolver_zheevdj(rocblas_handle handle,
                                 const rocblas_evect evect,
                                 const rocblas_fill uplo,
                                 const rocblas_int n,
                                 rocblas_double_complex* A,
                                 const rocblas_int lda,
                                 double* D,
                                 rocblas_int* info)
{
    return rocsolver_syevdj_heevdj_impl<rocblas_double_complex>(handle, evect, uplo, n, A, lda, D,
                                                                info);
}

} // extern C
