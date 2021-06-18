/* ************************************************************************
 * Copyright (c) 2019-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_gebrd.hpp"

template <typename T, typename S, typename U>
rocblas_status rocsolver_gebrd_impl(rocblas_handle handle,
                                    const rocblas_int m,
                                    const rocblas_int n,
                                    U A,
                                    const rocblas_int lda,
                                    S* D,
                                    S* E,
                                    T* tauq,
                                    T* taup)
{
    ROCSOLVER_ENTER_TOP("gebrd", "-m", m, "-n", n, "--lda", lda);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_gebd2_gebrd_argCheck(handle, m, n, lda, A, D, E, tauq, taup);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;
    rocblas_int shiftX = 0;
    rocblas_int shiftY = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideA = 0;
    rocblas_stride strideD = 0;
    rocblas_stride strideE = 0;
    rocblas_stride strideQ = 0;
    rocblas_stride strideP = 0;
    rocblas_stride strideX = 0;
    rocblas_stride strideY = 0;
    rocblas_int batch_count = 1;

    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // size of arrays of pointers (for batched cases) and re-usable workspace
    size_t size_work_workArr;
    // extra requirements for calling GEDB2 and LABRD
    size_t size_Abyx_norms;
    // size for temporary resulting orthogonal matrices when calling LABRD
    size_t size_X;
    size_t size_Y;
    rocsolver_gebrd_getMemorySize<false, T>(m, n, batch_count, &size_scalars, &size_work_workArr,
                                            &size_Abyx_norms, &size_X, &size_Y);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_scalars, size_work_workArr,
                                                      size_Abyx_norms, size_X, size_Y);

    // memory workspace allocation
    void *scalars, *work_workArr, *Abyx_norms, *X, *Y;
    rocblas_device_malloc mem(handle, size_scalars, size_work_workArr, size_Abyx_norms, size_X,
                              size_Y);

    if(!mem)
        return rocblas_status_memory_error;

    scalars = mem[0];
    work_workArr = mem[1];
    Abyx_norms = mem[2];
    X = mem[3];
    Y = mem[4];
    if(size_scalars > 0)
        init_scalars(handle, (T*)scalars);

    // execution
    return rocsolver_gebrd_template<false, false, T>(
        handle, m, n, A, shiftA, lda, strideA, D, strideD, E, strideE, tauq, strideQ, taup, strideP,
        (T*)X, shiftX, m, strideX, (T*)Y, shiftY, n, strideY, batch_count, (T*)scalars,
        work_workArr, (T*)Abyx_norms);
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_sgebrd(rocblas_handle handle,
                                const rocblas_int m,
                                const rocblas_int n,
                                float* A,
                                const rocblas_int lda,
                                float* D,
                                float* E,
                                float* tauq,
                                float* taup)
{
    return rocsolver_gebrd_impl<float>(handle, m, n, A, lda, D, E, tauq, taup);
}

rocblas_status rocsolver_dgebrd(rocblas_handle handle,
                                const rocblas_int m,
                                const rocblas_int n,
                                double* A,
                                const rocblas_int lda,
                                double* D,
                                double* E,
                                double* tauq,
                                double* taup)
{
    return rocsolver_gebrd_impl<double>(handle, m, n, A, lda, D, E, tauq, taup);
}

rocblas_status rocsolver_cgebrd(rocblas_handle handle,
                                const rocblas_int m,
                                const rocblas_int n,
                                rocblas_float_complex* A,
                                const rocblas_int lda,
                                float* D,
                                float* E,
                                rocblas_float_complex* tauq,
                                rocblas_float_complex* taup)
{
    return rocsolver_gebrd_impl<rocblas_float_complex>(handle, m, n, A, lda, D, E, tauq, taup);
}

rocblas_status rocsolver_zgebrd(rocblas_handle handle,
                                const rocblas_int m,
                                const rocblas_int n,
                                rocblas_double_complex* A,
                                const rocblas_int lda,
                                double* D,
                                double* E,
                                rocblas_double_complex* tauq,
                                rocblas_double_complex* taup)
{
    return rocsolver_gebrd_impl<rocblas_double_complex>(handle, m, n, A, lda, D, E, tauq, taup);
}

} // extern C
