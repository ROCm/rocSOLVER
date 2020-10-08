/* ************************************************************************
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocauxiliary_labrd.hpp"

template <typename S, typename T, typename U>
rocblas_status rocsolver_labrd_impl(rocblas_handle handle,
                                    const rocblas_int m,
                                    const rocblas_int n,
                                    const rocblas_int k,
                                    U A,
                                    const rocblas_int lda,
                                    S* D,
                                    S* E,
                                    T* tauq,
                                    T* taup,
                                    U X,
                                    const rocblas_int ldx,
                                    U Y,
                                    const rocblas_int ldy)
{
    if(!handle)
        return rocblas_status_invalid_handle;

    // logging is missing ???

    // argument checking
    rocblas_status st = rocsolver_labrd_argCheck(m, n, k, lda, ldx, ldy, A, D, E, tauq, taup, X, Y);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;
    rocblas_int shiftX = 0;
    rocblas_int shiftY = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideA = 0;
    rocblas_stride strideX = 0;
    rocblas_stride strideY = 0;
    rocblas_stride strideD = 0;
    rocblas_stride strideE = 0;
    rocblas_stride strideQ = 0;
    rocblas_stride strideP = 0;
    rocblas_int batch_count = 1;

    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // size of arrays of pointers (for batched cases) and re-usable workspace
    size_t size_work_workArr;
    // extra requirements for calling LARFG
    size_t size_norms;
    rocsolver_labrd_getMemorySize<T, false>(m, n, k, batch_count, &size_scalars, &size_work_workArr,
                                            &size_norms);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_scalars, size_work_workArr,
                                                      size_norms);

    // memory workspace allocation
    void *scalars, *work_workArr, *norms;
    rocblas_device_malloc mem(handle, size_scalars, size_work_workArr, size_norms);

    if(!mem)
        return rocblas_status_memory_error;

    scalars = mem[0];
    work_workArr = mem[1];
    norms = mem[2];
    T sca[] = {-1, 0, 1};
    RETURN_IF_HIP_ERROR(hipMemcpy((T*)scalars, sca, size_scalars, hipMemcpyHostToDevice));

    // execution
    return rocsolver_labrd_template<S, T>(handle, m, n, k, A, shiftA, lda, strideA, D, strideD, E,
                                          strideE, tauq, strideQ, taup, strideP, X, shiftX, ldx,
                                          strideX, Y, shiftY, ldy, strideY, batch_count,
                                          (T*)scalars, work_workArr, (T*)norms);
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_slabrd(rocblas_handle handle,
                                const rocblas_int m,
                                const rocblas_int n,
                                const rocblas_int k,
                                float* A,
                                const rocblas_int lda,
                                float* D,
                                float* E,
                                float* tauq,
                                float* taup,
                                float* X,
                                const rocblas_int ldx,
                                float* Y,
                                const rocblas_int ldy)
{
    return rocsolver_labrd_impl<float, float>(handle, m, n, k, A, lda, D, E, tauq, taup, X, ldx, Y,
                                              ldy);
}

rocblas_status rocsolver_dlabrd(rocblas_handle handle,
                                const rocblas_int m,
                                const rocblas_int n,
                                const rocblas_int k,
                                double* A,
                                const rocblas_int lda,
                                double* D,
                                double* E,
                                double* tauq,
                                double* taup,
                                double* X,
                                const rocblas_int ldx,
                                double* Y,
                                const rocblas_int ldy)
{
    return rocsolver_labrd_impl<double, double>(handle, m, n, k, A, lda, D, E, tauq, taup, X, ldx,
                                                Y, ldy);
}

rocblas_status rocsolver_clabrd(rocblas_handle handle,
                                const rocblas_int m,
                                const rocblas_int n,
                                const rocblas_int k,
                                rocblas_float_complex* A,
                                const rocblas_int lda,
                                float* D,
                                float* E,
                                rocblas_float_complex* tauq,
                                rocblas_float_complex* taup,
                                rocblas_float_complex* X,
                                const rocblas_int ldx,
                                rocblas_float_complex* Y,
                                const rocblas_int ldy)
{
    return rocsolver_labrd_impl<float, rocblas_float_complex>(handle, m, n, k, A, lda, D, E, tauq,
                                                              taup, X, ldx, Y, ldy);
}

rocblas_status rocsolver_zlabrd(rocblas_handle handle,
                                const rocblas_int m,
                                const rocblas_int n,
                                const rocblas_int k,
                                rocblas_double_complex* A,
                                const rocblas_int lda,
                                double* D,
                                double* E,
                                rocblas_double_complex* tauq,
                                rocblas_double_complex* taup,
                                rocblas_double_complex* X,
                                const rocblas_int ldx,
                                rocblas_double_complex* Y,
                                const rocblas_int ldy)
{
    return rocsolver_labrd_impl<double, rocblas_double_complex>(handle, m, n, k, A, lda, D, E, tauq,
                                                                taup, X, ldx, Y, ldy);
}

} // extern C
