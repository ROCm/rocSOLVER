/* ************************************************************************
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocauxiliary_larfb.hpp"

template <typename T>
rocblas_status rocsolver_larfb_impl(rocblas_handle handle,
                                    const rocblas_side side,
                                    const rocblas_operation trans,
                                    const rocblas_direct direct,
                                    const rocblas_storev storev,
                                    const rocblas_int m,
                                    const rocblas_int n,
                                    const rocblas_int k,
                                    T* V,
                                    const rocblas_int ldv,
                                    T* F,
                                    const rocblas_int ldf,
                                    T* A,
                                    const rocblas_int lda)
{
    if(!handle)
        return rocblas_status_invalid_handle;

    // logging is missing ???

    // argument checking
    rocblas_status st
        = rocsolver_larfb_argCheck(side, trans, direct, storev, m, n, k, ldv, ldf, lda, V, A, F);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftV = 0;
    rocblas_int shiftA = 0;
    rocblas_int shiftF = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride stridev = 0;
    rocblas_stride stridea = 0;
    rocblas_stride stridef = 0;
    rocblas_int batch_count = 1;

    // memory workspace sizes:
    // size of re-usable workspace
    size_t size_work;
    // size of array for temporary computations with
    // triangular part of V
    size_t size_tmptr;
    // size of arrays of pointers (for batched cases)
    size_t size_workArr;
    rocsolver_larfb_getMemorySize<T, false>(side, m, n, k, batch_count, &size_work, &size_tmptr,
                                            &size_workArr);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_work, size_tmptr, size_workArr);

    // memory workspace allocation
    void *tmptr, *work, *workArr;
    rocblas_device_malloc mem(handle, size_work, size_tmptr, size_workArr);
    if(!mem)
        return rocblas_status_memory_error;

    work = mem[0];
    tmptr = mem[1];
    workArr = mem[2];

    //  execution
    return rocsolver_larfb_template<false, false, T>(
        handle, side, trans, direct, storev, m, n, k, V, shiftV, ldv, stridev, F, shiftF, ldf,
        stridef, A, shiftA, lda, stridea, batch_count, (T*)work, (T*)tmptr, (T**)workArr);
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_slarfb(rocblas_handle handle,
                                const rocblas_side side,
                                const rocblas_operation trans,
                                const rocblas_direct direct,
                                const rocblas_storev storev,
                                const rocblas_int m,
                                const rocblas_int n,
                                const rocblas_int k,
                                float* V,
                                const rocblas_int ldv,
                                float* T,
                                const rocblas_int ldt,
                                float* A,
                                const rocblas_int lda)
{
    return rocsolver_larfb_impl<float>(handle, side, trans, direct, storev, m, n, k, V, ldv, T, ldt,
                                       A, lda);
}

rocblas_status rocsolver_dlarfb(rocblas_handle handle,
                                const rocblas_side side,
                                const rocblas_operation trans,
                                const rocblas_direct direct,
                                const rocblas_storev storev,
                                const rocblas_int m,
                                const rocblas_int n,
                                const rocblas_int k,
                                double* V,
                                const rocblas_int ldv,
                                double* T,
                                const rocblas_int ldt,
                                double* A,
                                const rocblas_int lda)
{
    return rocsolver_larfb_impl<double>(handle, side, trans, direct, storev, m, n, k, V, ldv, T,
                                        ldt, A, lda);
}

rocblas_status rocsolver_clarfb(rocblas_handle handle,
                                const rocblas_side side,
                                const rocblas_operation trans,
                                const rocblas_direct direct,
                                const rocblas_storev storev,
                                const rocblas_int m,
                                const rocblas_int n,
                                const rocblas_int k,
                                rocblas_float_complex* V,
                                const rocblas_int ldv,
                                rocblas_float_complex* T,
                                const rocblas_int ldt,
                                rocblas_float_complex* A,
                                const rocblas_int lda)
{
    return rocsolver_larfb_impl<rocblas_float_complex>(handle, side, trans, direct, storev, m, n, k,
                                                       V, ldv, T, ldt, A, lda);
}

rocblas_status rocsolver_zlarfb(rocblas_handle handle,
                                const rocblas_side side,
                                const rocblas_operation trans,
                                const rocblas_direct direct,
                                const rocblas_storev storev,
                                const rocblas_int m,
                                const rocblas_int n,
                                const rocblas_int k,
                                rocblas_double_complex* V,
                                const rocblas_int ldv,
                                rocblas_double_complex* T,
                                const rocblas_int ldt,
                                rocblas_double_complex* A,
                                const rocblas_int lda)
{
    return rocsolver_larfb_impl<rocblas_double_complex>(handle, side, trans, direct, storev, m, n,
                                                        k, V, ldv, T, ldt, A, lda);
}

} // extern C
