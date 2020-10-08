/* ************************************************************************
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_geql2.hpp"

template <typename T, typename U>
rocblas_status rocsolver_geql2_impl(rocblas_handle handle,
                                    const rocblas_int m,
                                    const rocblas_int n,
                                    U A,
                                    const rocblas_int lda,
                                    T* ipiv)
{
    if(!handle)
        return rocblas_status_invalid_handle;

    // logging is missing ???

    // argument checking
    rocblas_status st = rocsolver_geql2_geqlf_argCheck(m, n, lda, A, ipiv);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideA = 0;
    rocblas_stride stridep = 0;
    rocblas_int batch_count = 1;

    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // size of arrays of pointers (for batched cases) and re-usable workspace
    size_t size_work_workArr;
    // extra requirements for calling LARF and LARFG
    size_t size_Abyx_norms;
    // size of temporary array to store diagonal elements
    size_t size_diag;
    rocsolver_geql2_getMemorySize<T, false>(m, n, batch_count, &size_scalars, &size_work_workArr,
                                            &size_Abyx_norms, &size_diag);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_scalars, size_work_workArr,
                                                      size_Abyx_norms, size_diag);

    // memory workspace allocation
    void *scalars, *work_workArr, *Abyx_norms, *diag;
    rocblas_device_malloc mem(handle, size_scalars, size_work_workArr, size_Abyx_norms, size_diag);

    if(!mem)
        return rocblas_status_memory_error;

    scalars = mem[0];
    work_workArr = mem[1];
    Abyx_norms = mem[2];
    diag = mem[3];
    T sca[] = {-1, 0, 1};
    RETURN_IF_HIP_ERROR(hipMemcpy((T*)scalars, sca, size_scalars, hipMemcpyHostToDevice));

    // execution
    return rocsolver_geql2_template<T>(handle, m, n, A, shiftA, lda, strideA, ipiv, stridep,
                                       batch_count, (T*)scalars, work_workArr, (T*)Abyx_norms,
                                       (T*)diag);
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_sgeql2(rocblas_handle handle,
                                const rocblas_int m,
                                const rocblas_int n,
                                float* A,
                                const rocblas_int lda,
                                float* ipiv)
{
    return rocsolver_geql2_impl<float>(handle, m, n, A, lda, ipiv);
}

rocblas_status rocsolver_dgeql2(rocblas_handle handle,
                                const rocblas_int m,
                                const rocblas_int n,
                                double* A,
                                const rocblas_int lda,
                                double* ipiv)
{
    return rocsolver_geql2_impl<double>(handle, m, n, A, lda, ipiv);
}

rocblas_status rocsolver_cgeql2(rocblas_handle handle,
                                const rocblas_int m,
                                const rocblas_int n,
                                rocblas_float_complex* A,
                                const rocblas_int lda,
                                rocblas_float_complex* ipiv)
{
    return rocsolver_geql2_impl<rocblas_float_complex>(handle, m, n, A, lda, ipiv);
}

rocblas_status rocsolver_zgeql2(rocblas_handle handle,
                                const rocblas_int m,
                                const rocblas_int n,
                                rocblas_double_complex* A,
                                const rocblas_int lda,
                                rocblas_double_complex* ipiv)
{
    return rocsolver_geql2_impl<rocblas_double_complex>(handle, m, n, A, lda, ipiv);
}

} // extern C
