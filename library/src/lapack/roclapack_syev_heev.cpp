/* ************************************************************************
 * Copyright (c) 2021-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_syev_heev.hpp"

template <typename T, typename S, typename W>
rocblas_status rocsolver_syev_heev_impl(rocblas_handle handle,
                                        const rocblas_evect evect,
                                        const rocblas_fill uplo,
                                        const rocblas_int n,
                                        W A,
                                        const rocblas_int lda,
                                        S* D,
                                        S* E,
                                        rocblas_int* info)
{
    const char* name = (!rocblas_is_complex<T> ? "syev" : "heev");
    ROCSOLVER_ENTER_TOP(name, "--evect", evect, "--uplo", uplo, "-n", n, "--lda", lda);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_syev_heev_argCheck(handle, evect, uplo, n, A, lda, D, E, info);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideA = 0;
    rocblas_stride strideD = 0;
    rocblas_stride strideE = 0;
    rocblas_int batch_count = 1;

    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // size of reusable workspace and stack
    size_t size_work_stack;
    // extra requirements to call SYTRD/HETRD and ORGTR/UNGTR
    size_t size_Abyx_norms_tmptr, size_tmptau_trfact;
    // size of array of pointers (only for batched case)
    size_t size_workArr;
    // size for temporary householder scalars
    size_t size_tau;

    rocsolver_syev_heev_getMemorySize<false, T, S>(evect, uplo, n, batch_count, &size_scalars,
                                                   &size_work_stack, &size_Abyx_norms_tmptr,
                                                   &size_tmptau_trfact, &size_tau, &size_workArr);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_scalars, size_work_stack,
                                                      size_Abyx_norms_tmptr, size_tmptau_trfact,
                                                      size_tau, size_workArr);

    // memory workspace allocation
    void *scalars, *work_stack, *Abyx_norms_tmptr, *tmptau_trfact, *tau, *workArr;
    rocblas_device_malloc mem(handle, size_scalars, size_work_stack, size_Abyx_norms_tmptr,
                              size_tmptau_trfact, size_tau, size_workArr);

    if(!mem)
        return rocblas_status_memory_error;

    scalars = mem[0];
    work_stack = mem[1];
    Abyx_norms_tmptr = mem[2];
    tmptau_trfact = mem[3];
    tau = mem[4];
    workArr = mem[5];
    if(size_scalars > 0)
        init_scalars(handle, (T*)scalars);

    // execution
    return rocsolver_syev_heev_template<false, false, T>(
        handle, evect, uplo, n, A, shiftA, lda, strideA, D, strideD, E, strideE, info, batch_count,
        (T*)scalars, work_stack, (T*)Abyx_norms_tmptr, (T*)tmptau_trfact, (T*)tau, (T**)workArr);
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_ssyev(rocblas_handle handle,
                               const rocblas_evect evect,
                               const rocblas_fill uplo,
                               const rocblas_int n,
                               float* A,
                               const rocblas_int lda,
                               float* D,
                               float* E,
                               rocblas_int* info)
{
    return rocsolver_syev_heev_impl<float>(handle, evect, uplo, n, A, lda, D, E, info);
}

rocblas_status rocsolver_dsyev(rocblas_handle handle,
                               const rocblas_evect evect,
                               const rocblas_fill uplo,
                               const rocblas_int n,
                               double* A,
                               const rocblas_int lda,
                               double* D,
                               double* E,
                               rocblas_int* info)
{
    return rocsolver_syev_heev_impl<double>(handle, evect, uplo, n, A, lda, D, E, info);
}

rocblas_status rocsolver_cheev(rocblas_handle handle,
                               const rocblas_evect evect,
                               const rocblas_fill uplo,
                               const rocblas_int n,
                               rocblas_float_complex* A,
                               const rocblas_int lda,
                               float* D,
                               float* E,
                               rocblas_int* info)
{
    return rocsolver_syev_heev_impl<rocblas_float_complex>(handle, evect, uplo, n, A, lda, D, E,
                                                           info);
}

rocblas_status rocsolver_zheev(rocblas_handle handle,
                               const rocblas_evect evect,
                               const rocblas_fill uplo,
                               const rocblas_int n,
                               rocblas_double_complex* A,
                               const rocblas_int lda,
                               double* D,
                               double* E,
                               rocblas_int* info)
{
    return rocsolver_syev_heev_impl<rocblas_double_complex>(handle, evect, uplo, n, A, lda, D, E,
                                                            info);
}

} // extern C
