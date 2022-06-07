/* ************************************************************************
 * Copyright (c) 2021-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_sygvd_hegvd.hpp"

template <typename T, typename S, typename U>
rocblas_status rocsolver_sygvd_hegvd_impl(rocblas_handle handle,
                                          const rocblas_eform itype,
                                          const rocblas_evect evect,
                                          const rocblas_fill uplo,
                                          const rocblas_int n,
                                          U A,
                                          const rocblas_int lda,
                                          U B,
                                          const rocblas_int ldb,
                                          S* D,
                                          S* E,
                                          rocblas_int* info)
{
    const char* name = (!rocblas_is_complex<T> ? "sygvd" : "hegvd");
    ROCSOLVER_ENTER_TOP(name, "--itype", itype, "--evect", evect, "--uplo", uplo, "-n", n, "--lda",
                        lda, "--ldb", ldb);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st
        = rocsolver_sygv_hegv_argCheck(handle, itype, evect, uplo, n, lda, ldb, A, B, D, E, info);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;
    rocblas_int shiftB = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideA = 0;
    rocblas_stride strideB = 0;
    rocblas_stride strideD = 0;
    rocblas_stride strideE = 0;
    rocblas_int batch_count = 1;

    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // size of reusable workspaces (and for calling TRSM, SYGST/HEGST, and SYEVD/HEEVD)
    bool optim_mem;
    size_t size_work1, size_work2, size_work3, size_work4;
    // extra requirements for calling POTRF and SYEVD/HEEVD
    size_t size_tau;
    size_t size_pivots_workArr;
    // size of temporary info array
    size_t size_iinfo;
    rocsolver_sygvd_hegvd_getMemorySize<false, false, T, S>(
        itype, evect, uplo, n, batch_count, &size_scalars, &size_work1, &size_work2, &size_work3,
        &size_work4, &size_tau, &size_pivots_workArr, &size_iinfo, &optim_mem);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_scalars, size_work1, size_work2,
                                                      size_work3, size_work4, size_tau,
                                                      size_pivots_workArr, size_iinfo);

    // memory workspace allocation
    void *scalars, *work1, *work2, *work3, *work4, *tau, *pivots_workArr, *iinfo;
    rocblas_device_malloc mem(handle, size_scalars, size_work1, size_work2, size_work3, size_work4,
                              size_tau, size_pivots_workArr, size_iinfo);

    if(!mem)
        return rocblas_status_memory_error;

    scalars = mem[0];
    work1 = mem[1];
    work2 = mem[2];
    work3 = mem[3];
    work4 = mem[4];
    tau = mem[5];
    pivots_workArr = mem[6];
    iinfo = mem[7];
    if(size_scalars > 0)
        init_scalars(handle, (T*)scalars);

    // execution
    return rocsolver_sygvd_hegvd_template<false, false, T>(
        handle, itype, evect, uplo, n, A, shiftA, lda, strideA, B, shiftB, ldb, strideB, D, strideD,
        E, strideE, info, batch_count, (T*)scalars, work1, work2, work3, work4, (T*)tau,
        pivots_workArr, (rocblas_int*)iinfo, optim_mem);
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_ssygvd(rocblas_handle handle,
                                const rocblas_eform itype,
                                const rocblas_evect evect,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                float* A,
                                const rocblas_int lda,
                                float* B,
                                const rocblas_int ldb,
                                float* D,
                                float* E,
                                rocblas_int* info)
{
    return rocsolver_sygvd_hegvd_impl<float>(handle, itype, evect, uplo, n, A, lda, B, ldb, D, E,
                                             info);
}

rocblas_status rocsolver_dsygvd(rocblas_handle handle,
                                const rocblas_eform itype,
                                const rocblas_evect evect,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                double* A,
                                const rocblas_int lda,
                                double* B,
                                const rocblas_int ldb,
                                double* D,
                                double* E,
                                rocblas_int* info)
{
    return rocsolver_sygvd_hegvd_impl<double>(handle, itype, evect, uplo, n, A, lda, B, ldb, D, E,
                                              info);
}

rocblas_status rocsolver_chegvd(rocblas_handle handle,
                                const rocblas_eform itype,
                                const rocblas_evect evect,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                rocblas_float_complex* A,
                                const rocblas_int lda,
                                rocblas_float_complex* B,
                                const rocblas_int ldb,
                                float* D,
                                float* E,
                                rocblas_int* info)
{
    return rocsolver_sygvd_hegvd_impl<rocblas_float_complex>(handle, itype, evect, uplo, n, A, lda,
                                                             B, ldb, D, E, info);
}

rocblas_status rocsolver_zhegvd(rocblas_handle handle,
                                const rocblas_eform itype,
                                const rocblas_evect evect,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                rocblas_double_complex* A,
                                const rocblas_int lda,
                                rocblas_double_complex* B,
                                const rocblas_int ldb,
                                double* D,
                                double* E,
                                rocblas_int* info)
{
    return rocsolver_sygvd_hegvd_impl<rocblas_double_complex>(handle, itype, evect, uplo, n, A, lda,
                                                              B, ldb, D, E, info);
}

} // extern C
