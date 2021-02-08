/* ************************************************************************
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_sygv_hegv.hpp"

template <typename S, typename T, typename U>
rocblas_status rocsolver_sygv_hegv_impl(rocblas_handle handle,
                                        const rocblas_eform itype,
                                        const rocblas_evect jobz,
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
    const char* name = (!is_complex<T> ? "sygv" : "hegv");
    ROCSOLVER_ENTER_TOP(name, "--itype", itype, "--evect", jobz, "--uplo", uplo, "-n", n, "--lda",
                        lda, "--ldb", ldb);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st
        = rocsolver_sygv_hegv_argCheck(handle, itype, jobz, uplo, n, lda, ldb, A, B, D, E, info);
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
    rocsolver_sygv_hegv_getMemorySize<T, false>(itype, jobz, n, batch_count, &size_scalars);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_scalars);

    // memory workspace allocation
    void* scalars;
    rocblas_device_malloc mem(handle, size_scalars);

    if(!mem)
        return rocblas_status_memory_error;

    scalars = mem[0];
    if(size_scalars > 0)
        init_scalars(handle, (T*)scalars);

    // execution
    return rocsolver_sygv_hegv_template<false, false, S, T>(handle, itype, jobz, uplo, n, A, shiftA,
                                                            lda, strideA, B, shiftB, ldb, strideB, D,
                                                            strideD, E, strideE, info, batch_count, (T*)scalars);
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_ssygv(rocblas_handle handle,
                               const rocblas_eform itype,
                               const rocblas_evect jobz,
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
    return rocsolver_sygv_hegv_impl<float, float>(handle, itype, jobz, uplo, n, A, lda, B, ldb, D, E,
                                                  info);
}

rocblas_status rocsolver_dsygv(rocblas_handle handle,
                               const rocblas_eform itype,
                               const rocblas_evect jobz,
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
    return rocsolver_sygv_hegv_impl<double, double>(handle, itype, jobz, uplo, n, A, lda, B, ldb, D, E,
                                                    info);
}

rocblas_status rocsolver_chegv(rocblas_handle handle,
                               const rocblas_eform itype,
                               const rocblas_evect jobz,
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
    return rocsolver_sygv_hegv_impl<float, rocblas_float_complex>(handle, itype, jobz, uplo, n, A,
                                                                  lda, B, ldb, D, E, info);
}

rocblas_status rocsolver_zhegv(rocblas_handle handle,
                               const rocblas_eform itype,
                               const rocblas_evect jobz,
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
    return rocsolver_sygv_hegv_impl<double, rocblas_double_complex>(handle, itype, jobz, uplo, n, A,
                                                                    lda, B, ldb, D, E, info);
}

} // extern C
