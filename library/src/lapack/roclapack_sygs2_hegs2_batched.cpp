/* ************************************************************************
 * Copyright (c) 2021-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_sygs2_hegs2.hpp"

template <typename T, typename U>
rocblas_status rocsolver_sygs2_hegs2_batched_impl(rocblas_handle handle,
                                                  const rocblas_eform itype,
                                                  const rocblas_fill uplo,
                                                  const rocblas_int n,
                                                  U A,
                                                  const rocblas_int lda,
                                                  U B,
                                                  const rocblas_int ldb,
                                                  const rocblas_int batch_count)
{
    const char* name = (!rocblas_is_complex<T> ? "sygs2_batched" : "hegs2_batched");
    ROCSOLVER_ENTER_TOP(name, "--itype", itype, "--uplo", uplo, "-n", n, "--lda", lda, "--ldb", ldb,
                        "--batch_count", batch_count);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st
        = rocsolver_sygs2_hegs2_argCheck(handle, itype, uplo, n, lda, ldb, A, B, batch_count);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;
    rocblas_int shiftB = 0;

    // batched execution
    rocblas_stride strideA = 0;
    rocblas_stride strideB = 0;

    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // size of reusable workspace (and for calling TRSV or TRMV)
    size_t size_work, size_store_wcs;
    // size of array of pointers (only for batched case)
    size_t size_workArr;
    rocsolver_sygs2_hegs2_getMemorySize<true, T>(itype, n, batch_count, &size_scalars, &size_work,
                                                 &size_store_wcs, &size_workArr);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_scalars, size_work,
                                                      size_store_wcs, size_workArr);

    // memory workspace allocation
    void *scalars, *work, *store_wcs, *workArr;
    rocblas_device_malloc mem(handle, size_scalars, size_work, size_store_wcs, size_workArr);

    if(!mem)
        return rocblas_status_memory_error;

    scalars = mem[0];
    work = mem[1];
    store_wcs = mem[2];
    workArr = mem[3];
    if(size_scalars > 0)
        init_scalars(handle, (T*)scalars);

    // execution
    return rocsolver_sygs2_hegs2_template<true, T>(handle, itype, uplo, n, A, shiftA, lda, strideA,
                                                   B, shiftB, ldb, strideB, batch_count,
                                                   (T*)scalars, work, store_wcs, (T**)workArr);
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_ssygs2_batched(rocblas_handle handle,
                                        const rocblas_eform itype,
                                        const rocblas_fill uplo,
                                        const rocblas_int n,
                                        float* const A[],
                                        const rocblas_int lda,
                                        float* const B[],
                                        const rocblas_int ldb,
                                        const rocblas_int batch_count)
{
    return rocsolver_sygs2_hegs2_batched_impl<float>(handle, itype, uplo, n, A, lda, B, ldb,
                                                     batch_count);
}

rocblas_status rocsolver_dsygs2_batched(rocblas_handle handle,
                                        const rocblas_eform itype,
                                        const rocblas_fill uplo,
                                        const rocblas_int n,
                                        double* const A[],
                                        const rocblas_int lda,
                                        double* const B[],
                                        const rocblas_int ldb,
                                        const rocblas_int batch_count)
{
    return rocsolver_sygs2_hegs2_batched_impl<double>(handle, itype, uplo, n, A, lda, B, ldb,
                                                      batch_count);
}

rocblas_status rocsolver_chegs2_batched(rocblas_handle handle,
                                        const rocblas_eform itype,
                                        const rocblas_fill uplo,
                                        const rocblas_int n,
                                        rocblas_float_complex* const A[],
                                        const rocblas_int lda,
                                        rocblas_float_complex* const B[],
                                        const rocblas_int ldb,
                                        const rocblas_int batch_count)
{
    return rocsolver_sygs2_hegs2_batched_impl<rocblas_float_complex>(handle, itype, uplo, n, A, lda,
                                                                     B, ldb, batch_count);
}

rocblas_status rocsolver_zhegs2_batched(rocblas_handle handle,
                                        const rocblas_eform itype,
                                        const rocblas_fill uplo,
                                        const rocblas_int n,
                                        rocblas_double_complex* const A[],
                                        const rocblas_int lda,
                                        rocblas_double_complex* const B[],
                                        const rocblas_int ldb,
                                        const rocblas_int batch_count)
{
    return rocsolver_sygs2_hegs2_batched_impl<rocblas_double_complex>(handle, itype, uplo, n, A,
                                                                      lda, B, ldb, batch_count);
}

} // extern C
