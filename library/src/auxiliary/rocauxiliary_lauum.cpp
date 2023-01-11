/* ************************************************************************
 * Copyright (c) 2022-2023 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocauxiliary_lauum.hpp"

template <typename T, typename U>
rocblas_status rocsolver_lauum_impl(rocblas_handle handle,
                                    const rocblas_fill uplo,
                                    const rocblas_int n,
                                    U* A,
                                    const rocblas_int lda)
{
    ROCSOLVER_ENTER_TOP("lauum", "--uplo", uplo, "-n", n, "--lda", lda);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_lauum_argCheck(handle, uplo, n, A, lda);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideA = 0;
    rocblas_int batch_count = 1;

    // get and return size of workspace
    size_t size_work;
    rocsolver_lauum_getMemorySize<U>(n, batch_count, &size_work);

    if(rocblas_is_device_memory_size_query(handle))
        rocblas_set_optimal_device_memory_size(handle, size_work);

    void* work;
    rocblas_device_malloc mem(handle, size_work);
    if(!mem)
        return rocblas_status_memory_error;

    work = mem[0];

    // execution
    return rocsolver_lauum_template<T>(handle, uplo, n, A, shiftA, lda, strideA, batch_count,
                                       (U*)work);
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_slauum(rocblas_handle handle,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                float* A,
                                const rocblas_int lda)
{
    return rocsolver_lauum_impl<float>(handle, uplo, n, A, lda);
}

rocblas_status rocsolver_dlauum(rocblas_handle handle,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                double* A,
                                const rocblas_int lda)
{
    return rocsolver_lauum_impl<double>(handle, uplo, n, A, lda);
}

rocblas_status rocsolver_clauum(rocblas_handle handle,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                rocblas_float_complex* A,
                                const rocblas_int lda)
{
    return rocsolver_lauum_impl<rocblas_float_complex>(handle, uplo, n, A, lda);
}

rocblas_status rocsolver_zlauum(rocblas_handle handle,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                rocblas_double_complex* A,
                                const rocblas_int lda)
{
    return rocsolver_lauum_impl<rocblas_double_complex>(handle, uplo, n, A, lda);
}

} // extern C
