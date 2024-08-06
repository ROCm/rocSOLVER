/* **************************************************************************
 * Copyright (C) 2019-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#include "roclapack_getrf.hpp"
#include "roclapack_getrf_nopiv.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, typename I, typename U>
rocblas_status rocsolver_getrf_strided_batched_impl(rocblas_handle handle,
                                                    const I m,
                                                    const I n,
                                                    U A,
                                                    const I lda,
                                                    const rocblas_stride strideA,
                                                    I* ipiv,
                                                    const rocblas_stride strideP,
                                                    I* info,
                                                    const bool pivot,
                                                    const I batch_count)
{
    const char* name = (pivot ? "getrf_strided_batched" : "getrf_npvt_strided_batched");
    ROCSOLVER_ENTER_TOP(name, "-m", m, "-n", n, "--lda", lda, "--strideA", strideA, "--strideP",
                        strideP, "--batch_count", batch_count);

    using S = decltype(std::real(T{}));

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st
        = rocsolver_getf2_getrf_argCheck(handle, m, n, lda, A, ipiv, info, pivot, batch_count);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_stride shiftA = 0;
    rocblas_stride shiftP = 0;

    // strided batched execution
    I inca = 1;

    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // size of reusable workspace (and for calling TRSM)
    bool optim_mem;
    size_t size_work1, size_work2, size_work3, size_work4;
    // extra requirements for calling GETF2
    size_t size_pivotval, size_pivotidx;
    // size to store info about singularity of each subblock
    size_t size_iinfo, size_iipiv;

    rocsolver_getrf_getMemorySize<false, true, T>(
        m, n, pivot, batch_count, &size_scalars, &size_work1, &size_work2, &size_work3, &size_work4,
        &size_pivotval, &size_pivotidx, &size_iipiv, &size_iinfo, &optim_mem, lda);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_scalars, size_work1, size_work2,
                                                      size_work3, size_work4, size_pivotval,
                                                      size_pivotidx, size_iipiv, size_iinfo);

    // memory workspace allocation
    void *scalars, *work1, *work2, *work3, *work4, *pivotval, *pivotidx, *iinfo, *iipiv;
    rocblas_device_malloc mem(handle, size_scalars, size_work1, size_work2, size_work3, size_work4,
                              size_pivotval, size_pivotidx, size_iipiv, size_iinfo);

    if(!mem)
        return rocblas_status_memory_error;

    scalars = mem[0];
    work1 = mem[1];
    work2 = mem[2];
    work3 = mem[3];
    work4 = mem[4];
    pivotval = mem[5];
    pivotidx = mem[6];
    iipiv = mem[7];
    iinfo = mem[8];
    if(size_scalars > 0)
        init_scalars(handle, (T*)scalars);

    // execution
    return rocsolver_getrf_template<false, true, T>(
        handle, m, n, A, shiftA, inca, lda, strideA, ipiv, shiftP, strideP, info, batch_count,
        (T*)scalars, work1, work2, work3, work4, (T*)pivotval, (I*)pivotidx, (I*)iipiv, (I*)iinfo,
        optim_mem, pivot);
}

template <typename T, typename U>
rocblas_status rocsolver_getrf_nopiv_strided_batched_impl(rocblas_handle handle,
                                                          const rocblas_int m,
                                                          const rocblas_int n,
                                                          U A,
                                                          const rocblas_int lda,
                                                          const rocblas_stride strideA,
                                                          rocblas_int* info,
                                                          const rocblas_int batch_count)
{
    const char* name = "getrf_nopiv_strided_batched";
    ROCSOLVER_ENTER_TOP(name, "-m", m, "-n", n, "--lda", lda, "--strideA", strideA, "--batch_count",
                        batch_count);

    using S = decltype(std::real(T{}));

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    {
        rocblas_int* const ipiv = nullptr;
        bool const pivot = false;

        rocblas_status st
            = rocsolver_getf2_getrf_argCheck(handle, m, n, lda, A, ipiv, info, pivot, batch_count);
        if(st != rocblas_status_continue)
            return st;
    }

    // working with unshifted arrays
    rocblas_int shiftA = 0;

    // memory workspace sizes:
    // size of reusable workspace (and for calling TRSM)
    bool optim_mem = true;
    size_t size_work1 = 0;
    size_t size_work2 = 0;
    size_t size_work3 = 0;
    size_t size_work4 = 0;

    // size to store info about singularity of each subblock
    size_t size_iinfo = 0;

    constexpr bool is_batched = false;
    constexpr bool is_strided = true;
    rocsolver_getrf_nopiv_getMemorySize<is_batched, is_strided, T>(
        m, n, batch_count, &size_work1, &size_work2, &size_work3, &size_work4, &size_iinfo,
        &optim_mem);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_work1, size_work2, size_work3,
                                                      size_work4, size_iinfo);

    // memory workspace allocation
    void* work1 = nullptr;
    void* work2 = nullptr;
    void* work3 = nullptr;
    void* work4 = nullptr;
    void* iinfo = nullptr;
    rocblas_device_malloc mem(handle, size_work1, size_work2, size_work3, size_work4, size_iinfo);

    if(!mem)
        return rocblas_status_memory_error;

    work1 = mem[0];
    work2 = mem[1];
    work3 = mem[2];
    work4 = mem[3];
    iinfo = mem[4];

    // execution
    return rocsolver_getrf_nopiv_template<is_batched, is_strided, T>(
        handle, m, n, A, shiftA, lda, strideA, info, batch_count, work1, work2, work3, work4,
        (rocblas_int*)iinfo, optim_mem);
}
ROCSOLVER_END_NAMESPACE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_sgetrf_strided_batched(rocblas_handle handle,
                                                const rocblas_int m,
                                                const rocblas_int n,
                                                float* A,
                                                const rocblas_int lda,
                                                const rocblas_stride strideA,
                                                rocblas_int* ipiv,
                                                const rocblas_stride strideP,
                                                rocblas_int* info,
                                                const rocblas_int batch_count)
{
    return rocsolver::rocsolver_getrf_strided_batched_impl<float>(
        handle, m, n, A, lda, strideA, ipiv, strideP, info, true, batch_count);
}

rocblas_status rocsolver_dgetrf_strided_batched(rocblas_handle handle,
                                                const rocblas_int m,
                                                const rocblas_int n,
                                                double* A,
                                                const rocblas_int lda,
                                                const rocblas_stride strideA,
                                                rocblas_int* ipiv,
                                                const rocblas_stride strideP,
                                                rocblas_int* info,
                                                const rocblas_int batch_count)
{
    return rocsolver::rocsolver_getrf_strided_batched_impl<double>(
        handle, m, n, A, lda, strideA, ipiv, strideP, info, true, batch_count);
}

rocblas_status rocsolver_cgetrf_strided_batched(rocblas_handle handle,
                                                const rocblas_int m,
                                                const rocblas_int n,
                                                rocblas_float_complex* A,
                                                const rocblas_int lda,
                                                const rocblas_stride strideA,
                                                rocblas_int* ipiv,
                                                const rocblas_stride strideP,
                                                rocblas_int* info,
                                                const rocblas_int batch_count)
{
    return rocsolver::rocsolver_getrf_strided_batched_impl<rocblas_float_complex>(
        handle, m, n, A, lda, strideA, ipiv, strideP, info, true, batch_count);
}

rocblas_status rocsolver_zgetrf_strided_batched(rocblas_handle handle,
                                                const rocblas_int m,
                                                const rocblas_int n,
                                                rocblas_double_complex* A,
                                                const rocblas_int lda,
                                                const rocblas_stride strideA,
                                                rocblas_int* ipiv,
                                                const rocblas_stride strideP,
                                                rocblas_int* info,
                                                const rocblas_int batch_count)
{
    return rocsolver::rocsolver_getrf_strided_batched_impl<rocblas_double_complex>(
        handle, m, n, A, lda, strideA, ipiv, strideP, info, true, batch_count);
}

constexpr static bool use_getrf_nopiv = true;
rocblas_status rocsolver_sgetrf_strided_batched_64(rocblas_handle handle,
                                                   const int64_t m,
                                                   const int64_t n,
                                                   float* A,
                                                   const int64_t lda,
                                                   const rocblas_stride strideA,
                                                   int64_t* ipiv,
                                                   const rocblas_stride strideP,
                                                   int64_t* info,
                                                   const int64_t batch_count)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_getrf_strided_batched_impl<float>(
        handle, m, n, A, lda, strideA, ipiv, strideP, info, true, batch_count);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_dgetrf_strided_batched_64(rocblas_handle handle,
                                                   const int64_t m,
                                                   const int64_t n,
                                                   double* A,
                                                   const int64_t lda,
                                                   const rocblas_stride strideA,
                                                   int64_t* ipiv,
                                                   const rocblas_stride strideP,
                                                   int64_t* info,
                                                   const int64_t batch_count)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_getrf_strided_batched_impl<double>(
        handle, m, n, A, lda, strideA, ipiv, strideP, info, true, batch_count);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_cgetrf_strided_batched_64(rocblas_handle handle,
                                                   const int64_t m,
                                                   const int64_t n,
                                                   rocblas_float_complex* A,
                                                   const int64_t lda,
                                                   const rocblas_stride strideA,
                                                   int64_t* ipiv,
                                                   const rocblas_stride strideP,
                                                   int64_t* info,
                                                   const int64_t batch_count)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_getrf_strided_batched_impl<rocblas_float_complex>(
        handle, m, n, A, lda, strideA, ipiv, strideP, info, true, batch_count);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_zgetrf_strided_batched_64(rocblas_handle handle,
                                                   const int64_t m,
                                                   const int64_t n,
                                                   rocblas_double_complex* A,
                                                   const int64_t lda,
                                                   const rocblas_stride strideA,
                                                   int64_t* ipiv,
                                                   const rocblas_stride strideP,
                                                   int64_t* info,
                                                   const int64_t batch_count)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_getrf_strided_batched_impl<rocblas_double_complex>(
        handle, m, n, A, lda, strideA, ipiv, strideP, info, true, batch_count);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_sgetrf_npvt_strided_batched(rocblas_handle handle,
                                                     const rocblas_int m,
                                                     const rocblas_int n,
                                                     float* A,
                                                     const rocblas_int lda,
                                                     const rocblas_stride strideA,
                                                     rocblas_int* info,
                                                     const rocblas_int batch_count)
{
    if(use_getrf_nopiv)
    {
        return rocsolver::rocsolver_getrf_nopiv_strided_batched_impl<float>(
            handle, m, n, A, lda, strideA, info, batch_count);
    }
    else
    {
        rocblas_int* ipiv = nullptr;
        return rocsolver::rocsolver_getrf_strided_batched_impl<float>(
            handle, m, n, A, lda, strideA, ipiv, 0, info, false, batch_count);
    }
}

rocblas_status rocsolver_dgetrf_npvt_strided_batched(rocblas_handle handle,
                                                     const rocblas_int m,
                                                     const rocblas_int n,
                                                     double* A,
                                                     const rocblas_int lda,
                                                     const rocblas_stride strideA,
                                                     rocblas_int* info,
                                                     const rocblas_int batch_count)
{
    if(use_getrf_nopiv)
    {
        return rocsolver::rocsolver_getrf_nopiv_strided_batched_impl<double>(
            handle, m, n, A, lda, strideA, info, batch_count);
    }
    else
    {
        rocblas_int* ipiv = nullptr;
        return rocsolver::rocsolver_getrf_strided_batched_impl<double>(
            handle, m, n, A, lda, strideA, ipiv, 0, info, false, batch_count);
    }
}

rocblas_status rocsolver_cgetrf_npvt_strided_batched(rocblas_handle handle,
                                                     const rocblas_int m,
                                                     const rocblas_int n,
                                                     rocblas_float_complex* A,
                                                     const rocblas_int lda,
                                                     const rocblas_stride strideA,
                                                     rocblas_int* info,
                                                     const rocblas_int batch_count)
{
    if(use_getrf_nopiv)
    {
        return rocsolver::rocsolver_getrf_nopiv_strided_batched_impl<rocblas_float_complex>(
            handle, m, n, A, lda, strideA, info, batch_count);
    }
    else
    {
        rocblas_int* ipiv = nullptr;
        return rocsolver::rocsolver_getrf_strided_batched_impl<rocblas_float_complex>(
            handle, m, n, A, lda, strideA, ipiv, 0, info, false, batch_count);
    }
}

rocblas_status rocsolver_zgetrf_npvt_strided_batched(rocblas_handle handle,
                                                     const rocblas_int m,
                                                     const rocblas_int n,
                                                     rocblas_double_complex* A,
                                                     const rocblas_int lda,
                                                     const rocblas_stride strideA,
                                                     rocblas_int* info,
                                                     const rocblas_int batch_count)
{
    if(use_getrf_nopiv)
    {
        return rocsolver::rocsolver_getrf_nopiv_strided_batched_impl<rocblas_double_complex>(
            handle, m, n, A, lda, strideA, info, batch_count);
    }
    else
    {
        rocblas_int* ipiv = nullptr;
        return rocsolver::rocsolver_getrf_strided_batched_impl<rocblas_double_complex>(
            handle, m, n, A, lda, strideA, ipiv, 0, info, false, batch_count);
    }
}

rocblas_status rocsolver_sgetrf_npvt_strided_batched_64(rocblas_handle handle,
                                                        const int64_t m,
                                                        const int64_t n,
                                                        float* A,
                                                        const int64_t lda,
                                                        const rocblas_stride strideA,
                                                        int64_t* info,
                                                        const int64_t batch_count)
{
#ifdef HAVE_ROCBLAS_64
    int64_t* ipiv = nullptr;
    return rocsolver::rocsolver_getrf_strided_batched_impl<float>(handle, m, n, A, lda, strideA,
                                                                  ipiv, 0, info, false, batch_count);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_dgetrf_npvt_strided_batched_64(rocblas_handle handle,
                                                        const int64_t m,
                                                        const int64_t n,
                                                        double* A,
                                                        const int64_t lda,
                                                        const rocblas_stride strideA,
                                                        int64_t* info,
                                                        const int64_t batch_count)
{
#ifdef HAVE_ROCBLAS_64
    int64_t* ipiv = nullptr;
    return rocsolver::rocsolver_getrf_strided_batched_impl<double>(
        handle, m, n, A, lda, strideA, ipiv, 0, info, false, batch_count);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_cgetrf_npvt_strided_batched_64(rocblas_handle handle,
                                                        const int64_t m,
                                                        const int64_t n,
                                                        rocblas_float_complex* A,
                                                        const int64_t lda,
                                                        const rocblas_stride strideA,
                                                        int64_t* info,
                                                        const int64_t batch_count)
{
#ifdef HAVE_ROCBLAS_64
    int64_t* ipiv = nullptr;
    return rocsolver::rocsolver_getrf_strided_batched_impl<rocblas_float_complex>(
        handle, m, n, A, lda, strideA, ipiv, 0, info, false, batch_count);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_zgetrf_npvt_strided_batched_64(rocblas_handle handle,
                                                        const int64_t m,
                                                        const int64_t n,
                                                        rocblas_double_complex* A,
                                                        const int64_t lda,
                                                        const rocblas_stride strideA,
                                                        int64_t* info,
                                                        const int64_t batch_count)
{
#ifdef HAVE_ROCBLAS_64
    int64_t* ipiv = nullptr;
    return rocsolver::rocsolver_getrf_strided_batched_impl<rocblas_double_complex>(
        handle, m, n, A, lda, strideA, ipiv, 0, info, false, batch_count);
#else
    return rocblas_status_not_implemented;
#endif
}

} // extern C
