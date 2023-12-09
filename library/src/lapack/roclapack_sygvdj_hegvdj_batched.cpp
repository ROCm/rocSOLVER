/* **************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "roclapack_sygvdj_hegvdj.hpp"

template <typename T, typename S, typename U>
rocblas_status rocsolver_sygvdj_hegvdj_batched_impl(rocblas_handle handle,
                                                    const rocblas_eform itype,
                                                    const rocblas_evect evect,
                                                    const rocblas_fill uplo,
                                                    const rocblas_int n,
                                                    U A,
                                                    const rocblas_int lda,
                                                    U B,
                                                    const rocblas_int ldb,
                                                    S* D,
                                                    const rocblas_stride strideD,
                                                    rocblas_int* info,
                                                    const rocblas_int batch_count)
{
    const char* name = (!rocblas_is_complex<T> ? "sygvdj_batched" : "hegvdj_batched");
    ROCSOLVER_ENTER_TOP(name, "--itype", itype, "--evect", evect, "--uplo", uplo, "-n", n, "--lda",
                        lda, "--ldb", ldb, "--strideD", strideD, "--batch_count", batch_count);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_sygvdj_hegvdj_argCheck(handle, itype, evect, uplo, n, lda, ldb, A,
                                                         B, D, info, batch_count);
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
    // size of reusable workspaces
    bool optim_mem;
    size_t size_work1, size_work2, size_work3, size_work4;
    // extra requirements for calling SYEVDJ/HEEVDJ
    size_t size_workE;
    size_t size_workTau;
    size_t size_workVec;
    size_t size_workSplits;
    size_t size_workArr;
    // size of temporary info array
    size_t size_iinfo;

    rocsolver_sygvdj_hegvdj_getMemorySize<true, false, T, S>(
        itype, evect, uplo, n, batch_count, &size_scalars, &size_work1, &size_work2, &size_work3,
        &size_work4, &size_workE, &size_workTau, &size_workVec, &size_workSplits, &size_iinfo,
        &size_workArr, &optim_mem);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(
            handle, size_scalars, size_work1, size_work2, size_work3, size_work4, size_workE,
            size_workTau, size_workVec, size_workSplits, size_iinfo, size_workArr);

    // memory workspace allocation
    void *scalars, *work1, *work2, *work3, *work4, *workE, *workTau, *workVec, *workSplits,
        *workArr, *iinfo;
    rocblas_device_malloc mem(handle, size_scalars, size_work1, size_work2, size_work3, size_work4,
                              size_workE, size_workTau, size_workVec, size_workSplits, size_iinfo,
                              size_workArr);

    if(!mem)
        return rocblas_status_memory_error;

    scalars = mem[0];
    work1 = mem[1];
    work2 = mem[2];
    work3 = mem[3];
    work4 = mem[4];
    workE = mem[5];
    workTau = mem[6];
    workVec = mem[7];
    workSplits = mem[8];
    iinfo = mem[9];
    workArr = mem[10];
    if(size_scalars > 0)
        init_scalars(handle, (T*)scalars);

    // execution
    return rocsolver_sygvdj_hegvdj_template<true, false, T>(
        handle, itype, evect, uplo, n, A, shiftA, lda, strideA, B, shiftB, ldb, strideB, D, strideD,
        info, batch_count, (T*)scalars, work1, work2, work3, work4, (S*)workE, (T*)workTau,
        (T*)workVec, (rocblas_int*)workSplits, (rocblas_int*)iinfo, workArr, optim_mem);
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_ssygvdj_batched(rocblas_handle handle,
                                         const rocblas_eform itype,
                                         const rocblas_evect evect,
                                         const rocblas_fill uplo,
                                         const rocblas_int n,
                                         float* const A[],
                                         const rocblas_int lda,
                                         float* const B[],
                                         const rocblas_int ldb,
                                         float* D,
                                         const rocblas_stride strideD,
                                         rocblas_int* info,
                                         const rocblas_int batch_count)
{
    return rocsolver_sygvdj_hegvdj_batched_impl<float>(handle, itype, evect, uplo, n, A, lda, B,
                                                       ldb, D, strideD, info, batch_count);
}

rocblas_status rocsolver_dsygvdj_batched(rocblas_handle handle,
                                         const rocblas_eform itype,
                                         const rocblas_evect evect,
                                         const rocblas_fill uplo,
                                         const rocblas_int n,
                                         double* const A[],
                                         const rocblas_int lda,
                                         double* const B[],
                                         const rocblas_int ldb,
                                         double* D,
                                         const rocblas_stride strideD,
                                         rocblas_int* info,
                                         const rocblas_int batch_count)
{
    return rocsolver_sygvdj_hegvdj_batched_impl<double>(handle, itype, evect, uplo, n, A, lda, B,
                                                        ldb, D, strideD, info, batch_count);
}

rocblas_status rocsolver_chegvdj_batched(rocblas_handle handle,
                                         const rocblas_eform itype,
                                         const rocblas_evect evect,
                                         const rocblas_fill uplo,
                                         const rocblas_int n,
                                         rocblas_float_complex* const A[],
                                         const rocblas_int lda,
                                         rocblas_float_complex* const B[],
                                         const rocblas_int ldb,
                                         float* D,
                                         const rocblas_stride strideD,
                                         rocblas_int* info,
                                         const rocblas_int batch_count)
{
    return rocsolver_sygvdj_hegvdj_batched_impl<rocblas_float_complex>(
        handle, itype, evect, uplo, n, A, lda, B, ldb, D, strideD, info, batch_count);
}

rocblas_status rocsolver_zhegvdj_batched(rocblas_handle handle,
                                         const rocblas_eform itype,
                                         const rocblas_evect evect,
                                         const rocblas_fill uplo,
                                         const rocblas_int n,
                                         rocblas_double_complex* const A[],
                                         const rocblas_int lda,
                                         rocblas_double_complex* const B[],
                                         const rocblas_int ldb,
                                         double* D,
                                         const rocblas_stride strideD,
                                         rocblas_int* info,
                                         const rocblas_int batch_count)
{
    return rocsolver_sygvdj_hegvdj_batched_impl<rocblas_double_complex>(
        handle, itype, evect, uplo, n, A, lda, B, ldb, D, strideD, info, batch_count);
}

} // extern C
